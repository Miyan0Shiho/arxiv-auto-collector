# Statistical Detection of Positional Bias in LLM-Based Recommender Systems: Methodologies and Empirical Evidence

**Authors**: Carlos Peláez-González, Francisco Herrera, Andrés Herrera-Poyatos

**Published**: 2026-10-27 20:48:30

**PDF URL**: [https://arxiv.org/pdf/2610.37962v1](https://arxiv.org/pdf/2610.37962v1)

## Abstract
Large Language Models (LLMs) are increasingly used as recommendation generators in conversational recommender systems (RS) via retrieval-augmented generation (RAG). This flexible design introduces new sources of structural bias. This paper studies positional bias in top-1 and single-turn LLM-based RS: systematic dependence of the recommended item on the order in which candidates are presented in the prompt. We propose two complementary detection methodologies: (i) POB-ICM, based on contingency matrix analysis and Chi-square independence testing, and (ii) POB-UNBP, a sample-efficient approach derived from a uniformity null model under permutation-based evaluation, enabling stochastic decoding with formal statistical guarantees. Using a controlled RAG testbed for product recommendation, built on a real product catalog, a fixed prompt template and 2000 template-generated user queries with 20 permutations each (40,000 instances), both methodologies reject the null hypothesis of positional independence and uniformity in all evaluated state-of-the-art models. We introduce three complementary metrics -- permutation sensitivity, positional bias, and strong positional bias -- to quantify this effect, the latter targeting cases where the position displaces the recommendation towards clearly less relevant items rather than only choosing among comparably relevant ones. A robustness analysis across candidate-set size and decoding temperature shows that the bias is structural and model-dependent, not an artifact of deterministic decoding or candidate-set size. These results establish positional bias as a measurable, reproducible property of prompt-mediated recommendation pipelines in the evaluated setting and motivate routine auditing and governance controls (e.g., order randomization, standardized prompting) for deployed LLM-based RS. Source code is publicly released.

## Full Text


<!-- PDF content starts -->

STATISTICALDETECTION OFPOSITIONALBIAS IN
LLM-BASEDRECOMMENDERSYSTEMS: METHODOLOGIES
ANDEMPIRICALEVIDENCE
Carlos Peláez-González∗,1, Francisco Herrera1, Andrés Herrera-Poyatos2
1Department of Computer Science and Artificial Intelligence, Andalusian Institute of Data Science and
Computational Intelligence (DaSCI), University of Granada, Spain.
2Department of Algebra, University of Granada, Spain.∗Corresponding author.
Emails:{carlosprog, andreshp}@ugr.es,herrera@decsai.ugr.es
October 30, 2026
ABSTRACT
Large Language Models (LLMs) are increasingly used as recommendation generators in conversa-
tional recommender systems (RS) via retrieval-augmented generation (RAG). This flexible design
introduces new sources of structural bias. This paper studiespositional biasin top-1 and single-turn
LLM-based RS: systematic dependence of the recommended item on the order in which candidates
are presented in the prompt. We propose two complementary detection methodologies: (i)POB-
ICM, based on contingency matrix analysis and Chi-square independence testing, and (ii)POB-
UNBP, a sample-efficient approach derived from a uniformity null model under permutation-based
evaluation, enabling stochastic decoding with formal statistical guarantees. Using a controlled RAG
testbed for product recommendation, built on a real product catalog, a fixed prompt template and
2000 template-generated user queries with 20 permutations each (40,000 instances), both method-
ologies reject the null hypothesis of positional independence and uniformity in all evaluated state-
of-the-art models. We introduce three complementary metrics — permutation sensitivity, positional
bias, and strong positional bias — to quantify this effect, the latter targeting cases where the position
displaces the recommendation towards clearly less relevant items rather than only choosing among
comparably relevant ones. A robustness analysis across candidate-set size and decoding temperature
shows that the bias is structural and model-dependent, not an artifact of deterministic decoding or
candidate-set size. These results establish positional bias as a measurable, reproducible property of
prompt-mediated recommendation pipelines in the evaluated setting and motivate routine auditing
and governance controls (e.g., order randomization, standardized prompting) for deployed LLM-
based RS. Source code is publicly released1.
KeywordsLarge language models·conversational recommender system·positional bias
1 Introduction
Recommender systems (RS) are a core component of modern digital platforms, supporting users in navigating and
filtering large volumes of information. By modeling user preferences and item characteristics, RS aims to deliver rel-
evant and personalized recommendations across domains such as e-commerce, media consumption, and information
retrieval. Despite substantial progress, RS remain susceptible to a range of biases that can affect both the reliability
and the consistency of their outputs. Previous work has documented several forms of bias—including exposure bias,
popularity bias, and positional bias—that can emerge at different stages of the recommendation pipeline and prop-
agate through system components, ultimately shaping the user experience in unintended ways [3]. Understanding,
1https://github.com/GoComputing/Positional-Bias.git
arXiv:2610.37962v1  [cs.IR]  27 Oct 2026

Statistical Detection of Positional Bias in LLM-Based Recommender SystemsA PREPRINT
measuring, and mitigating such biases is, therefore, essential for ensuring transparent and reliable recommendation
behavior.
In parallel, Large Language Models (LLMs) have recently emerged as a powerful paradigm for building intelligent
systems. Trained in large-scale corpora using self-supervised objectives, LLMs exhibit a wide range of emergent
capabilities, including question answering, reasoning, and decision-making [19]. This has led to a shift from traditional
model training toward prompt-based interaction, where system behavior is largely determined by how information is
presented in natural language form. As a consequence, LLMs are increasingly being adopted as central components
in RS, particularly in conversational settings.
A common design for LLM-based RS is based on a retrieval-augmented generation (RAG) architecture [5]. In this
setting, a retriever first computes semantic similarity scores between a user query and a collection of candidate items
using an embedding model. The top-kmost relevant items are then selected and inserted into a prompt, together with
the user query, which is provided to the LLM. Based on this prompt, the LLM produces a recommendation, typically
by selecting or describing one of the candidate items. This architecture decouples retrieval from generation while
allowing the LLM to reason over a constrained candidate set.
The increasing adoption of LLMs within information systems has been identified as a major research frontier in the
Information Systems (IS) community. Recent work emphasizes that generative AI systems introduce new forms of
socio-technical risk, requiring systematic analysis of their behavioral properties, reliability, and governance when
deployed in real-world organizational settings [21]. In this context, understanding how design choices in prompt-
mediated systems affect model behavior is critical to advance both theory and practice in LLM-enabled information
systems.
However, LLM-based RS operate largely as black boxes. Although the retriever stage is often well understood and
explicitly optimized, the final recommendation decision is made by the LLM through implicit reasoning over a sequen-
tial text prompt. As a result, unintended behaviors may arise that are not directly related to user preferences or item
relevance. These behaviors are not necessarily associated with socially sensitive attributes such as gender or race but
instead stem from subtle interactions between prompt structure and model internals. Ideally, a recommendation should
depend only on the semantic relationship between the user query and the candidate items. In practice, additional latent
variables may influence the outcome.
One such variable is the position of candidate items within the prompt. Because text is inherently sequential, the
retrieved items must be presented in a specific order. This ordering introduces positional information that is not
semantically meaningful, but can nevertheless affect the LLM’s output. When the probability of recommending an
item depends on its position in the input list rather than solely on its relevance, the system exhibits positional bias.
Although positional bias has been extensively studied in traditional RS and user interfaces, its manifestation and
measurement in LLM-based RS remain unstudied.
In this work, we focus on the measurement and statistical characterization of positional bias in top-1 and single-turn
LLM-based RS, where the model recommends a single item from the retrieved candidates in one conversational turn.
Rather than proposing mitigation strategies, our objective is to rigorously determine whether positional bias is present,
how it can be detected, and how strong its effects are under controlled experimental conditions. To structure this
investigation, we formalize the following research questions:
RQ1: Does output selection in LLM-based RS depend on the position of candidate items within the input prompt?
RQ2: Is any observed dependence between item position and model output statistically significant under
permutation-based testing?
RQ3: How large is the positional bias, and does its magnitude vary across models?
To address these questions, we propose two complementary statistical methodologies for positional bias detection. The
first, Positional Bias detection based on Independence of the Contingency Matrix (POB-ICM), models the relationship
between input positions and output selections as a contingency table and evaluates whether these variables are statis-
tically independent. Under the null hypothesis of no positional bias, the distribution of recommended items should be
independent of their input positions. Violations of this assumption are assessed using a query-level permutation test of
independence that allows both detection and interpretation of positional bias.
The second methodology, Positional Bias detection based on Uniformity assuming Non-Biased Positions (POB-
UNBP), is designed to reduce sample complexity by leveraging a stronger null assumption. Specifically, in the absence
of positional bias, on expectation over all the possible queries, the probability of recommending the item at a given
position is uniform. By repeatedly permuting the order of candidate items and analyzing the resulting recommendation
distributions, this method enables efficient detection of systematic positional preferences while requiring fewer sam-
2

Statistical Detection of Positional Bias in LLM-Based Recommender SystemsA PREPRINT
ples than full contingency-based testing. Together, these methodologies provide a principled and statistically grounded
framework for evaluating positional bias in LLM-based RS.
Using these approaches, we conducted an empirical study on state-of-the-art LLMs embedded within a RAG-based RS
for product recommendation, built on a real product catalog with template-generated user queries and a fixed prompt
template. Our experiments reveal statistically significant positional bias, demonstrating that item order can systemati-
cally influence recommendation outcomes even when item relevance is kept constant. We complement these statistical
tests with three proposed metrics. Permutation sensitivity metric evaluates how the model changes its decision on
permutation changes, so it is an upper bound of positional bias. Positional bias metric measures the total variation
distance between the uniform distribution and the distribution of positions of items selected by the RS, as deviation
from the uniform distribution indicates a systematic preference for certain positions. Finally, the strong positional
bias metric builds on relevance sets, the query-dependent sets of candidate items that the model treats as comparably
relevant. Relevance sets decompose the overall positional bias detected by our methodologies into soft positional bias,
where the position determines which item of the relevance set is recommended, and strong positional bias, where the
position displaces the recommendation outside the relevance set, which this metric quantifies. In summary, the main
contributions of this paper are as follows:
1. We introduce POB-ICM, a statistically grounded methodology for detecting and interpreting overall posi-
tional bias in LLM-based RS based on contingency matrix independence.
2. We propose POB-UNBP, a sample-efficient positional bias detection methodology based on uniformity as-
sumptions under the absence of positional bias, which allows for stochastic decoding, and complement it with
two quantification metrics (permutation sensitivity and positional bias) that disentangle generic permutation
sensitivity from structural positional preference.
3. We decompose overall positional bias into soft and strong positional bias through query-dependent relevance
sets, quantify the strong component, and show through counterfactual analyses that it is not explained by the
identifiers assigned to the items and, for six of the eight evaluated models, not by stochastic decoding alone.
The formal detection of each component separately is left as future work.
4. We implement and evaluate an end-to-end experimental framework to assess positional bias in LLM-based RS
using a RAG architecture in a single-turn product recommendation setting, including a systematic robustness
analysis across candidate-set sizekand decoding temperatureton eight state-of-the-art LLMs, showing that
positional bias is a structural, model-dependent property rather than an artifact of deterministic decoding or
candidate-set size.
The empirical results indicate that positional bias is a systematic and statistically detectable factor in LLM-based
recommender systems implemented under an RAG pipeline, within the evaluated setting. Across permutation-based
evaluations, both proposed methodologies (POB-ICM and POB-UNBP) reject the null hypothesis that recommen-
dation outcomes are independent of the positional placement of candidate items within the prompt in all evaluated
models, and the resulting selection distributions exhibit structured, non-uniform positional bias with certain prompt
positions associated with reliably elevated selection probabilities.
The remainder of this paper is organized as follows. In section 2 we review related work on positional bias and
related phenomena in RS and language models. In section 3 we describe the system design, formal modeling, and the
proposed POB-ICM and POB-UNBP methodologies. The experimental setup and empirical results are presented in
section 4. Finally, section 5 discusses the implications of our findings, limitations, and directions for future work.
2 Related Work
Conversational RS generate personalized recommendations by interpreting user preferences expressed in natural lan-
guage. Early conversational RS relied on rule-based or traditional machine learning approaches, while more recent sys-
tems increasingly incorporate deep learning models. With the advent of Large Language Models (LLMs), recommen-
dation pipelines have undergone a substantial shift, as LLMs enable direct reasoning over textual user queries and item
descriptions. Previous work has demonstrated the applicability of LLMs to a variety of recommendation tasks, includ-
ing rating prediction, top-Nrecommendation, and conversational recommendation scenarios [12]. These systems can
generate recommendations as item identifiers or as natural-language responses, leveraging the linguistic and reasoning
capabilities of LLMs.
A common instantiation of LLM-based RS adopts a retrieval-augmented generation (RAG) architecture. In this set-
ting, a retriever selects a set ofKcandidate items from a larger database based on semantic similarity to the user
query, and these candidates are subsequently provided to the LLM for recommendation generation [22]. In the top-N
3

Statistical Detection of Positional Bias in LLM-Based Recommender SystemsA PREPRINT
recommendation, the model selects or ranks multiple items (withK > N), while in the special case of the Top-1
recommendation, the LLM is required to select a single item from the candidate set. The latter setting is particularly
relevant for conversational interfaces, where concise recommendations are often preferred, and is the setting studied
in this work, in which a single item is recommended in a single conversational turn.
Bias in RS has been extensively studied and is commonly categorized asunfair biasesandstructural biases. Unfair
biases involve systematic disparities that affect sensitive user groups, such as those defined by gender, race, or age [6,
15]. In contrast, structural biases arise from the design of the system itself, including data preprocessing choices,
model architecture, or prompt formulation. In LLM-based RS, structural biases can emerge from seemingly innocuous
prompt design decisions, such as wording or synonym selection, which can influence the model output independently
of user intent [6].
Beyond recommender systems, structural bias in LLMs has been studied across several application domains. We define
positional bias as a special case of structural bias in which a model’s output depends systematically on the sequential
position of an input element such as a candidate item, an answer option, or a retrieved passage, rather than solely on
its semantic content. Because language models process text as ordered sequences, any system that presents multiple
elements in a list implicitly encodes positional information that the model may exploit, independently of relevance or
quality. In the context ofmultiple-choice question answering, it has been shown that the order in which choices are
listed can affect model answers [17]. LLMs also exhibit selection bias by assigning more probability mass to specific
option ID tokens regardless of content [28]. Additional studies have demonstrated that even the symbols used to
delimit answer options can introduce systematic biases in model output [26]. Inlong-context question answering and
retrieval-augmented generation, [13] demonstrated that LLM performance follows a U-shaped curve as a function of
where relevant information appears in the context. They named this phenomenon as “lost in the middle” effect, with
[11] tracing this degradation to intrinsic U-shaped attention patterns in transformer models independent of content
relevance. In the domain ofLLM-as-a-Judge evaluation, [29] documented that strong LLMs used as evaluators exhibit
positional bias when judging response quality, a finding confirmed at scale by [20] across 15 judge models and over
150,000 evaluation instances. InLLM-based recommender systems, [10] observed that recommendation performance
deteriorates when highly relevant items appear in later positions of the candidate list. These findings provide empirical
indicators of positional bias in different areas. However, across all these domains, these characterize positional bias
through performance degradation metrics and does not provide a hypothesis-testing framework that formally isolates
position as a statistically independent variable.
Top-1 RS based on LLM can be naturally framed as a multiple-choice decision problem: a set ofKcandidate items
is presented to the model, and the LLM must select the item that best satisfies the user’s preferences. However, unlike
standard multiple-choice tasks, recommendation scenarios often admit multiple plausible answers rather than a single
correct option. This distinction complicates the direct transfer of positional bias analysis from question answering to
recommendation settings, and similarly limits the applicability of long-context QA approaches such as the lost-in-the-
middle framework [13], which rely on a ground-truth answer to measure positional bias and do not isolate position as a
statistically independent variable. Moreover, unlike ranking-based formulations of LLM-based recommendation [10],
where the model orders all the candidates, a top-1 recommender only reveals the item it selects, so the relevance it
assigns to the remaining candidates must be inferred indirectly, which makes it harder to separate positional effects
from item relevance. There are nevertheless strong reasons to hypothesize that positional bias affects LLM-based
RS. Training data for LLMs include numerous instances of recommendations and ranked lists, which may themselves
reflect user interaction biases, such as a tendency to favor items appearing earlier in lists [4]. Previous surveys have
warned that such biases can propagate into learned recommendation behaviors [3, 24]. Empirical observations further
suggest that the performance of LLM-based RS deteriorates when highly relevant items appear in later positions within
the candidate set [10], though these studies do not isolate positional bias from confounders such as retrieval quality or
item relevance.
Despite these observations and intuition, the closest existing approaches to statistically grounded quantification of po-
sitional bias in LLMs rely on scalar summary metrics rather than formal hypothesis testing. [28] propose the standard
deviation of per-option recall (RStd) as a measure of selection bias in MCQ tasks. [14] define positional bias as the
divergence between the marginalized selection distribution and the uniform distribution, implicitly assuming, as we
formally prove in subsection 3.4, that absence of positional bias implies uniformity. [23] introduce the Fluctuation
Rate, a label-free metric defined as the fraction of instances where a fixed forward ordering and a single fixed back-
ward ordering yield different answers; because it compares exactly two predetermined permutations rather than the full
permutation space, it may not capture the full extent of positional bias. Crucially, none of these approaches frames de-
tection as a hypothesis test or provides statistical significance guarantees, and all are developed for MCQ tasks where
a unique correct answer exists. Moreover, positional bias analysis developed for multiple-choice question answering
does not directly transfer to recommendation settings, where multiple candidates may be plausibly relevant and a single
ground-truth label may not exist. To address these gaps, we develop two complementary hypothesis-testing method-
4

Statistical Detection of Positional Bias in LLM-Based Recommender SystemsA PREPRINT
Figure 1: LLM-based RS architecture used in this paper.
ologies tailored to single-turn Top-1 LLM-based recommendation under controlled permutations:POB-ICM, which
detects and interprets positional dependence via contingency-matrix independence testing, andPOB-UNBP, which
provides a sample-efficient alternative via a uniformity-based null model with formal statistical guarantees. Further-
more, since several candidates may be comparably relevant, we distinguish soft positional bias, where the position
only determines which of these candidates is recommended, from strong positional bias, where the position displaces
the recommendation towards clearly less relevant items, and, besides the overall positional bias, we quantify its strong
component. Together, these methods enable principled detection and quantification of positional bias independently
of item quality and retrieval performance.
3 Methodology for positional bias detection in LLM-based RS
LLM-based RS expose candidate items to the model as an ordered list within a text prompt. Let{p(1), p(2), . . . , p(k)}
denote the marginal probability distribution over prompt positions, wherep(i)is the probability that the item appearing
at positioniis recommended, averaged over queries and random permutations of the candidate set (formally defined
in subsection 3.2). In the absence of positional bias, a model should recommend each item based solely on its relevance
to the user query, regardless of its position in the prompt. As we formally prove in subsection 3.4, this condition
implies thatp(i)follows a uniform distribution: when items are presented in random order, no prompt position is
systematically preferred, and thusp(1) =p(2) =···=p(k) = 1/k.
To detect violations of this property, we propose two complementary statistical methodologies, POB-ICM and POB-
UNBP, which trade off interpretability against sample efficiency. We first describe the RAG-based RS testbed used as
our experimental framework in subsection 3.1, establish the formal probabilistic notation in subsection 3.2, and then
introduce each methodology in subsection 3.3 and subsection 3.4 respectively.
3.1 An implementation of an LLM-based RS for positional bias analysis
To empirically study positional bias under controlled conditions, we implement an LLM-based RS following a RAG
architecture. This implementation serves as a testbed that allows systematic manipulation of item order while keeping
all other components fixed.
The implemented RS based on LLM, illustrated in Figure 1, follows an RAG architecture. It takes as input a user
query and is used to semantically match several products from a database, sorting them using the distance between the
query and the product titles. Then, this query, the titles and the descriptions of the products are fed to an LLM used
to recommend one of these products. As a result, the RS generates a recommendation selecting the position of one of
the related products. The specific steps are described below.
1. An embedding is computed using a pre-trained model, namely, the Beijing Academy of Artificial Intelligence
(BAAI) general embedding model [25].
2. The cosine distance is computed along each precomputed embedding from a product database, specifically
from the products’ titles. These products are sorted using the cosine distance, as it represents the semantic
similarity of the products with the user query.
3. The top-kproducts are selected, wherekis an hyperparameter of the system.
4. The selected products (titles and descriptions) and the user query are used to construct a prompt, following
the template illustrated in Prompt 1. This prompt asks the model to generate a recommendation given the
products and the user query. A JSON format is required to automatically parse the answer. The response is a
recommendation for a specific product, and the chatbot generates a conversation for the user.
5

Statistical Detection of Positional Bias in LLM-Based Recommender SystemsA PREPRINT
Prompt 1: Recommender system template prompt
You are a shop assistant that recommends the user the given shop articles based on his/her preferences.
Here are some articles in my shop:
Title 1: {title}
Description 1: {description}
...
Titlek: {title}
Descriptionk: {description}
The user query is {query}. Based on the user query and the provided articles, recommend one of these articles in
a chat way. Think step by step why you choose your selected article.
At the end, please provide your answer using a JSON format with the fields "article_number" (int), "article_title"
(str) and "recommendation" (str). Provide a single JSON as described.
The above implementation defines an end-to-end mapping from a user query and a set of retrieved items to a single
recommended item. To enable a rigorous statistical analysis of positional bias, we next formalize this pipeline as
a function that maps a query instance and an item permutation to a selected output position. This formalization
allows us to define explicit null hypotheses and apply statistical tests that isolate the effect of item position from other
components of the system.
3.2 Mathematical notation and probability distributions of the LLM-based RS
LetS kbe the set of all permutations of{1,2, . . . , k}. LetDbe a probability distribution over possible user queries
for the recommendation system and the associated top-kproducts, that is, we sample fromDtuples of the form
(Q, P 1, P2, . . . , P k), whereQis a user query andP 1, P2, . . . , P kare the top-kproducts selected within the RAG
architecture for queryQ. For an instanceX∼ Dand a permutationσ∈S k, we use Prompt 1 with the items listed
in the orderP σ(1), Pσ(2), . . . , P σ(k), obtaining a recommended item by the LLM. We denote byR(X, σ)the position
(in the prompt) of the item recommended. For example, ifR(X, σ) = 1, this means that the LLM recommends the
first item in the prompt, which corresponds toP σ(1). That is, in the original listP 1, P2, . . . , P kof top-kproducts, the
recommended item actually has indexσ(1).
In our analysis we sampleσfrom the uniform distribution overS k, denotedU(S k). The use of random permutations
enables the evaluation of positional bias on the recommendationR, consistent with the principles of permutation
testing [7]. For example, the probability that the LLM will recommend the first item to appear in the prompt is
p(i) = Pr X∼D,σ∼U(S k)[R(X, σ) =i] =E X∼D,σ∼U(S k)1[R(X, σ) =i],
where1(A)is1if the eventAholds, and1(A) = 0otherwise. Generally, the probability that the element at position
iin the prompt is the one recommended in our experiment is
p(i) =E X∼D,σ∼U(S k)1[R(X, σ) =i].(1)
Obviously, we havep(1) +p(2) +···+p(k) = 1. If the LLM-based RS does not suffer from positional bias, we
would havep(1) =p(2) =···=p(k) = 1/k, which will be exploited in one of our proposed methodologies.
We design two complementary methodologies to test the null hypothesis of positional independence. The first method-
ology, POB-ICM, is described in subsection 3.3 and evaluates dependence through a contingency matrix and statistical
independence tests. Although robust and interpretable, this approach requires a large number of samples. To address
this limitation, we propose a second methodology, POB-UNBP, described in subsection 3.4, which tests positional
bias through uniformity assumptions and requires substantially fewer samples, at the cost of reduced interpretability.
3.3 POB-ICM: POsitional Bias detection methodology based on the Independency of the Contingency Matrix
In the absence of positional bias, applying a permutation to the top-kitems in the LLM query should not statistically
affect the final recommendation. Concretely, the choice ofσ∼U(S k)should be independent of the final recommen-
dation. Building on this observation, we present a methodology that employs the Chi-square test for independence [1],
6

Statistical Detection of Positional Bias in LLM-Based Recommender SystemsA PREPRINT
whosep-value is calibrated with a query-level permutation test to account for the dependence among the permutations
evaluated on the same query. In addition to showing statistical significance, we assess the strength of the detected
association using three different metrics described in subsection 4.2. Implementing this test requires building a con-
tingency matrix for the following two random variables: the selected product and the random shuffling applied. First,
we formalize these random variables.
Recall that in our experiment we start with an instanceX= (Q, P 1, P2, . . . , P k)∼ D, chooseσ∈S kuniformly
at random and determineR(X, σ) =j, that is, the recommended product isP σ(j). We define the random variable
“original index”, denotedY, asσ(j). Note thatYis actually a function ofXandσ, so we can writeY=f(X, σ)
forX∼ Dandσ∼U(S k). Moreover,Yyields an integer between1andk. Lack of positional bias implies that the
random variablesYandσare conditionally independent givenX, that is,Pr[Y=i|σ, X] = Pr[Y=i|X]for all
i∈ {1,2, . . . , k}andσ∈S k.
We build the contingency table ofYandσas follows. For ease of notation, we assume that we have indexed all the
permutations inS k, obtaining a listσ 1, σ2, . . . , σ k!. Thei-th row in this table correspond toY=i, that is, the case
when the recommended product has positioniin the original order (before shuffling). Thej-th column corresponds
to the permutationσ j. LetX 1, X2, . . . , X tbe samples from the distributionD. Fori∈ {1,2, . . . , k}andσ j∈Sk,
thei, jentry of this contingency matrix is formally defined as
Ci,j=tX
ℓ=11[f(X ℓ, σj) =i] =tX
ℓ=11[σj(R(X ℓ, σj)) =i].(2)
That is,C i,jcounts how many instances(Q, P 1, P2, . . . , P k)of the dataset sampled fromDsatisfy that the recom-
mended item, when the prompt presents the items following the orderσ j, is the one in positioniin the original order
P1, P2, . . . , P k. Note that if there is no positional bias, then all columns of the matrixCshould be the same since the
item recommended is the same independently of the permutationσ, that is,σ(R(X, σ)) =φ(R(X, φ))for any two
permutationsφ, σ∈S k.
The null-hypothesis is that the variablesYandσare conditionally independent givenX. We compute the Pearson
Chi-square statisticχ2
obsof the contingency matrixC. SinceCaggregates several permutations of the same instance
Xℓ, its entries are not independent observations and the asymptotic Chi-square distribution may yield miscalibrated
p-values. Hence, we calibrate thep-value with a query-level (stratified) permutation test. Under the null-hypothesis,
the recommended itemYdoes not depend onσ, so, for each instanceX ℓ, the values ofYare exchangeable among the
permutations applied toX ℓ. We generateBcontingency matricesC(1), . . . , C(B)by randomly shuffling the values
ofYamong the permutations of each instance, never mixing values across different instances, and compute their
Chi-square statisticsχ2
(1), . . . , χ2
(B). This empirical distribution preserves both the query-level dependence and the
marginals ofC. The Monte Carlop-value isp= 
1 +PB
b=11[χ2
(b)≥χ2
obs]
/(1 +B), withB= 2000in our
experiments. If the null-hypothesis is rejected, there is a dependency between the product selection and the position
of the products, indicating a positional bias.
The generation of the contingency matrix described in Equation 2 requires the generation ofk!permutations for each
of thetsamples from the distributionD. In practice, a subset of permutations may be used, since the permutation
test above does not rely on the asymptotic Chi-square approximation and thus does not require a minimum number of
counts per cell. We can display the contingency matrix via a heatmap, which allows the user to inspect the positional
bias induced by the LLM, see Figure 4. However, this matrix might be overwhelming due to its size. To simplify this
visualization, we also use a matrix where each row represents the position of a given productbeforethe shuffling, and
the columns represent the position of this productafterthe shuffling. Then each celli, javerages the number of times
the productiwas selected whenσplaced it at positionj, that is,
Ai,j=1
ci,jX
X∈DX
σ∈Σ X:σ(j)=i1[R(X, σ) =j],(3)
whereΣ Xis the set of permutations evaluated for instanceX, andc i,jis the average number of permutations per
instance inΣ Xthat place the productiat positionj. If all possible permutations are evaluated,Σ X=S kand
ci,j= (k−1)!.
We could interpret the matrixA i,jby columns. Each column represents the number of times that a recommender LLM
selects a product in the positionjafter moving the products to this position. If a column is more likely to be selected,
this would mean that there is a positional bias with a positive tendency. In the same way, if a column is less likely to
be selected, then there is a negative positional bias towards this position.
7

Statistical Detection of Positional Bias in LLM-Based Recommender SystemsA PREPRINT
3.4 POB-UNBP: Positional Bias detection methodology based on Uniformity assuming Non-Biased Positions
The positional bias evaluation methodology described in the previous section requires many samples in order to be
robust. Thus, its evaluation on closed-weights models could be too expensive. In order to alleviate this, a second
methodology is proposed based on the uniformity of the probability distribution generated by the model. In this
section we work with the probability distributionp(i)given in Equation 1.
First, we mathematically prove that, under the assumption of absence of positional bias,p(i)follows a uniform distri-
bution. Secondly, we generalize this result to the stochastic setting, where the RS output is understood as a probability
distribution on the set of items in the prompt. Here we allow for a small tolerance of positional bias and show that,
despite this tolerance and the stochastic setting,p(i)is very close to the uniform distribution. As a consequence,
positional bias can be statistically tested by empirically sampling fromp(i)and checking if sampled data comes from
a probability distribution close to the uniform distribution.
3.4.1 Zero-temperature setting
We note that in the absence of positional bias, given an instanceX= (Q, P 1, P2, . . . , P k), for any two permutations
σ, φ∈S k, the recommended item is the same. Recall that we denote byR(X, σ)the position of the recommended
item within the prompt, and that this prompt contains the items in the orderP σ(1), Pσ(2), . . . , P σ(k). Hence, the
recommended item isP σ(R(X,σ)) , and, by assuming absence of positional bias, we haveP σ(R(X,σ)) =P φ(R(X,φ)) .
That is,σ(R(X, σ)) =φ(R(X, φ))for all permutationsσ, φ∈S k. Equivalently, there exists a mapgfrom instances
to{1,2, . . . , k}such thatσ(R(X, σ)) =g(X)for allσ∈S k.
Proposition 1.In the absence of positional bias,p(i|X=x) = 1/kfor alli∈ {1,2, . . . , k}and all instancesx.
Proof.Fix an instanceX=x. Sinceσ(R(x, σ)) =g(x)for allσ∈S k, we haveR(x, σ) =σ−1(g(x)). Therefore,
p(i|X=x) =E σ∼U(S k)1[R(x, σ) =i]
=Eσ∼U(S k)1[σ−1(g(x)) =i]
=Eσ∼U(S k)1[σ(i) =g(x)]
=1
k,
where the last equality holds because, forσdrawn uniformly fromS k, the valueσ(i)is uniformly distributed over
{1,2, . . . , k}.
Once we have determined that the probability distributionpdefined in Equation 1 should be close to a uniform
distribution under the absence of positional bias, we can evaluate positional bias by testing whetherpdeviates from
the uniform distribution. However, the notion of absence of positional bias used in Proposition 1 assumes that, for
each instance, a single item is recommended regardless of the permutation applied. This excludes both decoding at
non-zero temperature, where the recommended item may vary even for a fixed prompt, and the case in which several
candidates are comparably relevant and the model may spread its recommendations among them. Hence, we describe
the statistical test in the stochastic setting below, where absence of positional bias is defined on the distribution of
the recommended item, which may be spread over several items as long as it does not depend on the ordering, and is
further relaxed to tolerate small positional effects. This setting includes the deterministic one as a particular case. This
test requires fewer samples compared to the methodology proposed in subsection 3.3, as shown in Appendix B.
3.4.2 Stochastic setting
In certain settings, the LLM may operate with non-zero temperature, so items recommended will vary each time we
query the same recommendation problem. In such a setting, the LLM recommends itemP jwith some probability
Pr(P j|X, σ)that depends on the query and itemsX= (Q, P 1, P2, . . . , P k)and the orderingσof the items in the
prompt. Ideally, in the absence of positional bias, this probability distribution over items should not depend onσ,
that is, we should havePr(P j|X, σ) = Pr(P j|X, φ)for any permutationsσ, φ∈S k. However, in practice,
such definition seems difficult to fully attain by current LLM architectures. Therefore, we introduce the concept of
ε-absence of positional bias, which only requires the distributionsPr(· |X, σ)andPr(· |X, φ)to be reasonably
similar.
Definition 1.We say that there isε-absence of positional bias if, for all instancesX=xand all permutations
σ, φ∈S k,
TV(Pr(· |x, σ),Pr(· |x, φ))≤ε,
8

Statistical Detection of Positional Bias in LLM-Based Recommender SystemsA PREPRINT
whereTV(π, µ) =1
2Pk
j=1|π(j)−µ(j)|denotes the total variation distance.
Theorem 1.Underε-absence of positional bias, we haveTV(p(·), U k)≤ε, whereU kdenotes the uniform distribu-
tion over{1,2, . . . , k}. In fact, for all instancesx,TV(p(· |x), U k)≤ε.
The proof of Theorem 1 is given in Appendix A.
Remark 1.Proposition 1 is a particular case of Theorem 1 withε= 0. Indeed,TV(p(· |x), U k) = 0implies
p(i|x) = 1/kfor alli. Moreover, here we still allow for stochastic answers of the RS.
Theorem 1 establishes thatε-absence of positional bias impliesTV(p(·), U k)≤ε. The null hypothesis naturally
associated withε-absence is therefore the interval hypothesis
Hε
0: TV(p(·), U k)≤εversusHε
1: TV(p(·), U k)> ε,(4)
and, by contraposition, rejectingHε
0provides evidence againstε-absence of positional bias. Testing exact uniformity
ofpcorresponds to the special caseε= 0. Since exact absence of positional bias is rarely attainable in practice, and
a test of exact uniformity may flag arbitrarily small deviations when the sample is large, we now describe a test ofHε
0
for arbitraryε≥0.
The test relies on the following characterization of the total variation distance to the uniform distribution. Writing
x+= max(x,0), we have
TV(p(·), U k) = max
A⊆{1,2,...,k}
p(A)−|A|
k
=kX
i=1
p(i)−1
k
+,(5)
wherep(A) =P
i∈Ap(i), and the maximum is attained atA∗={i:p(i)>1/k}. Writingµ i=p(i)−1/kfor
each prompt positioni, it therefore suffices to obtain simultaneous lower confidence bounds for thekscalar quantities
µ1, . . . , µ k.
LetX 1, . . . , X nbe instances sampled independently from our data distributionD(see subsection 3.2). For eachX j
we sampleℓpermutationsσ j,1, . . . , σ j,ℓuniformly and independently fromS k, and determineR(X j, σj,t)for allt.
We define
ˆqj(i) =1
ℓ|{t∈ {1,2, . . . , ℓ}:R(X j, σj,t) =i}|,
that is,ˆq j(i)is the fraction of evaluated permutations for instanceX jin which the LLM selected the item at position
iin the prompt.2For each positioni∈ {1,2, . . . , k}, we define the instance-level statistics
Zj(i) = ˆq j(i)−1
k, j= 1,2, . . . , n.
Since the instances are sampled independently and the permutations are drawn independently of each other and of the
instances, the variablesZ 1(i), . . . , Z n(i)are independent and identically distributed. Moreover,E[ˆq j(i)] =p(i)by
Equation 1, so their common mean isE[Z j(i)] =µ i=p(i)−1/k.
Fix a significance levelα∈(0,1); in our experimentsα= 0.05. Let ¯Z(i)ands idenote the sample mean and standard
deviation ofZ 1(i), . . . , Z n(i). For each positioniwe define the one-sided lower confidence bound
Li=¯Z(i)−t 1−α/k, n−1si√n,(6)
wheret 1−α/k, n−1 denotes the(1−α/k)-quantile of the Student’stdistribution withn−1degrees of freedom, and
we aggregate these bounds into
L=kX
i=1(Li)+.(7)
The quantitiess i,LiandLdepend onn, which we omit from the notation.
Proposition 2.Fixk≥2,ℓ≥1andα∈(0,1), and letLbe the bound defined in Equation 7, computed fromn
independent instances. Then
lim inf
n→∞Pr[L≤TV(p(·), U k) ]≥1−α.
The proof of Proposition 2 is given in Appendix A.
2When some generations are discarded due to parsing errors or refusals (see subsection 4.3),ˆq j(i)is computed over the valid
generations ofX j, and the analysis below is understood conditionally on their number.
9

Statistical Detection of Positional Bias in LLM-Based Recommender SystemsA PREPRINT
Corollary 1.Fixk≥2,ℓ≥1andα∈(0,1), and letLbe the bound defined in Equation 7, computed fromn
independent instances. For everyε≥0, the test that rejectsHε
0wheneverL > εhas asymptotic levelα, that is,
lim supn→∞ Pr[L > ε]≤αwheneverTV(p(·), U k)≤ε.
Proof.UnderHε
0, the eventL > εimpliesL >TV(p(·), U k), whose probability haslim supat mostαby Proposi-
tion 2.
By Corollary 1,ε-absence of positional bias is rejected at levelαfor everyε < L, so that no tolerance needs to be
fixed in advance. Onlykstatistics are computed to determineL, rather than the2ksubsets appearing in Equation 5, so
the procedure is computationally efficient for every candidate-set size considered in this work. Beyond the aggregate
boundL, the individual boundsL iare informative in their own right: a positioniwithL i>0is over-selected with
respect to the uniform baseline, at simultaneous confidence level1−α. We reportLas a percentage in subsection 4.4.
The construction above is valid for any number of permutations per instanceℓ≥1, includingℓ= 1, sinceE[ˆq j(i)] =
p(i)for everyℓ, and hence ¯Z(i)is always an unbiased estimator ofµ i. The choice ofℓaffects only the variance of the
instance-level statistics, which decomposes as
Var(Z j(i)) = Var X(p(i|X)) +1
ℓEX[Var σ(1[R(X, σ) =i]|X)].
Only the second, within-instance term decreases withℓ, while the first, between-instance term is irreducible. Conse-
quently, increasingℓyields diminishing returns, and beyond a moderate value the power of the test is governed by the
number of instancesn. In the simulation study of Appendix B, the between-instance term vanishes, sincep(i|X)
does not depend onXin the simulated recommender, and the required number of instances decreases roughly as1/ℓ
(Table 9); in real data, where the relevance of the candidates varies across queries, the gains from increasingℓare
expected to saturate earlier. In our experiments we useℓ= 20, matching the query-permutation dataset described in
subsection 4.1.
4 Empirical results and analysis of positional bias
This section presents the empirical evaluation and statistical analysis conducted to assess positional bias in LLM-based
RS. The experiments are designed to test the null hypotheses of positional independence and uniformity formalized
in section 3, using the POB-ICM and POB-UNBP detection methodologies introduced in subsection 3.3 and subsec-
tion 3.4, respectively.
We first describe the construction of the evaluation datasets in subsection 4.1, including the generation of user queries,
candidate item sets, and their permutations. Secondly, subsection 4.3 details the experimental setup and implemen-
tation choices, which strictly follow the RS design introduced in subsection 3.1. Finally, subsection 4.4 reports the
empirical results and analyzes the observed positional bias in the different recommended LLMs, highlighting both
statistical significance and qualitative behavioral patterns.
4.1 Evaluation dataset
The dataset used in this study is derived by integrating and transforming domain-specific data sources. The main
external source is theAmazon ML Challenge 2023dataset. This dataset, in conjunction with user queries automatically
generated, forms the basis of the final RAG-query dataset.
User-query dataset. This dataset contains2000different user queries for RS. Each query asks for a specific type
of product, stating the user preferences. See Figure 2 for examples of such queries. The user-query dataset can
be generated using our source code3. To construct this dataset, we start with56distinct query templates, which
ask the RS for a generic recommendation, using different wording and user preferences in each. Each template has
placeholders for a product, up to three product attributes (such as quality, brand, feature, style, color, size, material, or
condition), and optionally a use case or feature. We consider194distinct products names, which can be grouped in18
categories: electronics, fashion, home appliances, beauty, sports, automotive, books, toys, groceries, office supplies,
pet supplies, fitness, gaming, travel, outdoor equipment, health, baby products, and jewelry. Our user-query dataset has
been constructed by sampling a category, a product within the category, a query template, and compatible attributes
uniformly at random (discarding the repetitions), ensuring that each selected attribute is compatible with the chosen
product category. Of the69,071,332total possible queries, we sampled2000in order to minimize the running time of
the experiments while having a large enough dataset to achieve statistical significance in our results.
3https://github.com/GoComputing/Positional-Bias.git
10

Statistical Detection of Positional Bias in LLM-Based Recommender SystemsA PREPRINT
Examples of queries
• Show me reputable brand fitness trackers .
• What’s the best modern portable speakers for students ?
• humidifiers that is large and new .
• What’s a good compact suspension trainers for gym ?.
Figure 2: Several user queries examples included in our dataset. These are generated from templates, where the
underlined words indicate a placeholder.
Products titles associated with "Find me an eco-friendly protein snack." query
• Green Sense Organic Multigrain Flour | Multigrain Atta | Gluten Free | Helps in Weight Loss, High
in Protein - 500gm x 1 | Pack of 1
• Go Vegan California Roasted Pistachios, 400 Gram | Pista, Lightly Salted - (Pack of 2 -200g Each)
• Go Vegan Stone Candy - 400g, Rock Candy, Milk Stone Chocolate [Jar Pack]
• Nutritional Designs Sugar-Free Liquid Collagen & Whey Protein Supplement - Non-GMO Drink -
Promotes Healthy Skin & Hair for Men & Women (Neutral)
• Dev.Pro. Date and Nuts Drops, Healthy Snack with High Fiber, Gluten Free, Vegan - Cocoa Coconut
40g, (Pack of 5)
Figure 3: Titles of the selected products for the query "Find me an eco-friendly protein snack". These are sorted by
the cosine similarity to the query using BAAI embedding model.
Products dataset. This dataset contains50,000distinct products, each with a name/title and a description, which we
have sampled from theAmazon ML Challenge 2023 dataset4. Examples of such products are shown in Figure 3.
Before sampling, we have discarded all products with description shorter than40characters or empty values in the
product type ID. The full Amazon ML Challenge 2023 dataset contains over2.9million products spanning9,919
distinct browse node categories. Its domain coverage is broad, encompassing a wide variety of product types such as
electronics, clothing, home appliances, food, movies, dietary supplements, and personal care products, among others.
While our evaluation is therefore restricted to the product recommendation domain, the diversity of product types
within this domain lends robustness to our findings and reduces the risk of conclusions being specific to a narrow
product niche.
RAG-query dataset. This dataset contains2000different queries. Each one of them consists of a user query and5
products from the product dataset. We have constructed this data set running a RAG scheme on the user-query data
set and the product dataset. That is, for each user query, the top5products (K= 5) are selected from the product
dataset, as described in subsection 3.1. Thus, we obtain a tuple (Q,P 1,P2,P3,P4,P5), whereQis the user query
andP 1, . . . , P 5are the top-5products. For the robustness analysis across candidate-set sizes (subsubsection 4.4.2),
analogous datasets are built from the same user queries with the top-kproducts, fork∈ {10,15,20}. In an LLM-
based RS, this RAG-query is sent to the LLM (by filling in a prompt template with the query and the5products), to
get a recommendation. Our hypothesis is that LLMs exhibit a systematic preference for certain positions within the
candidate list, independently of the relevance of the products.
Query-permutation dataset. This dataset contains, for each tuple (Q,P 1,P2,P3,P4,P5) in the RAG-query dataset,20
independently and uniformly sampled permutations of the5candidate products. Thus, this dataset contains2000·20 =
40,000instances. In our experiments, each LLM analyzed is evaluated on this dataset as explained in subsection 4.3.
The resulting query-permutation dataset yields sufficient sample size to support the tests employed in subsection 4.4,
whosep-values are calibrated with a query-level permutation test and thus do not rely on asymptotic expected-count
conditions (subsection 3.3). The same procedure is applied to the datasets withk∈ {10,15,20}, each also yielding
40,000instances. Unless stated otherwise, all results are based on these sampled configurations. In addition, to better
4Available athttps://www.kaggle.com/datasets/ashisparida/amazon-ml-challenge-2023
11

Statistical Detection of Positional Bias in LLM-Based Recommender SystemsA PREPRINT
illustrate the relevance signal of the retriever, fork= 5we also use an exhaustive configuration, in which all5! = 120
permutations are evaluated for a subset of730queries of the RAG-query dataset.
4.2 Positional bias metrics
Quantifying positional bias requires distinguishing betweenpermutation sensitivityand positional bias itself, as noted
by [14]. Permutation sensitivity captures the instability of the model under different orderings of the same candidate
set, alocalproperty that may change on a per-query basis. Positional bias, by contrast, refers to a systematic tendency
to favor certain prompt positions regardless of item relevance, aglobalproperty reflecting consistent patterns across
queries. Crucially, permutation sensitivity does not necessarily imply positional bias, but positional bias necessarily
induces some degree of permutation sensitivity. To capture both aspects, we introduce the following metrics.
Permutation sensitivity (PS).We quantify permutation sensitivity as the probability that the recommended item
differs from the item most frequently recommended for the same instance across permutations:
PS = 100·E X∼D
Pr
σ∼U(S k)[σ(R(X, σ))̸=m(X)]
,(8)
wherem(X) = arg max jPrσ∼U(S k)[σ(R(X, σ)) =j]is the most frequently recommended item forX. PS provides
a directly interpretable practical impact metric: it is the fraction of cases in which the top recommendation changes
solely due to item reordering.
Positional bias (PB).We quantify positional bias as the total variation distance between the position-selection dis-
tributionp(·)and the uniform distributionU k, expressed as a percentage:
PB = 100·TV(p(·), U k).(9)
To quantify the statistical uncertainty of this estimate, we construct a(1−α)confidence interval via percentile boot-
strap resampling at the query level. Let cPB∗
1, . . . ,cPB∗
Bdenote the PB values computed fromBbootstrap resamples.
The confidence interval is then
CI1−α(PB) =h
cPB∗
(⌊Bα/2⌋) ,cPB∗
(⌈B(1−α/2)⌉)i
,(10)
wherecPB∗
(j)denotes thej-th order statistic of the bootstrap distribution. In our experiments we setα= 0.05, yielding
95% confidence intervals. Note thatPB≥0by definition of total variation distance, withPB = 0only under exact
uniformity; in practice the confidence interval is bounded away from zero.
Together, PS measures thepractical instabilityof recommendations under reordering, while PB quantifies thestruc-
tural positional preferenceof the model. The latter is meaningful as evidence of positional bias only when the null
hypothesis of uniformity has been rejected by the POB-UNBP methodology; otherwise, deviations from uniformity
may reflect sampling variability alone.
Remark 2.Permutation sensitivity is an upper bound of positional bias,PB≤PS. Indeed, the proof of Theo-
rem 1 (Appendix A) only uses thatr φ0is a probability distribution over items, so it shows thatTV(p(· |x), U k)≤
Eσ[TV(r σ, µ)]for any instancexand any distributionµover items. Takingµas the point mass atm(x), we have
TV(r σ, µ) = 1−r σ(m(x)), and henceTV(p(· |x), U k)≤Pr σ[σ(R(x, σ))̸=m(x)]. Taking expectations over
X∼ Dand using the convexity of the total variation distance yields the claim.
Strong positional bias (sPB).The metrics PS and PB in Equation 8–Equation 9, as well as the POB-ICM and
POB-UNBP methodologies, treat any dependence of the recommended item on its position as positional bias, re-
gardless of the relevance of the candidates involved. Such a dependence can manifest in two distinct ways. When
several candidates are comparably relevant for the user query, the position of the items may determine which of them
is recommended; we refer to this assoft positional bias. This form of bias does not degrade the relevance of the
recommendation, but it still makes the exposure of comparably relevant items depend on the prompt construction
rather than on their content. Alternatively, the position may lead the model to recommend an item that is clearly less
relevant than the alternatives; we refer to this asstrong positional bias. Positional bias, as detected by POB-ICM and
POB-UNBP and quantified by PB, comprises both forms; we refer to this combination asoverall positional bias. To
make this distinction operational, we introduce a query-dependent relevance setR(X)⊆ {1,2, . . . , k}, identifying the
candidate items considered comparably relevant for instanceX. For each instance, we count how often each candidate
item is recommended across its sampled permutations.
12

Statistical Detection of Positional Bias in LLM-Based Recommender SystemsA PREPRINT
Definition 2.Given a thresholdτ∈(0,1], the relevance setR(X)of an instanceXis the smallest set of its most
frequently recommended items that together account for at least a fractionτof its recommendations.
The thresholdτis calibrated from the data. The recommendation frequencies of an instance, sorted in decreasing
order, usually show a natural break between the items the model treats as plausibly interchangeable and those it rarely
selects regardless of position. For each instance, we locate this break as the largest gap between consecutive sorted
frequencies and compute the fraction of recommendations captured by the items above it;τis the average of this
fraction across instances. Using a common threshold, instead of locating the break separately in each instance, makes
the relevance set less sensitive to the noise of the frequencies of individual instances, which are estimated from a
limited number of permutations.
We then define strong positional bias as the proportion of recommendations that fall outside the relevance set:
sPB = 100·Pr
X∼D,σ∼U(S k)[σ(R(X, σ))/∈ R(X)].(11)
To avoid calibrating the threshold on the same instances on which sPB is evaluated, we use5-fold cross-validation
over instances. For each fold,τis calibrated on the other four folds, and the relevance sets and sPB are computed
on the held-out fold. The reported sPB is the average over the five folds, so each instance is evaluated exactly once.
Note that sPB and PB are not on a comparable scale, and sPB may exceed PB even though it targets a component of
the overall positional bias. The calibrated values ofτfor each setting are reported in Appendix E, and they are stable
across folds.
4.3 Experimental setup and implementation details
In this section we include relevant details of our experimental setup and implementation.
1. For the RAG scheme, we use the BAAI general embedding model [25] as a semantic match between user
queries and product titles. For each user query, the top-5most similar products (K= 5) are selected, leading
to our RAG-query dataset.
2. When evaluating a tuple (Q,σ,P 1,P2,P3,P4,P5) from the query-permutation dataset, the final prompt sent
to the LLM uses the prompt template given in Prompt 1, filled in with the titles and descriptions of the top-5
selected products, and the user query. The top-5products appear in this prompt following the orderσ, that is,
first the productP σ(1), then the productP σ(2), etc.
3. Several recommender LLMs are tested5, including LLaMA 3.1 (8B and 70B) [8], LLaMA 4 Scout [2], GPT-
OSS (20B and 120B) [16], and DeepSeek R1 (8B LLaMA distill, 7B Qwen distill and 32B Qwen distill) [9].
The temperature of the recommender model is kept to zero, so the results are deterministic. Thus, if there
were no positional bias, the recommender should output the same recommendation for a given user query and
top-5products, independently ofσ.
4. ARecommender parserhas been implemented to correct the JSON format. As LLM might fail in generating
a JSON following correct syntax, this algorithm fixes the most common mistakes the model makes, including
not generating the quotes, not escaping the quotes inside the strings, and so on. As a remark, this recom-
mender model does not always generate a valid recommendation6or the model even refuses to recommend
a product if no suitable product is found in the list. We report error rates and refusal rates on Table 1. Since
parsing errors are detected and the corresponding samples discarded prior to applying POB-ICM and POB-
UNBP, we assess their impact in Appendix G, where restricting POB-UNBP to the queries with no discarded
generations leaves its conclusions unchanged. If failure to produce a valid response is itself influenced by
item ordering, this would reflect positional bias already present in the model rather than bias introduced by
the methodology.
All of the described implementation details, both for the generation of the dataset and the evaluation of bias, are
available in our source code7, which allows reproduction of all of the results presented.
4.4 Results and analysis
Table 2 summarizes the outcome of both proposed methodologies, POB-ICM and POB-UNBP, together with the quan-
tification metrics PS, PB, and sPB for the top-5setting, providing a unified overview of the statistical significance and
5Tested models are in FP16 precision, except for GPT-OSS and LLaMA 3.1 70B
6A valid recommendation generates a JSON with the specified format and a valid product position.
7https://github.com/GoComputing/Positional-Bias.git
13

Statistical Detection of Positional Bias in LLM-Based Recommender SystemsA PREPRINT
Model nameError rate Refusal rate
Top-5 Top-10 Top-15 Top-20 Top-5 Top-10 Top-15 Top-20
DeepSeek R1 32B 0.35% 0.39% 0.24% 0.26% 4.03% 3.89% 3.82% 4.06%
DeepSeek R1 7B 0.81% 1.58% 1.99% 2.46% 7.39% 5.38% 5.14% 4.36%
DeepSeek R1 8B 0.37% 0.40% 0.21% 0.23% 0.01% 0.01% 0.00% 0.00%
LLaMA 3.1 8B 0.60% 1.99% 3.23% 4.01% 0.01% 0.01% 0.01% 0.01%
LLaMA 4 Scout 0.21% 0.48% 2.05% 3.20% 0.17% 0.06% 0.08% 0.05%
GPT OSS 120B 0.45% 0.50% 0.45% 0.36% 0.01% 0.02% 0.01% 0.03%
GPT-OSS 20B 1.01% 1.07% 1.11% 1.22% 0.04% 0.03% 0.03% 0.02%
LLaMA 3.1 70B 0.31% 0.41% 0.44% 0.54% 0.12% 0.17% 0.22% 0.26%
Table 1: Recommendation parsing error rate and refusal rate for each candidate-set size, each computed over the
40,000samples of the corresponding query-permutation dataset.
Model namep-value
(POB-ICM)p-value
(POB-UNBP)LPermutation
sensitivity
(PS)Positional bias (PB) Strong
Positional
Bias (sPB)
DeepSeek R1 32B<0.0005 7.456e-070.97% 21.857 2.563, CI: (2.0, 3.2) 3.1
DeepSeek R1 7B<0.0005 6.197e-604.15% 33.983 5.612, CI: (5.1, 6.3) 9.8
DeepSeek R1 8B<0.0005 5.875e-050.79% 29.758 2.391, CI: (1.8, 3.0) 6.7
LLaMA 3.1 8B<0.0005 4.457e-263.01% 30.395 4.187, CI: (3.6, 4.8) 6.7
LLaMA 4 Scout<0.0005 1.188e-040.87% 22.393 2.886, CI: (2.4, 3.4) 3.2
GPT OSS 120B<0.0005 8.879e-303.14% 15.487 4.368, CI: (3.8, 5.0) 1.0
GPT-OSS 20B<0.0005 4.814e-654.73% 17.709 5.944, CI: (5.3, 6.6) 2.1
LLaMA 3.1 70B<0.0005 7.216e-243.40% 22.654 4.638, CI: (4.0, 5.3) 3.1
Table 2: Results from the proposed bias quantification metrics and proposed statistical methodologies for top-5setting.
POB-ICMp-values are Monte Carlo estimates fromB= 2000permutations, so their smallest attainable value is
1/(B+ 1) = 0.0005.Lvalue is the lower bound ofTV(p(·), U k)according to the POB-UNBP test. Confidence
intervals of sPB reported at Appendix E.
magnitude of positional bias across all evaluated models. We then examine these results in turn. Subsection 4.4.1 de-
tails the POB-ICM results, including the contingency-matrix visualization and the associated bias-inspection analysis,
characterizinghowpositional bias manifests for each model. Subsection 4.4.2 reports the POB-UNBP results, which
corroborate these findings with a more sample-efficient test and additionally assess the robustness of positional bias as
the candidate-set sizekand the decoding temperaturetvary. Finally, Subsection 4.4.3 analyzes the PS, PB, and sPB
metrics across these same conditions, disentangling generic permutation sensitivity from the structural, position-driven
component of the bias.
4.4.1 Results and visualization of POB-ICM methodology
We first apply the POB-ICM methodology from subsection 3.3. To do so, the contingency matrix is built from the
results generated using Equation 2. This empirical contingency matrix is represented in Figure 4 for LLaMA 8B. The
statistical test results of the POB-ICM methodology are shown in Table 2.
Result 4.1
Under the POB-ICM methodology, the calibrated Chi-square test rejects the null hypothesis of independence
between candidate-item position and the selected output in all evaluated models, with p-values below the
standard significance threshold of0.05, providing strong evidence that positional dependence is a substantive
and reproducible property of the evaluated pipelines. See Table 2 for exact results.
We can further analyze the positional bias through Figure 4. The figure includes two configurations: 730 queries
with exhaustive permutations (120 per query), and 2000 queries with sampled permutations (20 per query). In the
exhaustive configuration, the products in the first position given by the semantic matcher (first row) are more likely
14

Statistical Detection of Positional Bias in LLM-Based Recommender SystemsA PREPRINT
(a) Generated using 730 queries and 120 permutations per query.
 (b) Generated using 2000 queries and 20 sampled permutations
per query.
Figure 4: Contingency matrix from subsection 4.3, shown as a heatmap. The plot was generated using LLaMA 3.1 8B
as recommender. Each column shows the empirical distribution of recommendations conditioned on a permutation,
with permutations sorted lexicographically. Red lines indicate when the first, second, third, fourth, and fifth product are
shuffled to the first position, respectively. Consistency was ensured by shuffling products according to the permutation
sample, generating the recommendation, and then unshuffling the recommendation positions.
to be selected, while the last product is less likely to be selected (last row). Indeed, the first product is the preferred
product by the semantic matcher, so it is more likely to also be preferred by the LLM. Note that the semantic matcher
only has access to the product title, while the LLM has access to both the title and the description of the product,
and thus, the recommendation of the LLM should be more reliable. We stress that this row structure reflects item
relevance, not positional bias. Rows are indexed by the original (pre-shuffling) positionY=σ(R(X, σ)), so unequal
row totals only indicate that the LLM tends to agree with the retriever on which candidates are more relevant. This
does not confound the detection of positional bias, for two reasons. First, the candidates are presented to the LLM in
a uniformly random orderσ, and retrieval scores are never included in the prompt, so the retriever’s ranking cannot
favor any prompt position. Second, the null hypothesis of POB-ICM (independence betweenYandσ) places no
restriction on the row totals: positional bias manifests instead as differences between columns, that is, as a dependence
of the selected item on the permutation applied. Similarly, the distributionp(·)used by POB-UNBP is indexed by
prompt position, and Proposition 1 and Theorem 1 guarantee its uniformity under absence of positional bias for every
instance, regardless of how unequal the relevance of its candidates is. In the sampled configuration, this influence
of the similarity ranker is less clearly visible, as the reduced number of permutations per query introduces additional
noise in the contingency matrix. Nevertheless, the overall positional bias structure remains consistent across both
configurations, supporting the robustness of the detected effect.
Positional bias in this setting is better visualized using our proposedbias inspection matrix, described in subsection 3.3.
This matrix is interpreted by columns. Each column represents products selected when these are moved to the position
j. This plot is shown in Figure 5, generated using the 2000-query configuration with 20 sampled permutations per
query. For example, in the plot associated with LLaMA 3.1 8B, we note that products moved to the first position
are less likely selected. We also note that products moved to the last position are more likely to be selected. There
are several potential explanations for this, as, for example, LLMs pay more attention to tokens that are closer to the
end of the prompt, as observed in [27]. This is not the same for models from the GPT-OSS family, where products
moved to the first position are more likely recommended, so the positional bias is dependent either on the architecture,
training, or post-training process. Interestingly, DeepSeek R1 8B and 32B exhibit comparatively weaker positional
bias. While the underlying causes are not investigated here, this observation suggests that architectural or training
differences may modulate positional sensitivity. Notably, these qualitative patterns are consistent with those observed
under the exhaustive 730-query configuration, supporting the robustness of the findings across evaluation settings.
Observation 4.1
Using the proposed POB-ICM visualization methodology in Figure 5, several models exhibit different posi-
tional biases. LLaMA 3.1 8B tends to select the last products in the prompt, while models like GPT-OSS and
LLaMA 3.1 70B tend to select first products.
15

Statistical Detection of Positional Bias in LLM-Based Recommender SystemsA PREPRINT
Figure 5: Bias analysis given the Equation 3 for different recommender LLMs. For each model, a column of its
associated heatmap represents the mean of the (non-normalized) counts of selected products by the model when these
were moved to the position represented by said column. In the absence of positional bias, all columns would be
identical.
Top-5 Top-10 Top-15 Top-20
Model name p-valueL p-valueL p-valueL p-valueL
DeepSeek R1 32B 7.456e-070.97%5.130e-172.83%2.202e-474.29%2.098e-486.78%
DeepSeek R1 7B 6.197e-604.15%9.362e-555.79%5.966e-799.51%4.449e-709.89%
DeepSeek R1 8B 5.875e-050.79%1.150e-111.96%2.253e-464.30%2.111e-717.24%
LLaMA 3.1 8B 4.457e-263.01%4.928e-173.13%3.379e-164.08%1.465e-263.77%
LLaMA 4 Scout 1.188e-040.87%1.438e-608.30%8.023e-20821.43%1.341e-24930.19%
GPT OSS 120B 8.879e-303.14%2.233e-413.62%6.968e-485.10%1.601e-587.00%
GPT-OSS 20B 4.814e-654.73%5.034e-825.69%6.788e-967.78%2.993e-1009.03%
LLaMA 3.1 70B 7.216e-243.40%4.845e-252.99%2.463e-294.40%1.045e-417.04%
Table 3: POB-UNBP statistical testp-value and the computed lower boundLfor different top-Kvalues and a fixed
temperaturet= 0. Bold values represent statistically detectable positional bias according to the threshold of0.05.
4.4.2 Results on POB-UNBP methodology
A more optimized methodology, POB-UNBP, is also proposed in subsection 3.4. This methodology mathematically
proves approximate uniformity of the probability distributionp(i)under the assumption of absence of positional bias,
both in deterministic and stochastic settings. This allows testing whetherp(i)deviates from the uniform distribution
through the lower confidence boundLonTV(p(·), U k)given in Equation 7. Thus, a much smaller number of queries
are required to evaluate the positional bias, at the expense of lower interpretability of the results (see Appendix B). This
reduction in sample complexity is enabled by averaging random permutations of the candidate set, a strategy aligned
with the classical permutation-based hypothesis testing [7]. Since the permutations of each query are aggregated into
the instance-level statisticsZ j(i), which are independent across queries, all20sampled permutations per query are
used without violating the independence assumption. Along withL, we report thep-value of the test in Corollary 1 for
ε= 0, that is, of the null hypothesis of exact uniformity, defined as the smallest significance levelαat whichL >0.
SinceL >0if and only ifL i>0for some positioni, thisp-value ismin (1, k·min ipi), wherep iis the one-sided
p-value of thet-statistic ¯Z(i)√n/siwithn−1degrees of freedom; see Table 3 for exact results.
Result 4.2
By the POB-UNBP methodology, positional bias is confirmed in all evaluated models, requiring a smaller
quantity of samples to confirm positional bias compared to the POB-ICM methodology (see Appendix B).
The statistical tests confirm non-uniformity in all evaluated models, providing further evidence of positional bias.
To assess whether the strength and location of this bias depend on the size of the candidate set, we visualize the
results in Figure 6. Two qualitatively distinct behaviors emerge. DeepSeek R1 variants and LLaMA 3.1 8B exhibit
a preference above uniform baseline for middle positions over the very first positions of the prompt, and the location
16

Statistical Detection of Positional Bias in LLM-Based Recommender SystemsA PREPRINT
t= 0.0t= 0.2t= 0.4t= 0.6
Model name p-valueL p-valueL p-valueL p-valueL
DeepSeek R1 32B 7.456e-070.97%2.001e-091.14%6.392e-060.79%6.290e-071.12%
DeepSeek R1 7B 6.197e-604.15%1.924e-644.29%7.115e-584.24%2.998e-664.40%
DeepSeek R1 8B 5.875e-050.79%8.181e-060.90%9.225e-081.21%1.701e-161.65%
LLaMA 3.1 8B 4.457e-263.01%2.049e-282.67%4.884e-212.27%1.925e-171.97%
LLaMA 4 Scout 1.188e-040.87%1.476e-050.97%5.367e-050.97%6.510e-051.00%
GPT OSS 120B 8.879e-303.14%3.367e-313.27%4.932e-313.40%1.528e-323.24%
GPT-OSS 20B 4.814e-654.73%1.098e-735.26%1.878e-725.06%1.475e-694.93%
LLaMA 3.1 70B 7.216e-243.40%1.690e-283.63%7.065e-263.78%6.539e-394.48%
Table 4: POB-UNBP statistical testp-value and the lower boundLfor different decodingttemperatures and a fixed
K= 5. Bold values represent statistically detectable positional bias according to the threshold of0.05.
of this preference is itself sensitive tok: the bias shifts toward later relative positions for smallerkand toward
earlier relative positions askincreases. LLaMA 4 Scout displays a related but substantially more pronounced pattern,
where the preference for later positions grows markedly withk, indicating that this model is particularly sensitive to
candidate-set size. In contrast, GPT-OSS (20B and 120B) and LLaMA 3.1 70B consistently favor the first positions of
the prompt, and this tendency remains stable across all evaluated values ofk, suggesting that positional bias in these
models is largely independent of the number of candidates presented.
To assess whether positional bias is an artifact of deterministic decoding or persists under stochastic sampling, we
repeat the POB-UNBP evaluation at top-5fort∈ {0.0,0.2,0.4,0.6}, wheret= 0.0corresponds to the deterministic
setting reported in subsection 4.4. We use these temperatures according to the used temperature in the literature [10].
The correspondingp-values are reported in Table 4.
Result 4.3
Positional bias is detected by the POB-UNBP methodology consistently across decoding temperatures, withp-
values remaining below the standard significance threshold of0.05for all models at every evaluated value oft.
The magnitude and significance of the effect do not vary substantially astincreases, indicating that positional
bias is not an artifact of deterministic decoding but persists under stochastic sampling.
Table 5 holds the position-selection distributions across temperatures for top-5. Consistently with the statistical results
in Table 4, the qualitative shape of the bias remains stable astvaries for all evaluated models. GPT-OSS (20B and
120B) and LLaMA 3.1 70B preferentially select the first positions of the prompt across all temperatures, mirroring the
behavior observed in the top-kexperiment. LLaMA 3.1 8B, DeepSeek R1 32B, and DeepSeek R1 8B instead favor
later positions, while DeepSeek R1 7B exhibits a distinct pattern, concentrating its bias on the second position rather
than the first or last. LLaMA 4 Scout shows the complementary behavior, systematically avoiding the first position
regardless of temperature. Taken together, these results indicate that positional bias is primarily determined by model
identity rather than by decoding stochasticity, reinforcing the conclusion that it reflects a structural property of the
underlying model rather than an effect of the sampling procedure.
4.4.3 Positional bias quantification
Beyond the hypothesis-testing results reported above, we now characterize positional bias through the quantification
metrics introduced in section 3: permutation sensitivity (PS), positional bias (PB), and strong positional bias (sPB).
Before reporting how these metrics evolve acrosskandt, we first illustrate the rationale behind the relevance set
R(X)on which sPB is based.
Figure 7 shows, for twenty randomly sampled queries under the exhaustive configuration (730 queries, 120 permu-
tations each), the empirical selection count of each candidate item in its original (pre-shuffling) position. These
histograms reveal substantial heterogeneity across queries in the number of items the model treats as plausible recom-
mendations. For several queries, a single item is selected overwhelmingly often regardless of the permutation applied,
indicating a clear, unambiguous preference largely unaffected by item ordering. For other queries, however, selection
counts are split across two or more items, suggesting that the model regards several candidates as comparably relevant
to the user query, and that residual instability across permutations within this subset reflects soft rather than strong
positional bias. This heterogeneity directly motivates the construction of a query-dependent relevance setR(X)(Def-
inition 2): treating every change in the recommended item as evidence of strong positional bias, without accounting
17

Statistical Detection of Positional Bias in LLM-Based Recommender SystemsA PREPRINT
(a)k= 5
(b)k= 10
(c)k= 15
(d)k= 20
Figure 6: Representation of probability distribution p(i) defined in Equation 1 for differentkvalues. Red-dashed lines
represent uniform baseline1/k.
for cases of comparably relevant alternatives, would overstate the practical impact of position on the relevance of the
recommendations.
Table 6 reports PS, PB, and sPB acrossk∈ {5,10,15,20}for all evaluated models. Several patterns emerge. First,
permutation sensitivity is substantial and grows monotonically withkfor every model, confirming that prompt sen-
sitivity is a genuine and increasingly impactful phenomenon as the candidate set grows: e.g., DeepSeek R1 7B rises
from34.0%atk= 5to58.1%atk= 20, meaning the recommended item changes on more than half of all reorder-
ings once twenty candidates are presented. Second, positional bias (PB) generally increases withk(the only exception
being a slight decrease for DeepSeek R1 7B fromk= 15tok= 20), though the magnitude of this growth varies
considerably across model families. LLaMA 4 Scout exhibits by far the steepest increase, with PB rising from2.9%
atk= 5to33.8%atk= 20(more than a tenfold increase), consistent with the pronouncedk-dependence already
observed in the probability distributions in Figure 6. In contrast, GPT-OSS (20B and 120B) and LLaMA 3.1 70B
show comparatively modest growth in PB (roughly a2–2.5×increase fromk= 5tok= 20), mirroring their stable
18

Statistical Detection of Positional Bias in LLM-Based Recommender SystemsA PREPRINT
Model name Temperature p(1) p(2) p(3) p(4) p(5)
DeepSeek R1 32B0.0 0.1874 0.1869 0.2060 0.2083 0.2114
0.2 0.1869 0.1868 0.2046 0.2080 0.2137
0.4 0.1873 0.1879 0.2055 0.2097 0.2097
0.6 0.1872 0.1852 0.2053 0.2103 0.2119
DeepSeek R1 7B0.0 0.1738 0.2514 0.2004 0.2043 0.1701
0.2 0.1686 0.2535 0.2012 0.2039 0.1729
0.4 0.1689 0.2509 0.2012 0.2069 0.1721
0.6 0.1644 0.2544 0.1994 0.2059 0.1759
DeepSeek R1 8B0.0 0.1883 0.2104 0.1878 0.2037 0.2098
0.2 0.1898 0.2083 0.1896 0.2001 0.2121
0.4 0.1847 0.2094 0.1884 0.2029 0.2146
0.6 0.1847 0.2057 0.1870 0.2006 0.2221
LLaMA 3.1 8B0.0 0.1644 0.1981 0.2122 0.1956 0.2297
0.2 0.1629 0.2021 0.2088 0.1973 0.2289
0.4 0.1679 0.2026 0.2097 0.1960 0.2237
0.6 0.1748 0.1998 0.2095 0.1951 0.2208
LLaMA 4 Scout0.0 0.1711 0.2092 0.2057 0.2041 0.2099
0.2 0.1691 0.2105 0.2049 0.2053 0.2102
0.4 0.1694 0.2097 0.2068 0.2055 0.2085
0.6 0.1702 0.2097 0.2072 0.2040 0.2090
GPT OSS 120B0.0 0.2318 0.2119 0.1895 0.1875 0.1793
0.2 0.2319 0.2122 0.1870 0.1900 0.1790
0.4 0.2320 0.2137 0.1874 0.1892 0.1777
0.6 0.2329 0.2112 0.1897 0.1878 0.1784
GPT-OSS 20B0.0 0.2538 0.2057 0.1834 0.1843 0.1728
0.2 0.2574 0.2073 0.1827 0.1813 0.1712
0.4 0.2574 0.2052 0.1833 0.1810 0.1730
0.6 0.2549 0.2071 0.1814 0.1838 0.1729
LLaMA 3.1 70B0.0 0.2299 0.2165 0.1953 0.1789 0.1794
0.2 0.2307 0.2166 0.1930 0.1807 0.1789
0.4 0.2289 0.2205 0.1944 0.1790 0.1773
0.6 0.2363 0.2202 0.1940 0.1760 0.1736
Table 5: Probability distributions p(i) defined in Equation 1 for different temperature values and top-5 setting. Uniform
baseline is0.2.
first-position preference acrossknoted previously. Third, strong positional bias (sPB) increases monotonically with
kfor every model, growing between roughly 3 and 11 fromk= 5tok= 20; this suggests that as the candidate set
grows, a progressively larger share of permutation-induced changes involve items outside the relevance set, at a model-
dependent rate ranging from DeepSeek R1 8B (6.7%to20.6%) to LLaMA 4 Scout (3.2%to36.6%). Taken together,
these results indicate that larger candidate sets do not merely amplify generic instability, but specifically exacerbate
the structural, position-driven component of that instability, with the magnitude of this effect strongly modulated by
model identity.
Table 7 reports the same metrics across decoding temperaturest∈ {0.0,0.2,0.4,0.6}at top-5. In contrast to the
k-dependence above, all three metrics remain remarkably stable as temperature increases: PS grows only mildly (e.g.,
LLaMA 3.1 8B rises from30.4%att= 0.0to33.0%att= 0.6), and sPB follows a similarly modest upward trend,
consistent with the added decoding stochasticity introducing a small amount of additional, content-driven variability in
the recommended item. Notably, PB does not exhibit a consistent upward trend with temperature; for several models
it remains essentially flat or even decreases slightly (e.g., LLaMA 3.1 8B decreases from4.2%att= 0.0to3.0%
att= 0.6, and DeepSeek R1 32B fluctuates within a narrow band of2.5–2.8%). This indicates that the structural,
position-driven component of the bias captured by PB is largely insensitive to sampling temperature, while the mild
increase observed in PS and sPB is better attributed to the additional stochasticity in item selection introduced by non-
zero temperature itself, rather than to a strengthening of positional bias. This observation reinforces the conclusion
19

Statistical Detection of Positional Bias in LLM-Based Recommender SystemsA PREPRINT
Figure 7: Selection counts of each candidate item, by original (pre-shuffling) position, for twenty random queries of the
exhaustive configuration, using LLaMA 3.1 8B. Each panel shows, for one query, how many of the 120 permutations
recommended each item.
Model namePS PB sPB
5 10 15 20 5 10 15 20 5 10 15 20
DeepSeek R1 32B 21.9 29.4 34.1 37.2 2.6 5.0 7.1 9.4 3.1 6.9 10.7 13.7
DeepSeek R1 7B 34.0 45.1 52.7 58.1 5.6 8.9 13.5 13.0 9.8 20.1 26.6 33.2
DeepSeek R1 8B 29.8 38.6 43.2 46.4 2.4 4.5 7.1 10.1 6.7 14.8 19.4 20.6
LLaMA 3.1 8B 30.4 41.9 46.9 49.6 4.2 5.3 6.8 7.0 6.7 15.7 20.7 24.1
LLaMA 4 Scout 22.4 38.3 51.5 61.6 2.9 10.5 24.3 33.8 3.2 12.9 24.5 36.6
GPT OSS 120B 15.5 21.9 24.9 27.3 4.4 5.4 7.5 10.2 1.0 3.4 5.9 6.7
GPT-OSS 20B 17.7 24.3 28.3 30.9 5.9 7.8 10.2 11.6 2.1 5.6 6.7 9.4
LLaMA 3.1 70B 22.7 31.7 36.7 40.2 4.6 5.3 7.3 10.7 3.1 9.5 12.2 14.9
Table 6: Permutation sensitivity (PS), positional bias (PB), and strong positional bias (sPB), all expressed as percent-
ages, acrossk∈ {5,10,15,20}. Confidence intervals of sPB metric reported in Appendix E.
already drawn from the POB-UNBP results in Table 4: positional bias is a structural property of the model that persists,
largely unchanged in magnitude, under stochastic decoding.
Since the relevance set is estimated from the selection frequencies of the model itself, sPB also captures recommen-
dations outsideR(X)that are not caused by the order of the items. These arise from stochastic decoding, that is, the
randomness of token sampling at non-zero temperature, which can change the recommended item even if the prompt
is identical, and from the estimation of the relevance set from a limited number of permutations. Together, they define
a noise floor, the level of sPB observed without any change in item order. Appendix C estimates this floor by repeating
the generation for a fixed order att= 0.6: it lies between0.9%and9.1%, comparable to the sPB obtained with
permutations, and permutations add between0.3and1.1percentage points to it for six of the eight models. Hence,
absolute sPB values should be read as an upper bound on the strong component of positional bias, while its trends
acrosskand across models are more informative.
20

Statistical Detection of Positional Bias in LLM-Based Recommender SystemsA PREPRINT
Model namePS PB sPB
0.0 0.2 0.4 0.6 0.0 0.2 0.4 0.6 0.0 0.2 0.4 0.6
DeepSeek R1 32B 21.9 22.7 22.9 23.4 2.6 2.6 2.5 2.8 3.1 3.3 3.4 3.5
DeepSeek R1 7B 34.0 34.6 35.3 35.9 5.6 5.9 5.9 6.0 9.8 10.0 10.4 10.4
DeepSeek R1 8B 29.8 30.9 31.5 32.6 2.4 2.1 2.7 2.8 6.7 6.9 8.7 9.9
LLaMA 3.1 8B 30.4 31.4 32.3 33.0 4.2 4.0 3.6 3.0 6.7 7.1 9.3 9.4
LLaMA 4 Scout 22.4 22.6 22.6 22.9 2.9 3.1 3.1 3.0 3.2 3.2 3.2 3.3
GPT OSS 120B 15.5 15.6 15.7 16.0 4.4 4.4 4.6 4.4 1.0 1.0 1.2 1.1
GPT-OSS 20B 17.7 17.9 18.6 18.6 5.9 6.5 6.3 6.2 2.1 2.5 2.6 2.7
LLaMA 3.1 70B 22.7 22.8 23.7 24.2 4.6 4.7 4.9 5.6 3.1 3.4 3.5 3.5
Table 7: Permutation sensitivity (PS), positional bias (PB), and strong positional bias (sPB), all expressed as percent-
ages, for top-5across decoding temperaturest∈ {0.0,0.2,0.4,0.6}. Confidence intervals of sPB metric reported in
Appendix E.
5 Conclusion and future work
This paper investigated positional bias in top-1 and single-turn LLM-based RS with the goal of determining whether,
how, and to what extent the order of candidate items influences recommendation outcomes. AddressingRQ1, our
empirical analysis provides statistical evidence that the output selection of LLM-based RS depends on the position
at which candidate items are presented in the input prompt. Through extensive permutation-based evaluations, we
show that recommendation behavior is not invariant to item ordering, even when item relevance and retrieval are held
constant. AddressingRQ2, we demonstrate that this dependence is statistically significant under formal hypothesis
testing, with all evaluated models rejecting the null hypotheses of positional independence and uniformity using two
complementary methodologies. Finally, with respect toRQ3, our results indicate that the magnitude and structure
of positional bias vary across models, even within the same family, such as LLaMA 3.1 8B and 70B, suggesting that
positional bias is influenced by design and training choices rather than being purely incidental. By increasing the
number of presented items in the prompt, the positional bias is observed to generally increase across models. This idea
is supported by how LLMs attend to different regions of the input prompt, where information appearing at the begin-
ning or end of the prompt is often more influential for the model, regardless of the semantic information distributed
across the prompt [13]. Furthermore, we distinguish two components of positional bias: soft positional bias, where
the position determines which of several comparably relevant candidates is recommended, and strong positional bias,
where the position displaces the recommendation towards clearly less relevant items. While the proposed statistical
tests detect the overall positional bias, which comprises both components, the proposed metrics quantify the overall
positional bias and its strong component, and our counterfactual analyses show that strong positional bias persists
when controlling for the identifiers assigned to the items, and for six of the eight evaluated models when controlling
for stochastic decoding.
These findings should be interpreted within the scope of our experimental setting. Our evaluation covers single-turn
product recommendation, combining user queries generated from templates with real components: a real product
catalog (Amazon ML Challenge 2023), a pre-trained embedding-based retriever, and a single fixed prompt template.
The heterogeneity of the real product titles and descriptions makes the task non-trivial; for instance, the titles generated
by the models do not always match the original ones exactly. However, the template-generated queries encode a limited
number of user preferences, and other domains, query formulations or prompt designs may yield different magnitudes
and patterns of positional bias. Moreover, we focus on top-1 recommendation, a particularly demanding setting for
positional bias analysis, since the relevance the model assigns to the candidates can only be inferred from a single
selected item rather than from a full ranking. Within this scope, the main contribution of this work is methodological:
a statistically grounded framework to detect overall positional bias in top-1 LLM-based RS, together with metrics
to quantify it and the distinction between soft and strong positional bias, quantifying the latter, which is applicable
beyond the evaluated setting, as discussed below.
Beyond its methodological contributions, this work has direct implications for the design, evaluation, and governance
of RS based on LLM in practice. In operational settings, positional bias can undermine the reliability of A/B testing,
distort offline evaluation metrics, and introduce unintended variability in user-facing recommendations. From a gover-
nance perspective, such biases complicate auditability and accountability, as system behavior may change solely due
to prompt construction choices rather than genuine differences in item relevance. For practitioners deploying LLM-
based RS, our findings highlight the importance of treating prompt order as a first-class design variable rather than an
implementation detail.
21

Statistical Detection of Positional Bias in LLM-Based Recommender SystemsA PREPRINT
Based on these findings, we outline several practical recommendations for organizations deploying LLM-based RS.
First, evaluation pipelines should explicitly incorporate randomized or counterbalanced item ordering to detect and
monitor positional bias. Second, prompt construction should be standardized or controlled to ensure consistency be-
tween deployments and experiments. Third, bias audits should include permutation-based tests, such as those proposed
in this work, as part of routine model validation and monitoring processes. Finally, when positional bias is detected,
mitigation strategies such as order randomization, ensemble prompting, or post-generation re-ranking should be con-
sidered to reduce its impact on end-user outcomes. Auditing positional bias is inexpensive: in our simulation study,
POB-UNBP reaches80%power with fewer than200queries and20permutations per query when the item placed first
is selected with an extra probability of0.03, which corresponds toPB≈2.4fork= 5, comparable to the weakest bias
among the evaluated models (Table 9). Since the simulation ignores the variability of relevance across queries, real
audits may require more queries. Thus, audits can be repeated whenever the model or the prompt template changes,
also to verify position-agnostic templates that avoid ordinal cues in the item identifiers. At inference time, randomiz-
ing the order of the candidates prevents items from being systematically favored by their retrieval rank at no additional
cost, and, when the model exposes probabilities over the candidate identifiers, the estimated distributionp(·)can serve
as a prior for post-hoc recalibration [28].
Regarding the extensibility of the proposed framework and future research lines, both POB-ICM and POB-UNBP are
designed to operate on any black-box system that (i) receives an indexed list of candidate products and (ii) returns a
recommendation expressed as a product position within that list. This abstraction is intentionally general and does
not depend on the internal architecture of the LLM, the retrieval mechanism, or the number of conversational turns.
Concretely:
•Multi-turn conversational RS.The framework can be applied to multi-turn settings provided that the
recommendation at each turn is expressed as a positional selection from a fixed candidate set. The
permutation-based evaluation can then be applied independently at each turn, treating each turn as a self-
contained Top-1 instance, although confirming that our empirical findings carry over to multi-turn interactions
requires a new set of experiments, which is outside the scope of this work.
•Top-Nrecommendation and re-ranking.These settings are less straightforward. When the system returns
an ordered list ofN >1items, or reorders the full candidate set, the notion of positional bias must be
generalized to account for how the relative ranking of selected items is affected by input ordering. This
requires adapting the contingency-based and uniformity-based null models to joint distributions over ranked
outputs, which we leave as a direction for future work.
•Prompt sensitivity analysis.We use a single fixed prompt template, not derived from a systematic prompt
engineering phase. This does not affect the validity of the proposed detection methodologies, but the magni-
tude and pattern of the measured bias may depend on the prompt design, so evaluating how different prompt
formulations modulate positional bias remains an important direction for future work.
•Other domains and richer user queries.Our experiments are restricted to product recommendation, al-
though the proposed methodologies apply without modification to other domains, such as news, music or
long-document recommendation. Since the position preferences are stable across decoding temperatures and,
for several models, across candidate-set sizes, we expect the differences between models to persist, although
their magnitude may vary: longer item descriptions may accentuate effects such as lost in the middle [13],
and queries encoding more user preferences may shift the balance between soft and strong positional bias.
•Independent detection of soft and strong positional bias.The proposed statistical tests detect the overall
positional bias, while strong positional bias is currently quantified through a metric based on relevance sets
estimated from the selection frequencies of the model itself. A natural research line is to develop statistical
methodologies that detect each component independently, with formal guarantees and robust to the noise
introduced by stochastic decoding and by the estimation of the relevance sets.
•Bias mitigation guided by detection metrics.The proposed metrics can guide mitigation, either by adding
PB as a regularization term when fine-tuning the recommender, as related work does in reranking set-
tings [18], or, without access to the model weights, by optimizing the prompt template with a genetic algo-
rithm whose fitness is the negated sPB on random batches of queries. Such an optimization should preserve
the recommendation task, e.g., by also penalizing PB, since a purely position-driven prompt could also lower
sPB, and be validated on held-out queries. More broadly, our framework provides a principled protocol to
benchmark any mitigation strategy.
Future work should therefore further extend positional bias analysis to top-N, multi-turn conversational and other
recommendation domains, incorporate explicit quality and utility metrics, and explore the interaction between po-
sitional bias and other structural biases such as popularity or exposure bias. Finally, the development of principled
22

Statistical Detection of Positional Bias in LLM-Based Recommender SystemsA PREPRINT
mitigation techniques and the integration of positional bias diagnostics into production-scale evaluation frameworks
remain important directions to advance the robustness and trustworthiness of LLM-based information systems.
Acknowledgments
This publication is part of the Project “Ethical, Responsible and General Purpose Artificial Intelligence: Applications
In Risk Scenarios” (IAFER) Exp.:TSI-100927-2023-1 funded through the Creation of university-industry research
programs (Enia Programs), aimed at the research and development of artificial intelligence, for its dissemination and
education within the framework of the Recovery, Transformation and Resilience Plan from the European Union Next
Generation EU through the Ministry for Digital Transformation and the Civil Service.
This publication used computational resources from the project "Thematic Center on Mountain Ecosystem & Remote
sensing, Deep learning-AI e-Services University of Granada-Sierra Nevada" (LifeWatch-2019-10-UGR-01), which
has been co-funded by the Ministry of Science and Innovation through the FEDER funds from the Spanish Plurire-
gional Operational Program 2014-2020 (POPE), LifeWatch-ERIC action line.
References
[1] Agresti, A.: Categorical Data Analysis. Wiley, Hoboken, NJ, USA, 3 edn. (2013)
[2] AI, M.: Introducing llama 4: Advancing multimodal intelligence (2024),https://ai.meta.com/blog/
llama-4-multimodal-intelligence/
[3] Chen, J., Dong, H., Wang, X., Feng, F., Wang, M., He, X.: Bias and Debias in Recommender System: A
Survey and Future Directions. ACM Trans. Inf. Syst.41(3) (Feb 2023), place: New York, NY , USA Publisher:
Association for Computing Machinery
[4] Collins, A., Tkaczyk, D., Aizawa, A., Beel, J.: Position Bias in Recommender Systems for Digital Libraries.
In: Chowdhury, G., McLeod, J., Gillet, V ., Willett, P. (eds.) Transforming Digital Worlds. pp. 335–344. Springer
International Publishing, Cham (2018)
[5] Deldjoo, Y ., He, Z., McAuley, J., Korikov, A., Sanner, S., Ramisa, A., Vidal, R., Sathiamoorthy, M., Kasirzadeh,
A., Milano, S.: A review of modern recommender systems using generative models (gen-recsys). In: Proceedings
of the 30th ACM SIGKDD conference on Knowledge Discovery and Data Mining. pp. 6448–6458 (2024)
[6] Gallegos, I.O., Rossi, R.A., Barrow, J., Tanjim, M.M., Kim, S., Dernoncourt, F., Yu, T., Zhang, R., Ahmed, N.K.:
Bias and fairness in large language models: A survey. Computational Linguistics50(3), 1097–1179 (Sep 2024).
doi:10.1162/coli_a_00524,https://aclanthology.org/2024.cl-3.8/
[7] Good, P.: Permutation Tests: A Practical Guide to Resampling Methods for Testing Hypotheses. Springer, New
York, NY , USA, 3 edn. (2013)
[8] Grattafiori, A., et al.: The Llama 3 Herd of Models (Nov 2024), arXiv:2407.21783 [cs]
[9] Guo, D., Yang, D., Zhang, H., et al.: Deepseek-r1 incentivizes reasoning in llms through reinforcement learning.
Nature645, 633–638 (2025). doi:10.1038/s41586-025-09422-z
[10] Hou, Y ., Zhang, J., Lin, Z., Lu, H., Xie, R., McAuley, J., Zhao, W.X.: Large Language Models are Zero-
Shot Rankers for Recommender Systems. In: Advances in Information Retrieval: 46th European Conference
on Information Retrieval, ECIR 2024, Glasgow, UK, March 24–28, 2024, Proceedings, Part II. pp. 364–381.
Springer-Verlag, Berlin, Heidelberg (Mar 2024)
[11] Hsieh, C.Y ., Chuang, Y .S., Li, C.L., Wang, Z., Le, L., Kumar, A., Glass, J., Ratner, A., Lee, C.Y ., Krishna,
R., Pfister, T.: Found in the middle: Calibrating positional attention bias improves long context utilization.
In: Ku, L.W., Martins, A., Srikumar, V . (eds.) Findings of the Association for Computational Linguistics:
ACL 2024. pp. 14982–14995. Association for Computational Linguistics, Bangkok, Thailand (Aug 2024).
doi:10.18653/v1/2024.findings-acl.890,https://aclanthology.org/2024.findings-acl.890/
[12] Li, L., Zhang, Y ., Liu, D., Chen, L.: Large language models for generative recommendation: A survey and
visionary discussions. In: Calzolari, N., Kan, M.Y ., Hoste, V ., Lenci, A., Sakti, S., Xue, N. (eds.) Proceedings
of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation
(LREC-COLING 2024). pp. 10146–10159. ELRA and ICCL, Torino, Italia (May 2024)
[13] Liu, N.F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., Liang, P.: Lost in the middle: How
language models use long contexts. Transactions of the Association for Computational Linguistics12, 157–173
(2024). doi:10.1162/tacl_a_00638,https://aclanthology.org/2024.tacl-1.9/
23

Statistical Detection of Positional Bias in LLM-Based Recommender SystemsA PREPRINT
[14] Liusie, A., Fathullah, Y ., Gales, M.: Teacher-student training for debiasing: General permutation debiasing for
large language models. In: Ku, L.W., Martins, A., Srikumar, V . (eds.) Findings of the Association for Compu-
tational Linguistics: ACL 2024. pp. 1376–1387. Association for Computational Linguistics, Bangkok, Thailand
(Aug 2024). doi:10.18653/v1/2024.findings-acl.81,https://aclanthology.org/2024.findings-acl.81/
[15] Navigli, R., Conia, S., Ross, B.: Biases in large language models: origins, inventory, and discussion. ACM
Journal of Data and Information Quality15(2), 1–21 (2023)
[16] OpenAI: gpt-oss-120b & gpt-oss-20b model card (2025),https://arxiv.org/abs/2508.10925
[17] Pezeshkpour, P., Hruschka, E.: Large language models sensitivity to the order of options in multiple-choice
questions. In: Duh, K., Gomez, H., Bethard, S. (eds.) Findings of the Association for Computational Linguistics:
NAACL 2024. pp. 2006–2017. Association for Computational Linguistics, Mexico City, Mexico (Jun 2024)
[18] Qiao, J., Huang, J., Ma, X., Wang, S., Yin, D., Kanoulas, E., Yates, A.: Llm-based listwise reranking under
the effect of positional bias. In: Campos, R., Jatowt, A., Lan, Y ., Aliannejadi, M., Bauer, C., MacAvaney, S.,
Anand, A., Ren, Z., Verberne, S., Bai, N., Mansoury, M. (eds.) Advances in Information Retrieval. pp. 131–146.
Springer Nature Switzerland, Cham (2026)
[19] Raiaan, M.A.K., Mukta, M.S.H., Fatema, K., Fahad, N.M., Sakib, S., Mim, M.M.J., Ahmad, J., Ali, M.E.,
Azam, S.: A Review on Large Language Models: Architectures, Applications, Taxonomies, Open Issues and
Challenges. IEEE Access12, 26839–26874 (2024)
[20] Shi, L., Ma, C., Liang, W., Diao, X., Ma, W., V osoughi, S.: Judging the judges: A systematic study of position
bias in LLM-as-a-judge. In: Inui, K., Sakti, S., Wang, H., Wong, D.F., Bhattacharyya, P., Banerjee, B., Ekbal, A.,
Chakraborty, T., Singh, D.P. (eds.) Proceedings of the 14th International Joint Conference on Natural Language
Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics.
pp. 292–314. The Asian Federation of Natural Language Processing and The Association for Computational
Linguistics, Mumbai, India (Dec 2025). doi:10.18653/v1/2025.ijcnlp-long.18,https://aclanthology.org/
2025.ijcnlp-long.18/
[21] Storey, V .C., Yue, W.T., Zhao, J.L., Lukyanenko, R., et al.: Generative artificial intelligence: Evolving technol-
ogy, growing societal impact, and opportunities for information systems research. Information Systems Frontiers
27, 2081–2102 (2025). doi:10.1007/s10796-025-10581-7
[22] Wang, L., Lim, E.P.: Zero-Shot Next-Item Recommendation using Large Pretrained Language Models (Apr
2023), arXiv:2304.03153 [cs]
[23] Wei, S.L., Wu, C.K., Huang, H.H., Chen, H.H.: Unveiling selection biases: Exploring order and token sen-
sitivity in large language models. In: Ku, L.W., Martins, A., Srikumar, V . (eds.) Findings of the Associa-
tion for Computational Linguistics: ACL 2024. pp. 5598–5621. Association for Computational Linguistics,
Bangkok, Thailand (Aug 2024). doi:10.18653/v1/2024.findings-acl.333,https://aclanthology.org/2024.
findings-acl.333/
[24] Wu, L., Zheng, Z., Qiu, Z., Wang, H., Gu, H., Shen, T., Qin, C., Zhu, C., Zhu, H., Liu, Q., et al.: A survey on
large language models for recommendation. World Wide Web27(5), 60 (2024)
[25] Xiao, S., Liu, Z., Zhang, P., Muennighoff, N., Lian, D., Nie, J.Y .: C-Pack: Packed Resources For General Chinese
Embeddings. In: Proceedings of the 47th International ACM SIGIR Conference on Research and Development
in Information Retrieval. pp. 641–649. SIGIR ’24, Association for Computing Machinery, New York, NY , USA
(Jul 2024)
[26] Yang, Z., Jian, P., Li, C.: Option symbol matters: Investigating and mitigating multiple-choice option symbol
bias of large language models. In: Chiruzzo, L., Ritter, A., Wang, L. (eds.) Proceedings of the 2025 Conference of
the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Tech-
nologies (V olume 1: Long Papers). pp. 1902–1917. Association for Computational Linguistics, Albuquerque,
New Mexico (Apr 2025)
[27] Zhang, M., Meng, Z., Collier, N.: Can We Instruct LLMs to Compensate for Position Bias? In: Al-Onaizan,
Y ., Bansal, M., Chen, Y .N. (eds.) Findings of the Association for Computational Linguistics: EMNLP 2024. pp.
12545–12556. Association for Computational Linguistics, Miami, Florida, USA (Nov 2024)
[28] Zheng, C., Zhou, H., Meng, F., Zhou, J., Huang, M.: Large Language Models Are Not Robust Multiple Choice
Selectors (2024)
[29] Zheng, L., Chiang, W.L., Sheng, Y ., Zhuang, S., Wu, Z., Zhuang, Y ., Lin, Z., Li, Z., Li, D., Xing, E.P., Zhang,
H., Gonzalez, J.E., Stoica, I.: Judging llm-as-a-judge with mt-bench and chatbot arena. In: Proceedings of the
37th International Conference on Neural Information Processing Systems. NIPS ’23, Curran Associates Inc.,
Red Hook, NY , USA (2023)
24

Statistical Detection of Positional Bias in LLM-Based Recommender SystemsA PREPRINT
A Proof of Theorem 1 and Proposition 2
Theorem 1.Underε-absence of positional bias, we haveTV(p(·), U k)≤ε, whereU kdenotes the uniform distribu-
tion over{1,2, . . . , k}. In fact, for all instancesx,TV(p(· |x), U k)≤ε.
Proof of Theorem 1.Fix an instanceX=xand a reference permutationφ 0∈Sk(e.g., the identity). Writer σ(j) =
Pr(P j|x, σ)for the distribution over items under the orderingσ. Recall that if the item at positioniin the prompt is
picked under orderingσ, then the item selected isP σ(i). Thus, for anyσ∈S k, we have
p(i|x) =E σ∼U(S k)[rσ(σ(i))].
Sinceσ(i)is uniformly distributed over{1, . . . , k}whenσ∼U(S k), for the fixed permutationφ 0we obtain
Eσ∼U(S k)[rφ0(σ(i))] =1
kkX
j=1rφ0(j) =1
k.
Therefore, combining the two equations above, we find that
p(i|x)−1
k=Eσ∼U(S k)[rσ(σ(i))]−E σ∼U(S k)[rφ0(σ(i))] =E σ[rσ(σ(i))−r φ0(σ(i))],
and we have
kX
i=1p(i|x)−1
k=kX
i=1|Eσ[rσ(σ(i))−r φ0(σ(i))]|
≤kX
i=1Eσ|rσ(σ(i))−r φ0(σ(i))|,
where we have used the inequality|E[Y]| ≤E[|Y|]. By using the definition of expectation and reordering the sums, it
follows
kX
i=1Eσ|rσ(σ(i))−r φ0(σ(i))|=kX
i=1kX
j=1X
σ∈S k:σ(i)=j1
k!|rσ(j)−r φ0(j)|
=kX
j=1X
σ∈S k1
k!|rσ(j)−r φ0(j)|
=X
σ∈S k1
k!kX
j=1|rσ(j)−r φ0(j)|
=Eσ∼U(S k)
kX
j=1|rσ(j)−r φ0(j)|

= 2E σ[TV(r σ, rφ0)]
≤2ε,
where the last inequality follows from Definition 1. Dividing both sides by2yields
TV(p(·|x), U k) =1
2kX
i=1p(i|x)−1
k≤1
2kX
i=1Eσ|rσ(σ(i))−r φ0(σ(i))| ≤ε,
as we wanted to prove. Now we tackle the unconditional distributionp(i) =E X[p(i|X)]. Indeed, by convexity of
total variation distance and Jensen’s inequality,
TV(p(·), U k) = TV(E X[p(· |X)], U k)≤E X[TV(p(· |X), U k)]≤ε.
Proposition 2.Fixk≥2,ℓ≥1andα∈(0,1), and letLbe the bound defined in Equation 7, computed fromn
independent instances. Then
lim inf
n→∞Pr[L≤TV(p(·), U k) ]≥1−α.
25

Statistical Detection of Positional Bias in LLM-Based Recommender SystemsA PREPRINT
Proof.We first show thatlim supn→∞ Pr[L i> µ i]≤α/kfor everyi∈ {1,2, . . . , k}. Fixi. The variables
Z1(i), Z 2(i), . . .are independent and identically distributed with meanµ i, and they are bounded since|Z j(i)| ≤1.
We distinguish two cases.
• CaseVar(Z 1(i)) = 0. ThenZ j(i) =µ ialmost surely for everyj, so thats i= 0andL i=µ i. Hence
Pr[L i> µi] = 0.
• CaseVar(Z 1(i))>0. LetT n(i) =√n(¯Z(i)−µ i)/si, with the conventionT n(i) = 0whens i= 0. By
the strong law of large numbers,s iconverges almost surely toVar(Z 1(i))1/2>0, soPr[s i= 0]→0.
Moreover, by the central limit theorem and Slutsky’s theorem,T n(i)converges in distribution to a standard
normal random variable. On the events i>0, we haveL i> µiif and only ifT n(i)> t 1−α/k, n−1 . Since
t1−α/k, n−1 →z 1−α/k asn→ ∞, wherez βdenotes theβ-quantile of the standard normal distribution, we
obtain
lim
n→∞Pr[L i> µi] = lim
n→∞Pr
Tn(i)> t 1−α/k, n−1
=α
k.
Now letE nbe the event “L i≤µifor alli”, and letEC
nbe its complement. By the union bound,
lim sup
n→∞Pr[EC
n]≤kX
i=1lim sup
n→∞Pr[L i> µi]≤α,
solim inf n→∞ Pr[E n]≥1−α. IfE nholds, the monotonicity ofx7→x +gives(L i)+≤(µ i)+for everyi, and
therefore
L=kX
i=1(Li)+≤kX
i=1(µi)+= TV(p(·), U k),
where the last equality is Equation 5. That is,E nimpliesL≤TV(p(·), U k), and we conclude that
1−α≤lim inf
n→∞Pr[E n]≤lim inf
n→∞Pr[L≤TV(p(·), U k)].
B Simulation study: Type-I error and statistical power
To assess the statistical validity and sample efficiency of POB-ICM and POB-UNBP independently of any specific
LLM, we run a simulation study on synthetic data with known ground-truth bias. We measure (i) empirical Type-I
error under the null hypothesis of no positional bias, and (ii) statistical power as a function of both bias strength and
sample size.
For each synthetic query8, we draw a uniformly random, non-empty relevance setR(X)⊆ {1, . . . , k}. For each
sampled permutationσ, the simulated recommender selects an item uniformly fromR(X)with probability1−p bias,
and otherwise (probabilityp bias) selects the item placed first byσ. Thusp bias= 0is an exact null model of no
positional bias, andp bias>0injects a known amount of strong positional bias; all experiments below are repeated for
k∈ {5,10,15,20}.
Experiment 1: Type-I error.We estimate the empirical Type-I error (false positive rate) of both tests, where synthetic
datasets with no induced bias are sampled and tested. We fixp bias= 0,n queries = 2000,n σ= 20(matching
subsection 4.1), and estimate the rejection rate atα= 0.05over1000replicates. A well calibratedp-value should
produce a type-I error equal to the nominal levelαon expectation. Table 8 shows POB-UNBP stays well calibrated at
everyk(0.039–0.053), while POB-ICM is calibrated atk= 5,10but becomes conservative fork≥15(0.016–0.020),
since most columns of its contingency matrix contain a single observation, which makes the Chi-square statistic nearly
constant across the permutation replicates and produces many ties with the observed value.
Experiment 2: power vs. bias strength.We measure power as a function of bias magnitude. Fixingn queries = 500,
nσ= 20, we sweepp bias∈(0,0.1)and estimate power over200replicates per point. Figure 8 shows POB-UNBP
saturates byp bias≈0.02atk= 5, and at even smallerp biasfor largerk, while POB-ICM needs roughlyp bias∈
(0.02,0.06)atk= 5and fails to gain power at all fork≥10.
Experiment 3: minimum sample size.We estimate the minimum number of queries needed for0.8power. Fixing
pbias= 0.03, we grid-searchn queries ∈[25,2500],n σ∈[2,80](60 replicates/cell), interpolating the smallestn queries
8In this context, a synthetic query is a numerical index with no associated text, unlike the queries used in subsection 4.1.
26

Statistical Detection of Positional Bias in LLM-Based Recommender SystemsA PREPRINT
Method k= 5k= 10k= 15k= 20
POB-ICM 0.055 0.050 0.016 0.020
POB-UNBP 0.053 0.044 0.043 0.039
Table 8: Type-I error (false positive rate) in the simula-
tion,1000synthetic dataset samples with no positional
bias induced.
nσPOB-ICM POB-UNBP
k= 5 k= 5k= 10k= 15k= 20
2 ≥2500 1393 707 688 562
5 ≥2500 659 368 338 225
10 1983 344 214 181 168
20 1068 175 98 72 65
40 589 88 46 44 39
80 246 40≤25≤25≤25
Table 9: Required number of queries to reach a power
of0.8, conditioned on the number of permutations per
query,n σ, used in the experiment.
Figure 8: Power depending on the bias strength, mea-
sured as the probability of switch the decision to the first
prompt position.
reaching80%power pern σ. Table 9 shows POB-UNBP requires approximately six times fewer queries than POB-
ICM atk= 5(e.g.,175vs.1,068atn σ= 20), and this requirement decreases further withk(65queries atk= 20);
POB-ICM never reaches80%power fork≥10, so it is reported only atk= 5.
Taken together, both tests are valid (Type-I error near nominal), and POB-UNBP is consistently more sample-efficient
than POB-ICM, an advantage that grows rather than shrinks withk. This motivates relying on POB-UNBP for the
candidate-set-size analysis in subsubsection 4.4.2 and reporting POB-ICM only atk= 5(Table 2); it also confirms
that then queries = 2000,n σ= 20operating point used throughout section 4 affords high power for both tests.
C Separating stochastic decoding variability from positional bias
Under non-zero temperature, the recommended item may change between two evaluations of the same instance either
because the candidate items were permuted or because of the randomness of the decoding itself. To disentangle
both mechanisms, we evaluate the top-5setting under three counterfactual configurations: (i)decoding only, with
temperaturet= 0.6and the identity permutation for every evaluation; (ii)permutation only, witht= 0and uniformly
sampled permutations; and (iii)both, witht= 0.6and uniformly sampled permutations. Configurations (ii) and (iii)
are the evaluations reported in subsection 4.4.
For every configuration, we compute three quantities on20evaluations per instance. The first is the instance-level
strong positional bias, that is, the fraction of evaluations whose recommended item lies outside the relevance setR(X)
(Equation 11). The second is the lower boundLof POB-UNBP (Equation 7), in percentage points, which quantifies
the overall positional bias; it is reported only for configurations (ii) and (iii), since under configuration (i) the order of
the items is fixed. The third is the intersection over union (IoU) between the relevance set of the instance and the one
obtained under configuration (i), averaged over instances. Relevance sets are computed as described in subsection 4.2,
with the folds and thresholdsτof configuration (ii) shared by the three configurations, so that the relevance set is
defined identically across them.9Confidence intervals for sPB are obtained by percentile bootstrap at the query level,
keeping the folds, the thresholdsτand hence the relevance sets fixed. They therefore reflect the sampling variability
of the queries conditionally on the relevance sets, which is the relevant uncertainty when comparing configurations
that share them, and are narrower than those of Appendix E, whereτis recalibrated in every resample.
9Hence, the sPB values of configuration (ii) coincide with Table 6, whereas those of configuration (iii) may differ from Table 7,
whereτis calibrated on the evaluations att= 0.6; the differences are largest for DeepSeek R1 8B and LLaMA 3.1 8B, due to the
discreteness of the selection frequencies discussed in Appendix E.
27

Statistical Detection of Positional Bias in LLM-Based Recommender SystemsA PREPRINT
Model namesPB L IoU ofR(X)p-value(i) (ii) (iii) (ii) (iii) (i), (ii) (i), (iii)
DeepSeek R1 32B 2.9 (2.8, 3.1) 3.1 (2.9, 3.2) 3.4 (3.3, 3.6) 1.0 1.1 78.9 79.8 2.2e-08
DeepSeek R1 7B 9.1 (8.8, 9.4) 9.8 (9.5, 10.1) 10.2 (9.9, 10.5) 4.2 4.4 70.2 70.6 7.4e-09
DeepSeek R1 8B 6.6 (6.3, 6.8) 6.7 (6.5, 6.9) 7.5 (7.3, 7.8) 0.8 1.6 73.6 73.4 4.3e-11
LLaMA 3.1 8B 6.9 (6.7, 7.2) 6.7 (6.4, 6.9) 7.5 (7.3, 7.7) 3.0 2.0 74.3 74.5 1.6e-08
LLaMA 4 Scout 2.3 (2.1, 2.5) 3.2 (3.0, 3.4) 3.3 (3.1, 3.5) 0.9 1.0 76.4 76.6 5.1e-19
GPT OSS 120B 0.9 (0.9, 1.0) 1.0 (1.0, 1.1) 1.1 (1.0, 1.1) 3.1 3.2 74.8 75.1 0.104
GPT-OSS 20B 2.5 (2.3, 2.6) 2.1 (2.0, 2.2) 2.3 (2.1, 2.4) 4.7 4.9 72.8 74.2 0.194
LLaMA 3.1 70B 3.2 (3.0, 3.3) 3.1 (3.0, 3.3) 3.5 (3.3, 3.7) 3.4 4.5 78.5 79.6 0.003
Table 10: Strong positional bias (sPB), POB-UNBP lower boundLofTV(p(·), U k)for the configurations with
permutations, and mean intersection over union (IoU) of the relevance sets with configuration (i), for top-5under
configurations (i) stochastic decoding, (ii) random permutation and (iii) both. sPB and IoU are percentages andL
is in percentage points, with95%confidence intervals for sPB. The last column reports the Holm-adjustedp-value
comparing the sPB of configurations (iii) and (i); bold values are significant at the0.05level.
To test whether permutations increase strong positional bias beyond the effect of stochastic decoding, we compare the
instance-level sPB of configurations (iii) and (i), which share the same temperature and differ only in the permutations.
Since both configurations evaluate the same2000instances, we apply a two-sided paired Wilcoxon signed-rank test,
discarding instances with identical values in both configurations, and adjust the resultingp-values for the eight models
with the Holm procedure, which controls the family-wise error rate at0.05.
Overall positional bias.Stochastic decoding does not remove the overall positional bias:Lremains of similar
magnitude with and without temperature, between0.8and4.7percentage points under configuration (ii) and between
1.0and4.9under configuration (iii), consistently with Table 4.
Strong positional bias beyond decoding noise.Recommendations outside the relevance set are rare in all config-
urations, with sPB between0.9%and10.2%, but they depend on the source of variability. Adding permutations to
stochastic decoding, configuration (iii) versus (i), significantly increases sPB for six of the eight models, by0.3to
1.1percentage points, the exceptions being the two GPT-OSS models, whose sPB is the lowest among the evaluated
models. Permutations alone, configuration (ii), yield an sPB comparable to that of stochastic decoding alone for most
models, and notably larger for LLaMA 4 Scout (3.2%against2.3%) and DeepSeek R1 7B (9.8%against9.1%).
Therefore, part of the strong positional bias observed under stochastic decoding is attributable to the permutation of
the candidate items and is not explained by temperature alone.
The GPT-OSS models are a notable exception. Their sPB under permutations does not exceed the level observed under
decoding only, and the95%confidence interval of the paired difference between configurations (iii) and (i) bounds
any increase of sPB due to permutations to at most0.2percentage points. Since sPB also captures recommendations
outside the relevance set caused by decoding noise and by the estimation of the relevance set from a limited number of
evaluations, we find no evidence of strong positional bias in these models beyond this noise level, although the identity
permutation used in configuration (i) prevents a fully clean comparison for them. Combined with their significant
overall positional bias (Table 2), this indicates that the positional bias of the GPT-OSS models is essentially soft: their
marked preference for the first prompt positions acts as a tie-breaker among comparably relevant candidates rather
than displacing the recommendation towards clearly less relevant items.
Stability of the relevance set.The mean intersection over union (IoU) between the relevance sets obtained under
decoding only and under permutations ranges from70%to80%, with almost identical values whether or not temper-
ature is added to the permutations. Hence, the relevance sets largely overlap across sources of variability, and their
differences typically involve a single item. Agreement is lowest for DeepSeek R1 7B, the model with the highest sPB,
whose recommendations are more dispersed across candidate items, and highest for DeepSeek R1 32B and LLaMA
3.1 70B. Since every relevance set is estimated from only20evaluations, part of these differences reflects estimation
noise in the items closest to the threshold.
One caveat applies: configuration (i) fixes the identity permutation, so the product ranked first by the semantic matcher
always occupies the first prompt position, which the GPT-OSS models strongly favor (Figure 5); this couples positional
preference with retrieval rank and may explain why the GPT-OSS models show no significant differences.
28

Statistical Detection of Positional Bias in LLM-Based Recommender SystemsA PREPRINT
Model namesPB L IoU ofR(X)p-value(i) (ii) (iii) (ii) (iii) (i), (ii) (i), (iii)
DeepSeek R1 32B 2.6 (2.4, 2.8) 3.1 (2.9, 3.2) 3.5 (3.3, 3.6) 1.0 2.6 78.6 78.9 3.0e-18
DeepSeek R1 7B 9.5 (9.2, 9.8) 9.8 (9.5, 10.1) 10.4 (10.1, 10.7) 4.2 3.9 71.0 71.3 2.4e-07
DeepSeek R1 8B 6.0 (5.8, 6.3) 6.7 (6.5, 6.9) 7.2 (6.9, 7.4) 0.8 3.7 72.4 72.4 1.3e-14
LLaMA 3.1 8B 6.0 (5.7, 6.2) 6.7 (6.4, 6.9) 7.3 (7.0, 7.5) 3.0 0.3 73.9 74.0 7.8e-21
LLaMA 4 Scout 2.7 (2.5, 2.9) 3.2 (3.0, 3.4) 3.2 (3.1, 3.4) 0.9 0.9 79.2 80.6 2.4e-07
GPT OSS 120B 1.0 (0.9, 1.1) 1.0 (1.0, 1.1) 1.5 (1.4, 1.6) 3.1 4.3 77.3 79.9 1.0e-14
GPT-OSS 20B 2.3 (2.2, 2.5) 2.1 (2.0, 2.2) 3.0 (2.8, 3.1) 4.7 7.5 73.9 76.7 5.7e-13
LLaMA 3.1 70B 2.4 (2.3, 2.6) 3.1 (3.0, 3.3) 3.2 (3.1, 3.4) 3.4 5.3 76.2 77.5 2.4e-13
Table 11: Strong positional bias (sPB), POB-UNBP lower boundLofTV(p(·), U k)for the configurations with per-
mutations, and mean intersection over union (IoU) of the relevance sets with configuration (i), for top-5under config-
urations (i) random labels, (ii) random permutation and (iii) both. sPB and IoU are percentages andLis in percentage
points, with95%confidence intervals for sPB. The last column reports the Holm-adjustedp-value comparing the sPB
of configurations (iii) and (i); bold values are significant at the0.05level.
D Separating item identifiers from positional bias
In the prompt of Prompt 1, each item is identified by its prompt position (“Title 1”, “Description 1”, etc.), and the model
returns this identifier in the fieldarticle_number. Hence, the position of an item and its identifier are coupled, and
part of the observed effect could reflect a preference for particular identifiers rather than for prompt positions. To
disentangle both, we evaluate the top-5setting with random labels: the identifiers1, . . . , kare assigned to the prompt
positions by a random permutation, drawn independently of the query and of the item permutation, and the label
returned by the model is mapped back to its prompt position before the analysis. As in Appendix C, we consider three
configurations att= 0: (i)random labels only, with the identity permutation; (ii)permutation only, with positional
labels and uniformly sampled permutations, i.e., the evaluation reported in subsection 4.4; and (iii)both, with random
labels and uniformly sampled permutations. sPB,Land the relevance sets are computed as in Appendix C, with the
folds and thresholds of configuration (ii), and configurations are compared with the same paired Wilcoxon signed-rank
tests and Holm correction.
Identifiers alone.With a fixed order, random labels already change the recommended item: the sPB of configuration
(i), between1.0%and9.5%, is comparable to the noise floor of stochastic decoding (Appendix C). Hence, the
identifiers act as a perturbation of the prompt, similar to decoding noise.
Overall positional bias beyond identifiers.With random labels, configuration (iii), POB-UNBP rejects the null
hypothesis of uniformity for all eight models (p≤3.5·10−4), so the overall positional bias is not explained by the
identifiers either. Its magnitude, however, depends on them:Lincreases for five models, most markedly for GPT-
OSS 20B (4.7to7.5percentage points), remains similar for DeepSeek R1 7B and LLaMA 4 Scout, and drops for
LLaMA 3.1 8B (3.0to0.3), whose preference for the last prompt positions under positional labels thus largely reflects
a preference for the corresponding identifiers.
Strong positional bias beyond identifiers.Adding permutations to random labels, configuration (iii) versus (i),
significantly increases sPB for all eight models, by0.5to1.3percentage points. Moreover, decoupling identifiers from
positions never reduces sPB: compared with positional labels, configuration (iii) versus (ii), sPB increases significantly
for six models and is unchanged for LLaMA 4 Scout and LLaMA 3.1 70B. Therefore, the strong positional bias is not
explained by the identifiers assigned to the items.
The increase of sPB from (ii) to (iii) is largest for the GPT-OSS models (1.0%to1.5%and2.1%to3.0%), whose
overall positional bias also increases (Lfrom3.1to4.3and from4.7to7.5percentage points). Hence, their preference
for the first prompt positions is not a preference for the identifier “1”. A possible explanation is that, when identifiers
no longer follow the order of the prompt, these models rely more on the position of the items, which also displaces
part of their recommendations outside the relevance set. In any case, the essentially soft positional bias of the GPT-
OSS models observed in Appendix C holds under positional labels, whereas their sPB increases when identifiers and
positions are decoupled.
Stability of the relevance set.The mean IoU between the relevance sets obtained with random labels only and under
permutations ranges from71%to81%, similar to the agreement observed for stochastic decoding (Appendix C). The
29

Statistical Detection of Positional Bias in LLM-Based Recommender SystemsA PREPRINT
Model nameTop-k
k= 5k= 10k= 15k= 20
PS
DeepSeek R1 32B 21.9 (21.0, 22.7) 29.4 (28.5, 30.4) 34.1 (33.1, 35.2) 37.2 (36.2, 38.3)
DeepSeek R1 7B 34.0 (33.1, 34.9) 45.1 (44.2, 45.9) 52.7 (51.8, 53.6) 58.1 (57.3, 59.0)
DeepSeek R1 8B 29.8 (28.8, 30.7) 38.6 (37.7, 39.5) 43.2 (42.3, 44.2) 46.4 (45.5, 47.4)
LLaMA 3.1 8B 30.4 (29.5, 31.3) 41.9 (41.0, 42.8) 46.9 (46.0, 47.8) 49.6 (48.6, 50.5)
LLaMA 4 Scout 22.4 (21.5, 23.3) 38.3 (37.3, 39.2) 51.5 (50.6, 52.3) 61.6 (60.9, 62.4)
GPT OSS 120B 15.5 (14.7, 16.3) 21.9 (21.0, 22.9) 24.9 (23.9, 25.9) 27.3 (26.3, 28.3)
GPT-OSS 20B 17.7 (16.8, 18.5) 24.3 (23.3, 25.3) 28.3 (27.2, 29.3) 30.9 (29.9, 32.0)
LLaMA 3.1 70B 22.7 (21.8, 23.6) 31.7 (30.7, 32.8) 36.7 (35.7, 37.7) 40.2 (39.2, 41.3)
sPB
DeepSeek R1 32B 3.1 (2.9, 3.2) 6.9 (6.7, 8.5) 10.7 (10.4, 11.1) 13.7 (11.3, 14.2)
DeepSeek R1 7B 9.8 (9.1, 10.2) 20.1 (19.2, 20.5) 26.6 (25.8, 28.2) 33.2 (32.1, 35.1)
DeepSeek R1 8B 6.7 (6.5, 7.0) 14.8 (14.4, 15.1) 19.4 (17.3, 19.8) 20.6 (20.2, 21.1)
LLaMA 3.1 8B 6.7 (6.4, 6.9) 15.7 (15.3, 16.1) 20.7 (20.0, 21.2) 24.1 (23.2, 25.0)
LLaMA 4 Scout 3.2 (3.0, 3.4) 12.9 (10.9, 14.4) 24.5 (23.9, 25.4) 36.6 (35.6, 37.6)
GPT OSS 120B 1.0 (1.0, 1.3) 3.4 (3.2, 3.6) 5.9 (5.3, 6.2) 6.7 (6.4, 6.9)
GPT-OSS 20B 2.1 (1.2, 2.5) 5.6 (4.0, 5.9) 6.7 (6.5, 7.6) 9.4 (8.5, 9.9)
LLaMA 3.1 70B 3.1 (3.0, 3.3) 9.5 (8.0, 9.8) 12.2 (10.9, 14.3) 14.9 (14.5, 15.3)
Table 12: Permutation sensitivity (PS) and strong positional bias (sPB) across candidate-set sizeskat temperature
t= 0, with95%confidence intervals obtained by percentile bootstrap at the query level. Values are percentages.
Model nameTop-k
k= 5 k= 10 k= 15 k= 20
Min. Max. Min. Max. Min. Max. Min. Max.
DeepSeek R1 32B 86.8 87.4 80.6 80.8 76.7 77.1 74.0 74.7
DeepSeek R1 7B 78.4 78.8 67.3 68.3 60.6 61.1 55.0 55.7
DeepSeek R1 8B 80.8 81.6 72.9 73.5 68.8 69.5 65.9 66.9
LLaMA 3.1 8B 81.7 82.1 71.4 71.9 66.4 67.0 63.4 64.1
LLaMA 4 Scout 86.6 86.9 74.8 75.1 63.1 64.0 52.5 53.0
GPT OSS 120B 90.5 91.1 85.8 86.5 83.7 84.1 81.6 82.0
GPT-OSS 20B 89.6 90.0 84.2 84.7 80.7 81.2 78.6 79.1
LLaMA 3.1 70B 86.2 86.7 78.9 79.6 74.7 75.2 71.9 72.3
Table 13: Minimum and maximum of the thresholdτof sPB across the five folds, for candidate-set sizeskat temper-
aturet= 0. Values are percentages.
caveat of Appendix C also applies: configuration (i) uses the identity permutation, so the product ranked first by the
retriever always occupies the first prompt position.
E Confidence intervals for permutation sensitivity and strong positional bias
This appendix reports95%confidence intervals for the permutation sensitivity (PS) and strong positional bias (sPB)
values of Table 6 and Table 7, obtained by percentile bootstrap at the query level withB= 2000resamples (Table 12
and Table 14). It also reports the thresholdsτacross folds computed to estimate the sPB metric.
Permutation sensitivity.PS is an average of per-query values, so each bootstrap resample recomputes this average
over the resampled queries. The intervals are narrow, about±1percentage point. The increase of PS withkis robust,
as the intervals for consecutive values ofknever overlap. The increase with temperature is mild: the intervals att= 0
andt= 0.6are disjoint only for DeepSeek R1 8B and LLaMA 3.1 8B.
Strong positional bias.Since sPB depends on a thresholdτcalibrated from the data (subsection 4.2), each bootstrap
resample repeats the whole estimation procedure: the partition into five folds, the calibration ofτon the training folds,
30

Statistical Detection of Positional Bias in LLM-Based Recommender SystemsA PREPRINT
Model nameTemperaturet
t= 0.0t= 0.2t= 0.4t= 0.6
PS
DeepSeek R1 32B 21.9 (21.0, 22.7) 22.7 (21.8, 23.6) 22.9 (22.0, 23.8) 23.4 (22.5, 24.3)
DeepSeek R1 7B 34.0 (33.1, 34.9) 34.6 (33.8, 35.5) 35.3 (34.4, 36.2) 35.9 (35.0, 36.8)
DeepSeek R1 8B 29.8 (28.8, 30.7) 30.9 (30.0, 31.8) 31.5 (30.5, 32.3) 32.6 (31.7, 33.4)
LLaMA 3.1 8B 30.4 (29.5, 31.3) 31.4 (30.5, 32.3) 32.3 (31.4, 33.1) 33.0 (32.0, 33.8)
LLaMA 4 Scout 22.4 (21.5, 23.3) 22.6 (21.7, 23.5) 22.6 (21.8, 23.5) 22.9 (21.9, 23.7)
GPT OSS 120B 15.5 (14.7, 16.3) 15.6 (14.8, 16.4) 15.7 (15.0, 16.5) 16.0 (15.2, 16.8)
GPT-OSS 20B 17.7 (16.8, 18.5) 17.9 (17.0, 18.8) 18.6 (17.7, 19.5) 18.6 (17.7, 19.4)
LLaMA 3.1 70B 22.7 (21.8, 23.6) 22.8 (21.9, 23.8) 23.7 (22.8, 24.6) 24.2 (23.3, 25.1)
sPB
DeepSeek R1 32B 3.1 (2.9, 3.2) 3.3 (3.1, 3.5) 3.4 (3.2, 4.0) 3.5 (3.3, 5.0)
DeepSeek R1 7B 9.8 (9.1, 10.2) 10.0 (9.6, 10.5) 10.4 (9.9, 10.8) 10.4 (10.0, 10.8)
DeepSeek R1 8B 6.7 (6.5, 7.0) 6.9 (6.7, 9.1) 8.7 (7.0, 9.9) 9.9 (9.4, 10.2)
LLaMA 3.1 8B 6.7 (6.4, 6.9) 7.1 (6.8, 8.7) 9.3 (7.0, 9.8) 9.4 (8.1, 10.1)
LLaMA 4 Scout 3.2 (3.0, 3.4) 3.2 (3.0, 3.4) 3.2 (3.1, 3.4) 3.3 (3.1, 3.5)
GPT OSS 120B 1.0 (1.0, 1.3) 1.0 (1.0, 1.4) 1.2 (1.1, 1.7) 1.1 (1.0, 1.6)
GPT-OSS 20B 2.1 (1.2, 2.5) 2.5 (2.1, 2.7) 2.6 (1.7, 2.8) 2.7 (2.4, 2.8)
LLaMA 3.1 70B 3.1 (3.0, 3.3) 3.4 (3.3, 3.6) 3.5 (3.3, 4.0) 3.5 (3.3, 4.0)
Table 14: Permutation sensitivity (PS) and strong positional bias (sPB) across decoding temperaturestat top-5, with
95%confidence intervals obtained by percentile bootstrap at the query level. Values are percentages.
Model nameTemperaturet
t= 0.0 t= 0.2 t= 0.4 t= 0.6
Min. Max. Min. Max. Min. Max. Min. Max.
DeepSeek R1 32B 86.8 87.4 86.1 86.2 85.6 86.0 85.1 85.3
DeepSeek R1 7B 78.4 78.8 77.9 78.2 77.7 78.6 76.9 77.6
DeepSeek R1 8B 80.8 81.6 80.1 80.8 79.7 80.1 78.5 79.1
LLaMA 3.1 8B 81.7 82.1 80.2 80.9 79.4 79.7 79.0 79.6
LLaMA 4 Scout 86.6 86.9 86.0 86.7 86.4 86.8 86.4 86.6
GPT OSS 120B 90.5 91.1 90.6 90.9 90.6 90.8 90.6 91.0
GPT-OSS 20B 89.6 90.0 89.1 89.4 89.1 89.7 88.8 89.2
LLaMA 3.1 70B 86.2 86.7 86.0 86.6 85.6 86.0 85.4 86.2
Table 15: Minimum and maximum of the thresholdτof sPB across the five folds, for decoding temperaturestat top-5.
Values are percentages.
and the computation of the relevance sets and sPB on the held-out fold. Copies of the same instance are always
assigned to the same fold. Most intervals are narrow, but some are asymmetric, e.g., DeepSeek R1 32B atk= 10,
because with20evaluations per instance the item frequencies lie on a grid of step0.05, so small variations ofτacross
resamples can move many instances between relevance sets of different sizes at once. The increase of sPB withkis
robust, as the intervals atk= 5andk= 20are far apart for every model, whereas its differences across temperatures
are of the same order as the interval widths, so sPB is essentially stable with respect to temperature, in agreement with
Appendix C.
The calibrated thresholds are stable across folds: their minimum and maximum differ by at most1percentage point
in every setting (Table 13 and Table 15). They decrease withk, as the recommendations of each instance spread over
more candidates, decrease slightly with temperature for most models, and are highest for the GPT-OSS models, whose
recommendations are the most concentrated.
31

Statistical Detection of Positional Bias in LLM-Based Recommender SystemsA PREPRINT
k= 5 k= 10 k= 15 k= 20
Model name ∆sPB/SD (95% CI) Rel. ∆sPB/SD (95% CI) Rel. ∆sPB/SD (95% CI) Rel. ∆sPB/SD (95% CI) Rel.
DeepSeek R1 32B -0.08 (-0.25, 0.09) -2.6% -0.41 (-0.66, -0.16) -5.8% -0.58 (-0.89, -0.27) -5.4% -0.66 (-1.05, -0.26) -4.8%
DeepSeek R1 7B -0.26 (-0.58, 0.05) -2.7% -0.60 (-0.96, -0.23) -3.0% -1.09 (-1.49, -0.68) -4.1% -0.95 (-1.33, -0.56) -2.9%
DeepSeek R1 8B -0.30 (-0.54, -0.05) -4.4% -0.84 (-1.19, -0.49) -5.7% -0.59 (-0.98, -0.20) -3.0% -0.94 (-1.32, -0.55) -4.5%
LLaMA 3.1 8B -0.22 (-0.47, 0.03) -3.3% -0.81 (-1.17, -0.46) -5.2% -0.90 (-1.29, -0.50) -4.3% -0.60 (-1.02, -0.17) -2.5%
LLaMA 4 Scout -0.30 (-0.48, -0.12) -9.3% -0.85 (-1.22, -0.48) -6.1% -0.89 (-1.27, -0.51) -3.7% -1.08 (-1.44, -0.72) -3.0%
GPT OSS 120B -0.01 (-0.10, 0.08) -1.0% -0.20 (-0.38, -0.02) -5.9% -0.38 (-0.63, -0.13) -6.4% -0.48 (-0.73, -0.23) -7.2%
GPT-OSS 20B -0.16 (-0.32, 0.01) -6.8% -0.39 (-0.64, -0.15) -7.2% -0.54 (-0.80, -0.28) -8.0% -0.49 (-0.82, -0.16) -5.1%
LLaMA 3.1 70B -0.07 (-0.24, 0.11) -2.1% -0.38 (-0.69, -0.07) -4.0% -0.35 (-0.74, 0.03) -2.5% -0.63 (-1.03, -0.22) -4.2%
Table 16: Change of sPB, in percentage points, per standard deviation of retrieval score dispersion (95% CI), and
relative to the sPB of each model (Rel.), across candidate-set sizes att= 0. Bold: significant at the0.05level after
Holm correction.
t= 0.0 t= 0.2 t= 0.4 t= 0.6
Model name ∆sPB/SD (95% CI) Rel. ∆sPB/SD (95% CI) Rel. ∆sPB/SD (95% CI) Rel. ∆sPB/SD (95% CI) Rel.
DeepSeek R1 32B -0.08 (-0.25, 0.09) -2.6% -0.07 (-0.25, 0.11) -2.2% -0.19 (-0.37, -0.02) -5.7% -0.05 (-0.23, 0.13) -1.5%
DeepSeek R1 7B -0.26 (-0.58, 0.05) -2.7% -0.66 (-0.96, -0.36) -6.6% -0.74 (-1.04, -0.43) -7.1% -0.38 (-0.68, -0.07) -3.6%
DeepSeek R1 8B -0.30 (-0.54, -0.05) -4.4% -0.16 (-0.40, 0.09) -2.2% -0.42 (-0.72, -0.13) -4.4% -0.51 (-0.81, -0.20) -5.1%
LLaMA 3.1 8B -0.22 (-0.47, 0.03) -3.3% -0.33 (-0.58, -0.08) -4.7% -0.55 (-0.85, -0.26) -5.9% -0.17 (-0.46, 0.12) -1.8%
LLaMA 4 Scout -0.30 (-0.48, -0.12) -9.3% -0.17 (-0.34, 0.00) -5.3% -0.34 (-0.51, -0.17) -10.4% -0.29 (-0.46, -0.12) -8.7%
GPT OSS 120B -0.01 (-0.10, 0.08) -1.0% -0.06 (-0.15, 0.03) -5.4% -0.06 (-0.15, 0.04) -4.7% -0.02 (-0.11, 0.07) -2.0%
GPT-OSS 20B -0.16 (-0.32, 0.01) -6.8% -0.10 (-0.27, 0.06) -4.1% -0.12 (-0.30, 0.05) -4.8% -0.16 (-0.32, 0.01) -5.9%
LLaMA 3.1 70B -0.07 (-0.24, 0.11) -2.1% -0.23 (-0.41, -0.05) -6.8% -0.16 (-0.33, 0.00) -4.8% -0.13 (-0.30, 0.05) -3.6%
Table 17: As Table 16, across decoding temperatures atk= 5.
F Effect of retrieval score dispersion on strong positional bias
Although the random permutations prevent retrieval relevance from favoring any prompt position (subsubsec-
tion 4.4.1), the relevance structure of the candidates may still modulate the magnitude of positional bias. We measure
it through the retrieval score dispersionδ(X), the standard deviation of the cosine similarities between the query and
its top-kcandidates, standardized within each setting. Low values correspond to candidates that the retriever barely
distinguishes.
For each model and setting, we fit a weighted linear regression of the instance-level strong positional bias, that is,
the percentage of valid evaluations ofXrecommending an item outsideR(X), onδ(X), weighting each instance
by its number of valid evaluations. Table 16 and Table 17 report the slope, in percentage points of sPB per standard
deviation ofδ(X), with95%confidence intervals based on heteroscedasticity-robust (HC1) standard errors, and the
same change relative to the sPB of the model in that setting. We test the null hypothesis that the slope is zero, i.e., that
the strong positional bias of a query does not depend linearly on its retrieval score dispersion, with a two-sided Wald
test based on the same HC1 standard errors. The resultingp-values are Holm-adjusted jointly across the32tests of
each table, that is, across the eight models and the four values ofk(Table 16) or oft(Table 17).
The slope is negative in every configuration: queries whose candidates are less separated by the retriever show slightly
more strong positional bias. The effect is small, at most about1pp per standard deviation, or a few percent of the
strong positional bias of each model, and it is significant mainly fork≥10, while atk= 5it is significant only for
a few models and shows no trend with temperature. Hence, retrieval score dispersion does not explain the positional
bias detected in this work, and only marginally modulates its strong component.
G Sensitivity to discarded generations
Generations that cannot be parsed, in which the model refuses to recommend an item, or that return an invalid position
are discarded before applying POB-UNBP (subsection 4.3). If these failures depended on the ordering of the candidate
items, discarding them could bias the estimated position distributions. To assess this, we repeat POB-UNBP restricted
to the complete-case queries, i.e., those for which none of the generations was discarded, so that every permutation of
each query contributes to the estimate. Theorem 1 holds for any set of queries, so restricting the analysis to a subset
of queries does not change the null hypothesis.
Table 18 reports, for each model and candidate-set size, the lower boundLof the total variation distance to the
uniform distribution and thep-value of the POB-UNBP test, computed on all queries (All) and on the complete-case
queries (CC). The complete-case subset retains between44%and98%of the queries, depending on the model and
32

Statistical Detection of Positional Bias in LLM-Based Recommender SystemsA PREPRINT
Model name MetricTop-5 Top-10 Top-15 Top-20
All CC All CC All CC All CC
DeepSeek R1 32BL 0.970% 0.563% 2.827% 2.291% 4.291% 4.114% 6.775% 6.618%
p-value 7.456e-7 0.002 5.130e-17 7.082e-17 2.202e-47 2.072e-38 2.098e-48 2.859e-40
DeepSeek R1 7BL 4.151% 5.279% 5.786% 7.430% 9.506% 9.432% 9.894% 9.969%
p-value 6.197e-60 2.169e-64 9.362e-55 1.299e-45 5.966e-79 2.045e-47 4.449e-70 2.075e-33
DeepSeek R1 8BL 0.788% 0.785% 1.959% 2.093% 4.303% 4.387% 7.238% 7.256%
p-value 5.875e-5 9.146e-7 1.150e-11 2.483e-12 2.253e-46 2.989e-46 2.111e-71 3.821e-70
LLaMA 3.1 8BL 3.009% 2.923% 3.131% 2.960% 4.078% 3.993% 3.768% 3.986%
p-value 4.457e-26 4.831e-24 4.928e-17 3.058e-15 3.379e-16 9.291e-15 1.465e-26 1.886e-15
LLaMA 4 ScoutL 0.872% 0.792% 8.298% 7.223% 21.430% 19.403% 30.189% 28.097%
p-value 1.188e-4 8.446e-5 1.438e-60 2.975e-48 8.023e-208 1.121e-125 1.341e-249 1.913e-137
GPT OSS 120BL 3.144% 3.188% 3.621% 3.614% 5.099% 5.088% 6.997% 7.053%
p-value 8.879e-30 3.145e-30 2.233e-41 3.083e-41 6.968e-48 2.612e-47 1.601e-58 9.038e-60
GPT-OSS 20BL 4.733% 4.670% 5.690% 5.603% 7.779% 7.187% 9.034% 8.474%
p-value 4.814e-65 5.262e-59 5.034e-82 5.072e-70 6.788e-96 3.230e-79 2.993e-100 1.016e-85
LLaMA 3.1 70BL 3.396% 3.328% 2.990% 3.030% 4.402% 4.417% 7.039% 7.096%
p-value 7.216e-24 3.409e-23 4.845e-25 9.634e-26 2.463e-29 1.214e-29 1.045e-41 3.221e-41
Table 18: Sensitivity of POB-UNBP to the discarded generations. For each model, the first row reports the lower
boundLof the total variation distance to the uniform distribution and the second row thep-value of the POB-UNBP
test, computed on all queries (All) and on the complete-case queries (CC), for which no generation was discarded.
candidate-set size, which is well above the sample sizes required for high power (Appendix B). The test rejects the null
hypothesis of absence of positional bias in all configurations on the complete-case queries, with the largestp-value
being0.002(DeepSeek R1 32B, top-5). The lower boundLalso remains stable: for most configurations it changes
by less than0.01, and the conclusions drawn in section 4 hold for both sets of queries. The largest relative decrease
corresponds to DeepSeek R1 32B at top-5 (0.010to0.006), which already exhibited the weakest bias, and the largest
absolute decrease to LLaMA 4 Scout at top-15 and top-20 (0.214to0.194and0.302to0.281), which remains the most
biased model by a wide margin. Conversely, for DeepSeek R1 7B, the model with the highest fraction of discarded
generations,Lincreases on the complete-case queries for top-5 and top-10 (0.042to0.053and0.058to0.074), so
discarding failures does not inflate the measured bias. Hence, the exclusion of failed generations does not explain our
findings.
Note that this analysis does not determine whether failures depend on the ordering of the candidate items; rather, it
shows that, whether or not they do, their exclusion does not alter the estimated positional bias.
33