# Brick-DICL: Dynamic In-Context Learning for Automated Brick Schema Classification

**Authors**: Yiyue Qian, Shinan Zhang, Huan Song, Negin Sokhandan, Hannah Marlowe, Diego Socolinsky

**Published**: 2026-06-16 07:46:51

**PDF URL**: [https://arxiv.org/pdf/2606.17637v1](https://arxiv.org/pdf/2606.17637v1)

## Abstract
Building Management Systems (BMS) are essential for optimizing energy efficiency and operational performance in modern buildings. However, the lack of standardization across BMS points from different manufacturers creates significant barriers to integration and data utilization. While the Brick schema offers a standardized ontology for building systems, mapping BMS points to appropriate Brick classes presents three critical challenges: (i) the extensive number of Brick classes (936 in the latest version), (ii) limited domain-specific knowledge in large language models (LLMs), and (iii) substantial manual effort required for verification. To address these challenges, we propose Brick-DICL, a two-stage dynamic in-context learning framework for automated Brick schema classification. Brick-DICL consists of two primary components: metadata-RAG, which retrieves relevant examples to enhance LLMs' domain knowledge, and class-RAG, which narrows down potential Brick classes to address the large classification space. Additionally, we implement a multi-LLM filtering mechanism that compares predictions across multiple models, flagging low-confidence classifications for human review. As a result: (i) General: Brick-DICL is applicable to any building management system regardless of manufacturer or metadata format; (ii) Novel and Powerful: as the first dynamic in-context learning approach for Brick schema classification, Brick-DICL achieves significant classification accuracy improvements on building datasets, outperforming existing methods; (iii) Efficient: our multi-LLM filtering strategy reduces manual verification effort, enabling rapid digital building onboarding. Extensive experiments demonstrate Brick-DICL's effectiveness across diverse building datasets, accelerating the path toward standardized, interoperable building management systems.

## Full Text


<!-- PDF content starts -->

Brick-DICL: Dynamic In-Context Learning for Automated Brick
Schema Classification
Yiyue Qian1, Shinan Zhang1, Huan Song1, Negin Sokhandan1, Hannah Marlowe1, Diego Socolinsky1
{iamyiyue, shinanz, huanso, ngnsl, marloweh, sclinsky}@amazon.com
1Amazon AWS Generative AI Innovation Center
ABSTRACT
Building Management Systems (BMS) are essential for optimizing
energy efficiency and operational performance in modern buildings.
However, the lack of standardization across BMS points from dif-
ferent manufacturers creates significant barriers to integration and
data utilization. While the Brick schema offers a standardized ontol-
ogy for building systems, mapping BMS points to appropriate Brick
classes presents three critical challenges: (i) the extensive number of
Brick classes (936 in the latest version), (ii) limited domain-specific
knowledge in large language models (LLMs), and (iii) substantial
manual effort required for verification. To address these challenges,
we propose Brick-DICL , a two-stage dynamic in-context learning
framework for automated Brick schema classification. Brick-DICL
consists of two primary components: metadata-RAG, which re-
trieves relevant examples to enhance LLMs’ domain knowledge,
and class-RAG, which narrows down potential Brick classes to ad-
dress the large classification space. Additionally, we implement a
multi-LLM filtering mechanism that compares predictions across
multiple models, flagging low-confidence classifications for hu-
man review. As a result: (i) General : Brick-DICL is applicable to
any building management system regardless of manufacturer or
metadata format; (ii) Novel and Powerful : as the first dynamic in-
context learning approach for Brick schema classification, Brick-
DICL achieves significant classification accuracy improvements on
building datasets, outperforming existing methods; (iii) Efficient :
our multi-LLM filtering strategy reduces manual verification effort,
enabling rapid digital building onboarding. Extensive experiments
demonstrate Brick-DICL’s effectiveness across diverse building
datasets, accelerating the path toward standardized, interoperable
building management systems.
KEYWORDS
Building Management System, In-context Learning, Large Lan-
guage Models, Brick Schema Classification, Multi-class Classifica-
tion
ACM Reference Format:
Yiyue Qian1, Shinan Zhang1, Huan Song1, Negin Sokhandan1, Hannah
Marlowe1, Diego Socolinsky1. 2026. Brick-DICL: Dynamic In-Context Learn-
ing for Automated Brick Schema Classification. In Proceedings of the Inter-
national Workshop on Resource-Efficient Learning for Knowledge Discovery
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than ACM
must be honored. Abstracting with credit is permitted. To copy otherwise, or republish,
to post on servers or to redistribute to lists, requires prior specific permission and/or a
fee. Request permissions from permissions@acm.org.
KDD RelKD ’26, Aug 10, 2026, Jeju, Korea
©2026 Association for Computing Machinery.
ACM ISBN 978-x-xxxx-xxxx-x/YY/MM.
https://doi.org/10.1145/nnnnnnn.nnnnnnn(KDD RelKD ’26), Aug 10, 2026, Jeju, Korea. ACM, New York, NY, USA, 9 pages.
https://doi.org/10.1145/nnnnnnn.nnnnnnn
1 INTRODUCTION
Building Management Systems (BMS) are vital for optimizing en-
ergy efficiency, operational performance, and sustainability in mod-
ern buildings [ 9,16,17,47]. However, the lack of standardization
across metadata from different manufacturers creates significant
barriers to seamless integration and onboarding of buildings onto
digital platforms. This fragmentation complicates the collection,
consolidation, and extraction of reliable data for insights and analyt-
ics, hindering efforts to achieve efficient building management [ 2].
To address these challenges, the Brick schema [ 2], a standardized
open-source ontology, has emerged as a promising solution. It
provides a structured framework for representing relationships
between building assets, systems, and devices, as illustrated in
Figure 1, enabling faster delivery of business outcomes through
streamlined data access. The Brick schema’s hierarchical graph
structure, where classes are organized in parent-child relationships
forming a directed acyclic graph, naturally lends itself to graph-
based representation techniques for capturing class similarities [ 46].
For example, Figure 1 shows how an Air Handling Unit (AHU1A) is
connected to Variable Air Volume Boxes (VAV2-4, VAV2-3), which
in turn are linked to rooms and associated sensors and setpoints,
illustrating the hierarchical and relational structure of the Brick
schema.
Despite its advantages, mapping BMS points to Brick schema
classes remains a daunting task due to two primary challenges: the
vast number of Brick classes (936 classes in the least version) and
the limited domain-specific knowledge of large language models
(LLMs). The sheer scale of point classification demands sophisti-
cated techniques to ensure accurate mapping across diverse build-
ing systems [ 45,49]. Meanwhile, LLMs, despite their impressive
general capabilities, often struggle with domain-specific tasks like
BMS point classification due to insufficient exposure to special-
ized datasets and contextual knowledge [ 13,14,35,51,55]. More-
over, manually verifying the point mapping classification results
generated by LLMs is labor-intensive and time-consuming, par-
ticularly when dealing with thousands of points across multiple
buildings [18, 20, 36, 48].
To handle the above challenges, this paper introduces Brick-
DICL , forBrick schema classification, a novel two-stage Dynamic
In-Context Learning (DICL) approach that leverages retrieval aug-
mented generation (RAG) techniques. Our approach addresses each
challenge systematically: First, to manage the vast number of Brick
classes, we implement a class-specific RAG that retrieves only the
most relevant classes, dramatically reducing the classification space.arXiv:2606.17637v1  [cs.AI]  16 Jun 2026

KDD RelKD ’26, Aug 10, 2026, Jeju, Korea Qian et al.
Figure 1: An example of Brick schema hierarchy, illustrating relationships among equipment, locations, and point classes in
BMS.
Second, to overcome limited domain knowledge in LLMs, we de-
velop a metadata RAG that retrieves similar, correctly-classified ex-
amples from a curated database, providing the model with context-
specific knowledge for each input point. For each classification task,
we first retrieve relevant examples from the metadata RAG to guide
the LLM’s initial prediction, then use this prediction to retrieve
related Brick classes, significantly narrowing the potential classi-
fication options. Finally, to reduce manual verification efforts, we
implement a multi-LLM framework with a sophisticated filtering
mechanism that identifies discrepancies between model predic-
tions, allowing human experts to focus exclusively on ambiguous
or low-confidence classifications. Our framework was evaluated
on two building datasets, demonstrating significant improvements
in both classification accuracy and efficiency over baseline models.
To conclude, our approach incorporates three key innovations:
•General : We introduce Brick-DICL, a general dynamic in-context
learning framework, encompassing metadata RAG and class RAG,
to map points to Brick classes. This general framework is appli-
cable to points in any BMS.
•Novel and Powerful : To the best of our knowledge, Brick-DICL
is the first approach that designs dynamic in-context learning on
LLMs to handle the large number of Brick classes and lacking
of domain knowledge issues. Brick-DICL achieves significant
improvements in mapping points in BMS to Brick classes over
two building datasets compared to existing methods.
•Efficient : Brick-DICL is designed to minimize human effort in
verification by using multi-LLMs to validate classifications and
automatically flagging low-confidence cases for human review.
2 RELATED WORK
Brick Point Classification. BMS points serve as the input/output
elements for buildings, allowing software to read sensor values
and write control commands [ 5,10,11,15]. However, BMS point
labels are largely ad-hoc constructions driven by vendor- or site-
specific conventions, creating challenges for standardization andinteroperability. Balaji et al. [ 3] introduced the Brick schema as a
comprehensive solution for representing building components and
their relationships, providing a foundation for standardized point
classification while acknowledging the complexity introduced by
the extensive class hierarchy. The Brick schema [ 3,5,11,15] ad-
dresses these challenges by providing a standardized ontology for
building metadata. It transforms semi-structured information im-
plied in point labels into explicit, standardized models accessible to
software. As noted by Balaji et al., the latest Brick schema contains
936 distinct classes, creating a large-scale multi-class classification
problem that is difficult to address with traditional machine learn-
ing approaches. The large volume of classes in the Brick schema
creates a fundamental challenge for classification systems, par-
ticularly when dealing with limited training data and the need
for high accuracy across all classes. Given that the Brick schema
is inherently a hierarchical graph with imbalanced class distribu-
tions, some works have explored graph-based approaches to model
building ontology structures and inter-class relationships [ 21,24–
27,30–34,37–40,46,52,53,57–60]. Inspired by the graph-based
hierachy structure of the Brick schema, in this paper, we propose
dynamic ICL, as it can adaptively retrieve relevant examples and
narrow down the potential class space, making the classification
task more manageable while maintaining high accuracy. To this
end, we propose dynamic ICL, as it can adaptively retrieve rele-
vant examples and narrow down the potential class space, making
the classification task more manageable while maintaining high
accuracy.
In-context Learning in LLMs. In-context learning (ICL), also
known as few-shot learning in LLMs, has emerged as a transforma-
tive capability in LLMs, enabling them to adapt to new tasks without
parameter updates [ 1,6,7,7,29,44,50,61]. Unlike traditional fine-
tuning methods that modify internal weights, ICL allows models to
learn from examples embedded directly within prompts, offering
significant advantages in flexibility, computational efficiency, and
effectiveness with limited examples. Specifically, Brown et al. [ 6]

Brick-DICL: Dynamic In-Context Learning for Automated Brick Schema Classification KDD RelKD ’26, Aug 10, 2026, Jeju, Korea
first demonstrated this capability in their seminal GPT-3 paper,
showing that LLMs can perform new tasks by conditioning on a
few examples within the prompt, without any gradient updates.
This few-shot learning paradigm represented a fundamental shift
in how models could be adapted to downstream tasks. Despite its
advantages, ICL faces several challenges: it suffers from scalability
issues when dealing with complex tasks, example sensitivity where
results depend on the quality of examples provided, and incon-
sistencies in responses. Peng et al. [ 29] explored the limitations
of ICL in specification-heavy tasks, finding that ICL performance
falls short in tasks requiring extensive specifications. Recent works
have explored LLM-empowered approaches for domain-specific
classification [ 23], multi-agent systems for data collection [ 28], and
inference-time model steering techniques [ 22,54]. These limita-
tions highlight the need for more sophisticated approaches to ICL.
To this end, we propose dynamic ICL in this paper to adaptively
select relevant examples.
3 PRELIMINARY
Definition 3.1. BMS Points and Brick Schema. A Building Man-
agement System (BMS) point 𝑝is a fundamental data entity in
building automation systems, representing a specific operational
data point. Each BMS point is described by a set of attributes
𝐴𝑝={𝑎1,𝑎2,...,𝑎𝑛}, such as equipment information, point in-
formation, and other metadata. However, these attributes are often
provided in non-standardized formats that vary across vendors and
systems.
To address this heterogeneity, the Brick schema provides a stan-
dardized ontology for representing building components, systems,
and their relationships. It defines a set of classes C={𝑐1,𝑐2,...,𝑐𝐾},
where𝐾=|C|is the total number of classes (936 in the latest ver-
sion). These classes are organized in a hierarchical structure, with
each class𝑐𝑖potentially having one or more parent classes, form-
ing a directed acyclic graph. By mapping BMS point attributes to
standardized Brick classes, the schema enables consistent represen-
tation of building metadata.
Definition 3.2. In-Context Learning (ICL). ICL is a technique
where a language model learns to perform a task by conditioning on
examples provided within the prompt, without updating its parame-
ters. Given a set of input-output examples {(𝑥1,𝑦1),(𝑥2,𝑦2),...,(𝑥𝑚,𝑦𝑚)}
and a new input 𝑥𝑛𝑒𝑤, the model predicts the output 𝑦𝑛𝑒𝑤by in-
ferring patterns from the provided examples. The performance of
ICL depends on the quality and relevance of the examples.
Problem 1. Brick Point Classification . Given a set of BMS
points𝑃={𝑝1,𝑝2,...,𝑝𝑁}, where each point 𝑝𝑖is described by a
set of metadata attributes 𝐴𝑝𝑖, and a set of Brick schema classes C=
{𝑐1,𝑐2,...,𝑐𝐾}, the goal is to develop a classification system that maps
each BMS point 𝑝𝑖to its corresponding Brick class 𝑐𝑗∈C. Formally,
the task is to learn a mapping function 𝑓:𝑃→C,where𝑓(𝑝𝑖)=
arg max
𝑐𝑗∈C𝑃(𝑐𝑗|𝐴𝑝𝑖)such that for each BMS point 𝑝𝑖, the most
appropriate Brick class 𝑐𝑗is assigned based on its metadata attributes.
4 PROPOSED MODEL
In this section, we present the details of Brick-DICL. The framework
consists of two main components: two-stage dynamic ICL andmulti-LLM mechanism for filtering low-confidence generations, as
illustrated in Figure 2.
4.1 Two-Stage Dynamic In-Context Learning
First of all, we preprocess various metadata attributes associated
with BMS points. After preprocessing BMS metadata, we feed meta-
data to LLMs to map points in BMS into Brick classes. As discussed,
traditional ICL enables LLMs to perform tasks by conditioning
on examples provided within the prompt without parameter up-
dates. However, standard ICL faces limitations when applied to
domain-specific tasks with large label spaces, such as Brick point
classification with 936 distinct classes. The fixed set of examples
used in traditional ICL may not provide sufficient context for ac-
curate classification across the diverse range of BMS points. To
this end, we introduce dynamic in-context learning (DICL) to pro-
vide relevant examples dynamically. Specifically, given a set of
pointsD={(𝑥1,𝑦1),(𝑥2,𝑦2),...,(𝑥𝑚,𝑦𝑚)}, a retrieval function
𝑅(𝑥new,D,𝑘)that selects 𝑘examples most similar to a new input
𝑥new, and a LLM model M, DICL is formulated as:
ˆ𝑦new=M(Instruction⊕{(𝑥𝑖,𝑦𝑖)|(𝑥𝑖,𝑦𝑖)∈𝑅(𝑥new,D,𝑘)}⊕𝑥new),
(1)
where⊕denotes text concatenation. Here ˆ𝑦newis the classifi-
cation that language model generated based on the dynamically
constructed prompt.
4.1.1 Stage One: Metadata Dynamic ICL. The first stage of our
approach focuses on producing point Brick classes using metadata-
based RAG (Metadata RAG). This stage lays the foundation for the
more fine-grained classification in stage two.
Metadata Retrieval. We first create a vector database of meta-
data for our training examples, enabling efficient similarity-based
retrieval. Specifically, for each BMS point 𝑝with attributes 𝐴𝑝, we
compute an embedding vector 𝑒𝑝using an embedding model, where
𝑒𝑝=M(𝐴𝑝). With embedding vector 𝑒𝑝, given a new BMS point
𝑝newwith attributes 𝐴𝑝new, we retrieve the 𝑘most similar examples
using cosine similarity between embedding vectors:
𝑅(𝑝new,D,𝑘)=argmax𝑘
𝑝𝑖∈D𝑒𝑝new·𝑒𝑝𝑖
||𝑒𝑝new||·||𝑒𝑝𝑖||, (2)
where argmax𝑘returns the𝑘examples with the highest similar-
ity scores andDis the point set.
Prediction Task. With the retrieved example 𝑟∈𝑅(𝑝new,D,𝑘),
the prompt𝑃stage1 at the stage one in metadata DICL can be formu-
lated as :
𝑃stage1 =Instruction⊕ 𝑘Ê
𝑖=1𝜙(𝑟𝑖)!
⊕𝜓(𝐴𝑝), (3)
where𝜙(𝑟𝑖)formats the retrieved example 𝑟𝑖, and𝜓(𝐴𝑝)formats
the input attributes of the new BMS point. The operator ⊕com-
bines all components into a single input prompt for the language
model. The retrieved examples 𝑟, obtained from metadata RAG
in Equation 3, provide domain-specific context by demonstrating
how similar attribute combinations map to specific Brick classes.
These examples are dynamically selected based on cosine similar-
ity between embedding vectors, ensuring relevance to the input

KDD RelKD ’26, Aug 10, 2026, Jeju, Korea Qian et al.
Figure 2: The framework of Brick-DICL: (a) Raw data is first preprocessed to extract and standardize meta-data and candidate
brick classes. (b) In Stage One (steps 1–5), meta-data is embedded and indexed using an embedding model and a vector database,
enabling retrieval of the top-K most relevant examples for each query. (c) In Stage Two (steps 6–9), a class-level retrieval-
augmented generation (RAG) process embeds and indexes candidate brick classes, retrieves the top-K classes, and combines
them with the meta-data context. (d) Multiple LLMs then perform the top three classified classes with their probability scores
and explanations.
attributes𝐴𝑝. Given the constructed prompt 𝑃stage1 , the language
modelMgenerates predictions as (𝑐(1)
brick,𝑗,𝑠)=M(𝑃), where
𝑐(1)
brickis the predicted point class at stage one, 𝑗is the justification
for the predictions, and 𝑠is the confidence score. The justification 𝑗
provides interpretability by explaining how the model arrived at its
predictions, while the confidence score 𝑠quantifies the model’s cer-
tainty. This formulation ensures that predictions are contextually
grounded in relevant examples while maintaining consistency with
Brick schema constraints. The structured outputs (𝑐brick,𝑗,𝑠)are
then passed to stage two for further refinement via class-specific
retrieval-augmented generation.
4.2 Stage Two: Class Dynamic ICL
The second stage refines the initial predictions through class-specific
RAG (Class RAG). This stage addresses the challenge posed by the
extensive Brick schema (936 classes) by dynamically focusing ona semantically relevant subset of classes, guided by stage one’s
predictions.
4.2.1 Class-Specific Retrieval. Given the predicted class 𝑐(1)
brickfrom
stage one, we retrieve a focused subset of Brick classes Cret⊂C
using the class similarity. For each class 𝑐𝑗∈C, the retrieval score
for class𝑐𝑗is calculated as sim(𝑐𝑗,𝑐(1)
brick)=𝑒𝑐𝑗·𝑒𝑐(1)
brick
∥𝑒𝑐𝑗∥∥𝑒𝑐(1)
brick∥, where
𝑒𝑐𝑗is the class embedding using the same embedding model. Then
the top-𝑚classes are selected as Cret=argmax
𝑐𝑗∈C𝑚sim(𝑐𝑗,𝑐(1)
brick),
where𝑚(i.e.,𝑚=20) is the hyper-parameter to control the number
of retrieved classes, which can ensure reduction in classification
complexity while maintaining high performance.
4.2.2 Two-Step LLM Generation. Once the top 𝑚(i.e., 20) classes
are retrieved, we further refine them through a two-step evaluation
process designed to progressively reduce difficulty.

Brick-DICL: Dynamic In-Context Learning for Automated Brick Schema Classification KDD RelKD ’26, Aug 10, 2026, Jeju, Korea
Specifically, at the first step, the LLM evaluates all 20 retrieved
classes and selects 5 preliminary candidates based on their align-
ment with input attributes and the retrieved examples. This step
reduces complexity by narrowing down to a smaller set of plausible
candidates while retaining flexibility to capture ambiguous or less
obvious matches.
At the second step, the LLM performs a more focused evalu-
ation of these 5 candidates to select the top 3 most likely Brick
classes. This step further simplifies decision-making by concen-
trating on a manageable number of options, allowing for detailed
reasoning about fine-grained distinctions. For example, among Dis-
charge_Air_Temperature_Sensor, Supply_Air_Temperature_Sensor,
and Return_Air_Temperature_Sensor, the LLM can prioritize Dis-
charge_Air_Temperature_Sensor based on its alignment with equip-
ment type (e.g., RTU discharge air system).
The progressive reduction ensures that each phase is easier for
the LLM to handle, as it focuses on increasingly smaller subsets of
candidates. This design reduces cognitive load for the model, lead-
ing to more accurate predictions. The stage two can be formulated
by integrating retrieved classes, retrieved examples from stage one,
and sample input information:
({(𝑐1,𝑝1),(𝑐2,𝑝2),(𝑐3,𝑝3)},𝑗,𝑠)=M Instruction⊕𝜙(Cret)
⊕ 𝑘Ê
𝑖=1𝜙(𝑟𝑖)!
⊕𝜓(𝐴𝑝).(4)
The output includes probabilities for each of the top three classes.
Probabilities 𝑝𝑖are normalized such thatÍ𝑝𝑖=1. The justification
𝑗explains how input attributes align with each predicted class and 𝑠
is the confidence score for the classification generated by the LLM.
4.3 Multi-LLM Mechanism for Low-Confidence
Generation Filter
To further enhance the reliability of the proposed approach and min-
imize human intervention, we incorporate a multi-LLM mechanism
to filter low-confidence predictions. This mechanism uses multiple
LLMs to generate classifications across both stages and applies a fil-
tering strategy to identify and flag low-confidence generations. By
comparing predictions from multiple LLMs, this approach ensures
robustness and reduces errors caused by individual model biases or
uncertainties. Specifically, for a BMS point 𝑝, let𝑐(1)
brick,𝑀𝑖represent
the stage one prediction from model 𝑀𝑖, and{𝑐1,𝑀𝑖,𝑐2,𝑀𝑖,𝑐3,𝑀𝑖}
represent the top three predictions from stage two for model 𝑀𝑖.
Four distinct filtering strategies are employed to assess confidence
levels:
•All Agreement : This strategy flags a generation as low-confidence
if any of the predicted results from stage one or classified results
from stage two differ across the LLMs:
𝑐(1)
brick,𝑀𝑖≠𝑐(1)
brick,𝑀𝑗∨{𝑐1,𝑀𝑖,𝑐2,𝑀𝑖,𝑐3,𝑀𝑖}≠{𝑐1,𝑀𝑗,𝑐2,𝑀𝑗,𝑐3,𝑀𝑗}.
(5)
If all predictions across both stages are identical, the generation
is considered high-confidence.
•Top-3 Consensus : This strategy flags a generation as low-confidence
if there is no complete overlap among the top three predictedclasses across all LLMs:Ù
𝑀𝑖∈M{𝑐1,𝑀𝑖,𝑐2,𝑀𝑖,𝑐3,𝑀𝑖}=∅. (6)
•Top-1 Agreement : This strategy flags a generation as low-confidence
if the top one predicted class differs across any of the LLMs:
∃𝑀𝑖,𝑀𝑗∈M|𝑐1,𝑀𝑖≠𝑐1,𝑀𝑗. (7)
•Any-2 Consensus : This strategy flags a generation as low-confidence
if fewer than two common predicted classes exist among the top
three predictions from multiple LLMs:
max
𝑐∈C∑︁
𝑀𝑖∈MI(𝑐∈{𝑐1,𝑀𝑖,𝑐2,𝑀𝑖,𝑐3,𝑀𝑖})<2. (8)
These strategies provide varying levels of strictness in assessing
confidence. The All Agreemen strategy is the most stringent, re-
quiring complete consistency across all models and stages, while
Any-2 Consensus is more tolerant, allowing for some disagreement
among models.
Our filtering mechanism evaluates these predictions using one
of the four strategies described above. Flagged generations are
routed to human reviewers through an interface that aggregates
all model outputs and highlights discrepancies for validation. The
multi-LLM mechanism significantly enhances classification relia-
bility by leveraging consensus among models to identify uncertain
or inconsistent predictions. By offering multiple filtering strategies
with varying levels of strictness, it allows flexibility based on ap-
plication requirements. For clarity, the pseudo-code is outlined in
Alogrithm 1.
This section shows the pseudo-code of Brick-DICL, which in-
cludes two main modules: two stage DICL, and multi-LLM for
low-confidence generation filter.
5 EXPERIMENT
In this section, we first introduce two building datasets B1 and B2.
Then we conduct extensive experiments to evaluate Brick-DICL on
Brick point classification tasks.
5.1 Experiment Setup
Datasets. We evaluate on two BMS datasets from two buildings
(B1 and B2). Both datasets contain BMS metadata and ground-truth
Brick class labels. The latest Brick schema has 936 point classes and
3,897 parent-class relation pairs. In our approach, each building
data is split into training set and testing set. At stage one, we utilize
training set to build the metadata knowledge base. At stage two,
we utilize all Brick classes to build the class knowledge base.
Baseline. To comprehensively evaluate our Brick-DICL frame-
work, we compare Brick-DICL with eight baseline models in two
groups: A1: prompting-based methods. We tried four sets of
prompting-based methods including standard prompting without
few-shot ICL [ 42], static ICL [ 12], random ICL [ 41], and dynami-
cal ICL [ 56]. For static/random/dynamic ICL, we select few-shot
input samples from the training set in a static/random/dynamic
manner and further ask LLMs to map input samples to the Brick
classes. A2: pre-trained language models. We fine-tune four pre-
trained language models over the building data, including BERT [ 8],
DistilBert [43], Roberta [19], and SciBert [4].

KDD RelKD ’26, Aug 10, 2026, Jeju, Korea Qian et al.
PromptStatic ICLRandom ICL Dynamic ICLBERT
DistilBertRoberta SciBert
Brick-DICL020406080100Accuracy (%)B1 Building
Stage2 Hits@1
Stage2 Hits@3
PromptStatic ICLRandom ICL Dynamic ICLBERT
DistilBertRoberta SciBert
Brick-DICL020406080100Accuracy (%)B2 Building
Stage2 Hits@1
Stage2 Hits@3
Figure 3: Stage2 Hits@1 and Hits@3 accuracy comparison for BMS point classification on B1 and B2 buildings.
1 7 13156080100
# of ShotsAccuracy (%)B1 (Metadata Shots)
1 7 13156080100
# of ShotsB2 (Metadata Shots)
5 15 20 309095100
# of ClassesAccuracy (%)B1 (Retrieved Classes)
5 15 20 309095100
# of ClassesB2 (Retrieved Classes)
Stage1 Hits@1 Stage2 Hits@1 Stage2 Hits@3
Figure 4: Impact of hyperparameters on Brick-DICL. Left two: number of shots in metadata ICL. Right two: number of retrieved
classes in class ICL.
Experimental Setting. To evaluate Brick-DICL, we adopt the
widely-used metrics at each stage: Stage1 hits@1, Stage2 hits@1,
and Stage2 hits@3. Hits@1 refers to the accuracy of the model’s
top generation result being correct, while hits@3 measures the
accuracy of the correct classification being among the model’s top
three generation results. Stage1 hits@1 is the prediction accuracy
at stage one, while Stage2 hits@1 and Stage2 hits@3 refer to the
classification accuracy at stage two. Mention that, if the predicted
class is the parent of the ground-truth class, following the existing
work [ 2], we consider it as a correct classification. Besides, we used
multiple LLMs to filter low-confidence generations.
5.2 Performance Discussion on Brick-DICL
5.2.1 Performance Comparison among Baselines. Figure 3 summa-
rizes the Hits@1 and Hits@3 accuracy for the BMS point classifica-
tion task on two buildings, B1 and B2. The results clearly highlight
several important trends. Firstly, ICL-based approaches (A1) ex-
hibit a substantial advantage over conventional transformer models
(A2). In particular, dynamic ICL achieves notably higher Hits@3
accuracy on both buildings compared to BERT-based models. Thisdemonstrates the strong capability of ICL strategies to leverage
context for enhanced semantic understanding.
Secondly, a closer examination within group A1 reveals that
dynamic ICL delivers markedly better performance than its static
counterpart. This underscores the crucial role of dynamic context
adaptation in extracting more relevant, task-specific representa-
tions from the data, as opposed to utilizing a fixed context.
Thirdly, among standard transformer models in group A2, RoBERTa
achieves the highest accuracy on B2, while DistilBERT shows the
lowest performance, suggesting that model architecture and pre-
training methodology significantly influence outcomes in this clas-
sification scenario.
Finally, Brick-DICL consistently outperforms all baseline meth-
ods across both buildings and both metrics, validating the effective-
ness and robustness of our proposed dynamic ICL framework for
Brick schema-based point classification.
5.2.2 Impact of Number of Shots in Metadata ICL. Figure 4 (left
two panels) shows the accuracy of Brick-DICL with different num-
bers of shots in metadata ICL on both buildings. From this figure,
we can conclude that: (i) Performance generally improves as the

Brick-DICL: Dynamic In-Context Learning for Automated Brick Schema Classification KDD RelKD ’26, Aug 10, 2026, Jeju, Korea
Algorithm 1: Brick-DICL Classification Pipeline
Require BMS point dataD, Brick classesC, LLMsM, Training
examplesT
Output Classified BMS points with Brick classes {(𝑝𝑖,𝑐final)}
1:Data Standardization:
2:foreach𝑝∈Ddo
3: Normalize metadata attributes
4: Generate missing descriptions using Mdesc
5:end for
6:Build RAG Databases:
7:Meta-data RAG: Create embedding vectors for training
examples
8:Class RAG: Create embedding vectors for Brick class
definitions
Classification Process :
9:foreach BMS point 𝑝𝑖∈Ddo
10: Stage One - Metadata ICL:
11: Retrieve top- 𝑘similar examples from Meta-data RAG
12: Construct dynamic prompt with examples and input
attributes
13: Generate initial Brick class prediction 𝑐(1)
brick
14: Stage Two - Class ICL:
15: Retrieve top- 𝑚relevant classes from Class RAG
16: Construct refinement prompt with class definitions
17: Generate top-3 predictions {(𝑐1,𝑝1),(𝑐2,𝑝2),(𝑐3,𝑝3)}
18: Multi-LLM Filtering:
19: foreach LLM𝑀𝑗∈M do
20: Get predictions(𝑐(1)
brick,𝑗,{𝑐1,𝑗,𝑐2,𝑗,𝑐3,𝑗})
21: end for
22: Apply filtering strategies (All/Top3/Top1/Any2)
23: iflow-confidence flagged then
24: Route to human validation interface
25: end if
26:end for
Human Validation :
27:foreach flagged prediction do
28: Present multi-LLM predictions and justifications
29: Store expert-validated results
30:end for
31:Return Final classifications with confidence scores
number of shots increases, with 15-shot ICL achieving the best
overall results across most metrics. (ii) The improvement is more
pronounced in B2 compared to B1, indicating that more examples
provide greater benefit when the metadata is more diverse. (iii)
Mid-range configurations (7-shot and 10-shot) offer competitive
Stage2 Hits@3 performance while requiring fewer examples. (iv)
B2 consistently outperforms B1 across all shot configurations, indi-
cating building-specific characteristics may influence classification
performance. To conclude, we select 𝑘=15for stage one.
5.2.3 Impact of Number of Retrieved Classes in Class ICL. Figure 4
(right two panels) shows the accuracy of Brick-DICL with different
numbers of retrieved classes in class ICL. From this figure, we canconclude that: (i) Class retrieval affects only Stage2 performance
as expected, since Stage1 prediction occurs before class retrieval.
(ii) For B1, Top 20 retrieved classes achieves the best performance,
while for B2, both Top 15 and Top 20 yield optimal results. (iii)
Performance improvement plateaus or slightly declines beyond 20
retrieved classes, suggesting that including too many candidate
classes may introduce noise in the classification process. (iv) The
improvement from Top 5 to Top 20 is more significant in Stage2
Hits@3 than in Stage2 Hits@1, indicating that retrieving more
classes primarily helps capture the correct class within the top
three predictions. In this work, we select 𝑚=20in consideration
of the performance and efficiency.
5.2.4 Comparison of Filtering Strategies in Multi-LLM Consensus.
Figure 5 displays a grouped bar and line chart comparing four fil-
tering strategies—All, Top-3, Top-1, and Any-2—across two large
language models (Model 1 and Model 2) on the B1 building dataset.
The bars represent Stage2 Hits@3 accuracy for Model 1 and Model
2, respectively; the teal line shows the disagreement ratio for each
strategy. Model 1 maintains stable accuracy across all filtering mech-
anisms, indicating robust and consistent performance, whereas
Model 2 exhibits greater fluctuation, reflecting less stability. The dis-
agreement ratio markedly drops from the All and Top-3 strategies
to Top-1 and Any-2, demonstrating that Top-1 and Any-2 strategies
substantially reduce the number of flagged low-confidence predic-
tions. The Any-2 strategy achieves the lowest disagreement ratio,
suggesting it is the most efficient for minimizing manual review
while preserving strong classification accuracy.
5.2.5 Ablation Study. Figure 6 presents the results of ablation ex-
periments on Stage2 Hits@3 accuracy for B1 and B2 buildings under
five different settings: removing Metadata DICL, Class DICL, Both
DICL, Two steps in Stage Two, and using the full Brick-DICL model.
The grouped bar chart shows that the removal of either metadata
or class DICL individually leads to a moderate decrease in perfor-
mance, while removing both simultaneously results in the most
significant drop in Hits@3 accuracy for both buildings. Excluding
the two-step process in Stage Two also reduces accuracy, though not
as drastically. The Brick-DICL setting achieves the highest Hits@3
All Top-3 Top-1 Any-2859095100
Filtering StrategyAccuracy (%)Model 1 Hits@3
Model 2 Hits@3
0204060
Disagreement Ratio (%)Disagreement Ratio
Figure 5: Comparison of four filtering strategies among
Model 1 and Model 2 on B1 building.

KDD RelKD ’26, Aug 10, 2026, Jeju, Korea Qian et al.
w/o Meta DICL w/o Class DICL w/o Both DICL w/o Two StepsBrick-DICL6080100Hits@3 (%)B1 Building
B2 Building
Figure 6: Performance comparison among model variants on
B1 and B2 buildings.
on both B1 and B2 buildings, underscoring the effectiveness of the
complete approach. These results demonstrate that each compo-
nent contributes to overall performance, with the combination of
all modules in Brick-DICL being essential for achieving optimal
classification accuracy.
6 CONCLUSION
In this paper, we propose Brick-DICL, a framework for automated
Brick point classification in building management systems. It con-
sists of two main modules: a two-stage dynamic in-context learning
pipeline and a multi-LLM filtering mechanism. The two-stage dy-
namic ICL learns to map BMS points to standardized Brick schema
classes by leveraging metadata RAG followed by class-specific
refinement. These learned mappings can be directly applied to
building commissioning, energy management, and interoperability
tasks. The multi-LLM filtering mechanism automatically identifies
low-confidence predictions requiring human review, significantly
reducing manual effort while maintaining high accuracy. Compre-
hensive experiments on multiple building datasets demonstrate
Brick-DICL’s superiority, achieving high classification accuracy
with minimal human intervention. Evaluation of filtering strate-
gies reveals the Any-2 consensus approach achieves the optimal
balance between accuracy and disagreement ratio. These results
highlight Brick-DICL’s capabilities in addressing the challenging
task of mapping diverse BMS points to standardized ontologies,
accelerating digital building creation.
REFERENCES
[1]Rishabh Agarwal, Avi Singh, Lei Zhang, Bernd Bohnet, Luis Rosias, Stephanie
Chan, Biao Zhang, Ankesh Anand, Zaheer Abbas, Azade Nova, et al. Many-shot
in-context learning. In NeurIPS , volume 37, pages 76930–76966, 2024.
[2]Bharathan Balaji, Arka Bhattacharya, Gabriel Fierro, Jingkun Gao, Joshua Gluck,
Dezhi Hong, Aslak Johansen, Jason Koh, Joern Ploennigs, Yuvraj Agarwal, et al.
Brick: Towards a unified metadata schema for buildings. In Proceedings of the 3rd
ACM International Conference on Systems for Energy-Efficient Built Environments ,
pages 41–50, 2016.
[3]Bharathan Balaji, Arka Bhattacharya, Gabriel Fierro, Jingkun Gao, Joshua Gluck,
Dezhi Hong, Aslak Johansen, Jason Koh, Joern Ploennigs, Yuvraj Agarwal, et al.
Brick: Metadata schema for portable smart building applications. Applied energy ,
226:1273–1292, 2018.
[4]Iz Beltagy, Kyle Lo, and Arman Cohan. Scibert: A pretrained language model for
scientific text. In EMNLP , pages 3615–3620, 2019.[5]Arka Bhattacharya, Joern Ploennigs, and David Culler. Short paper: Analyzing
metadata schemas for buildings: The good, the bad, and the ugly. In Proceedings
of the 2nd ACM International Conference on Embedded Systems for Energy-Efficient
Built Environments , pages 33–34, 2015.
[6]Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan,
Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, et al. Language models are few-shot learners. In NeurIPS , 2020.
[7]Julian Coda-Forno, Marcel Binz, Zeynep Akata, Matt Botvinick, Jane Wang, and
Eric Schulz. Meta-in-context learning in large language models. In NeurIPS , 2023.
[8]Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-
training of deep bidirectional transformers for language understanding. In
NAACL-HLT , pages 4171–4186, 2019.
[9]Ying Ding, Yi Liu, and Yong Wu. Metadata standardization in iot-based building
energy management systems. Journal of Cleaner Production , 210:1234–1243, 2019.
[10] Kevin Fiorelli, Kaveh Dehghanpour, and Alberto Sangiovanni-Vincentelli. Auto-
mated classification of building energy metadata using ontology-based natural
language processing. Applied Energy , 314:118915, 2023.
[11] Jingkun Gao, Joern Ploennigs, and Mario Berges. A data-driven meta-data
inference framework for building automation systems. In Proceedings of the 2nd
ACM International Conference on Embedded Systems for Energy-Efficient Built
Environments , pages 23–32, 2015.
[12] Tianyu Gao, Adam Fisch, and Danqi Chen. Making pre-trained language models
better few-shot learners. arXiv preprint arXiv:2012.15723 , 2020.
[13] Yingqiang Ge, Wenyue Hua, Kai Mei, Juntao Tan, Shuyuan Xu, Zelong Li,
Yongfeng Zhang, et al. Openagi: When llm meets domain experts. In NeurIPS ,
volume 36, pages 5539–5568, 2023.
[14] Paul Hager, Friederike Jungmann, Robbie Holland, Kunal Bhagat, Inga Hubrecht,
Manuel Knauer, Jakob Vielhauer, Marcus Makowski, Rickmer Braren, Georgios
Kaissis, et al. Evaluation and mitigation of the limitations of large language
models in clinical decision-making. Nature medicine , 30(9):2613–2622, 2024.
[15] Dezhi Hong, Jorge Ortiz, Kamin Whitehouse, and David Culler. Towards auto-
matic spatial verification of sensor placement in buildings. In Proceedings of the
5th ACM Workshop on Embedded Systems For Energy-Efficient Buildings , pages
1–8, 2013.
[16] Ibrahim Katib, Waleed Abdulla, and Mohammed Nasr. Integrating building
automation systems for energy efficiency: A review. Renewable and Sustainable
Energy Reviews , 59:1571–1581, 2016.
[17] Seungwon Lee, Hui Li, and Yong Weng. Data integration challenges in smart
building systems: A review. Energy and Buildings , 140:241–252, 2017.
[18] Henger Li, Shuangjie You, Flavio Di Palo, Yiyue Qian, and Ayush Jain. Verification-
guided context optimization for tool calling via hierarchical llms-as-editors. arXiv
preprint arXiv:2512.13860 , 2025.
[19] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer
Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly
optimized bert pretraining approach. arXiv preprint arXiv:1907.11692 , 2019.
[20] Lezhi Ma, Shangqing Liu, Yi Li, Xiaofei Xie, and Lei Bu. Specgen: Automated
generation of formal program specifications via large language models. arXiv
preprint arXiv:2401.08807 , 2024.
[21] Tianyi Ma, Yiyang Li, Yiyue Qian, Zheyuan Zhang, Zehong Wang, Chuxu Zhang,
and Yanfang Ye. Opbench: A graph benchmark to combat the opioid crisis. arXiv
preprint arXiv:2602.14602 , 2026.
[22] Tianyi Ma, Yiyue Qian, Yiyang Li, Zehong Wang, Yifan Ding, Zheyuan Zhang, Yan
Liang, Chuxu Zhang, and Yanfang Ye. Non-monotonic autoregressive sequence
model. In ICML , 2026.
[23] Tianyi Ma, Yiyue Qian, Zehong Wang, Zheyuan Zhang, Chuxu Zhang, and
Yanfang Ye. Llm-empowered class imbalanced graph prompt learning for online
drug trafficking detection. In Findings of ACL , 2025.
[24] Tianyi Ma, Yiyue Qian, Zehong Wang, Zheyuan Zhang, Chuxu Zhang, and
Yanfang Ye. Bhygnn+: Unsupervised representation learning for heterophilic
hypergraphs. arXiv preprint arXiv:2602.14919 , 2026.
[25] Tianyi Ma, Yiyue Qian, Zehong Wang, Zheyuan Zhang, Shinan Zhang, Chuxu
Zhang, and Fanny Ye. Hypergraph representation learning with adaptive broad-
casting and receiving. In ICDM , 2025.
[26] Tianyi Ma, Yiyue Qian, Chuxu Zhang, and Yanfang Ye. Hypergraph contrastive
learning for drug trafficking community detection. In ICDM , 2023.
[27] Tianyi Ma, Yiyue Qian, Shinan Zhang, Chuxu Zhang, and Yanfang Ye. Adaptive
expansion for hypergraph learning. arXiv preprint arXiv:2502.15564 , 2025.
[28] Tianyi Ma, Yiyue Qian, Zheyuan Zhang, Zehong Wang, Xiaoye Qian, Feifan Bai,
Yifan Ding, Xuwei Luo, Shinan Zhang, Keerthiram Murugesan, et al. Autodata:
A multi-agent system for open web data collection. In NeurIPS , 2025.
[29] Baolin Peng, Chunyuan Li, Pengcheng He, Michel Galley, and Jianfeng Gao.
Instruction tuning with gpt-4. arXiv preprint arXiv:2304.03277 , 2023.
[30] Yiyue Qian. Graph Representation Learning Techniques for the Combat Against
Online Abusive Activity . University of Notre Dame, 2024.
[31] Yiyue Qian, Philip Chen, Song Cui, and De Chen. Universal ring-of-abusers
detection via multi-modal heterogeneous graph learning. 2023.
[32] Yiyue Qian, Tianyi Ma, Chuxu Zhang, and Yanfang Ye. Dual-level hypergraph
contrastive learning with adaptive temperature enhancement. In Companion of

Brick-DICL: Dynamic In-Context Learning for Automated Brick Schema Classification KDD RelKD ’26, Aug 10, 2026, Jeju, Korea
WWW , pages 859–862, 2024.
[33] Yiyue Qian, Tianyi Ma, Chuxu Zhang, and Yanfang Ye. Adaptive graph enhance-
ment for imbalanced multi-relation graph learning. In WSDM , 2025.
[34] Yiyue Qian, Chunhui Zhang, Yiming Zhang, Qianlong Wen, Yanfang Ye, and
Chuxu Zhang. Co-modality graph contrastive learning for imbalanced node
classification. In NeurIPS , volume 35, pages 15862–15874, 2022.
[35] Yiyue Qian, Shinan Zhang, Lanhao Chen, Diego Socolinsky, Negin Sokhandan,
Song Cui, De Chen, and Suchitra Sathyanarayana. Enhancing e-commerce
representation learning via hypergraph contrastive learning and interpretable
llm-driven analysis. In Companion WWW , pages 2512–2520, 2025.
[36] Yiyue Qian, Shinan Zhang, Yun Zhou, Haibo Ding, Diego Socolinsky, and
Yi Zhang. Collabeval: Enhancing llm-as-a-judge via multi-agent collaboration.
arXiv preprint arXiv:2603.00993 , 2026.
[37] Yiyue Qian, Yiming Zhang, Nitesh Chawla, Yanfang Ye, and Chuxu Zhang. Mali-
cious repositories detection with adversarial heterogeneous graph contrastive
learning. In CIKM , 2022.
[38] Yiyue Qian, Yiming Zhang, Qianlong Wen, Yanfang Ye, and Chuxu Zhang.
Rep2vec: Repository embedding via heterogeneous graph adversarial contrastive
learning. In KDD , 2022.
[39] Yiyue Qian, Yiming Zhang, Yanfang Ye, and Chuxu Zhang. Adapting meta knowl-
edge with heterogeneous information network for covid-19 themed malicious
repository detection. In IJCAI , 2021.
[40] Yiyue Qian, Yiming Zhang, Yanfang Ye, and Chuxu Zhang. Distilling meta
knowledge on heterogeneous graph for illicit drug trafficker detection on social
media. In NeurIPS , 2021.
[41] Ohad Rubin, Jonathan Herzig, and Jonathan Berant. Learning to retrieve prompts
for in-context learning. In NAACL-HLT , pages 2655–2671, 2022.
[42] Pranab Sahoo, Ayush Kumar Singh, Sriparna Saha, Vinija Jain, Samrat Mondal,
and Aman Chadha. A systematic survey of prompt engineering in large language
models: Techniques and applications. arXiv preprint arXiv:2402.07927 , 2024.
[43] Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. Distilbert,
a distilled version of bert: smaller, faster, cheaper and lighter. arXiv preprint
arXiv:1910.01108 , 2019.
[44] Huan Song, Deeksha Razdan, Yiyue Qian, Arijit Ghosh Chowdhury, Parth
Patwa, Aman Chadha, Shinan Zhang, Sharlina Keshava, and Hannah Marlowe.
Learning from generalization patterns: An evaluation-driven approach to en-
hanced data augmentation for fine-tuning small language models. arXiv preprint
arXiv:2510.18143 , 2025.
[45] Stefan Steiniger, Tilman Lange, Dirk Burghardt, and Robert Weibel. An approach
for the classification of urban building structures based on discriminant analysis
techniques. Transactions in GIS , 12(1):31–59, 2008.
[46] Filippo Vittori, Chuan Fu Tan, Anna Laura Pisello, Adrian Chong, Cristina Piselli,
and Clayton Miller. Bim-to-brick: Using graph modeling for iot/bms and spatial
semantic data interoperability within digital data models of buildings. Energy
and Buildings , 348:116368, 2025.
[47] Roman Volk, Joachim Stengel, and Frank Schultmann. Building information
modeling (bim) for existing buildings — literature review and future needs. Au-
tomation in Construction , 38:109–127, 2014.
[48] Stefan Wagner, Marvin Muñoz Barón, Davide Falessi, and Sebastian Baltes. To-
wards evaluation guidelines for empirical studies involving llms. arXiv preprint
arXiv:2411.07668 , 2024.
[49] Weimin Wang, Michael R Brambley, Woohyun Kim, Sriram Somasundaram, and
Andrew J Stevens. Automated point mapping for building control systems: Recent
advances and future research needs. Automation in Construction , 85:107–123,
2018.
[50] Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian
Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al.
Emergent abilities of large language models. Transactions on Machine Learning
Research .
[51] Laslo Welz and Carsten Lanquillon. Enhancing large language models through
external domain knowledge. In HCII, pages 135–146. Springer, 2024.
[52] Qianlong Wen, Zhongyu Ouyang, Chunhui Zhang, Yiyue Qian, Chuxu Zhang,
and Yanfang Ye. Gcvr: reconstruction from cross-view enable sufficient and
robust graph contrastive learning. In UAI, 2024.
[53] Qianlong Wen, Zhongyu Ouyang, Jianfei Zhang, Yiyue Qian, Yanfang Ye, and
Chuxu Zhang. Disentangled dynamic heterogeneous graph learning for opioid
overdose prediction. In KDD , 2022.
[54] Renhao Xue, Rui Wang, Yawei Wang, Yueying Cui, Yiyue Qian, Praneetha Vad-
damanu, Huan Song, and Hannah Marlowe. Inference-time model steering via
predictive-state intervention: A survey. 2026.
[55] Zhiyi Xue, Liangguo Li, Senyue Tian, Xiaohong Chen, Pingping Li, Liangyu
Chen, Tingting Jiang, and Min Zhang. Domain knowledge is all you need: A
field deployment of llm-powered test case generation in fintech domain. In
ICSE-Companion , pages 314–315, 2024.
[56] Qinyuan Ye, Bill Yuchen Lin, and Xiang Ren. Crossfit: A few-shot learning
challenge for cross-task generalization in nlp. In EMNLP , pages 7163–7189, 2021.
[57] Yanfang Ye, Yujie Fan, Shifu Hou, Yiming Zhang, Yiyue Qian, Shiyu Sun, Qian
Peng, Mingxuan Ju, Wei Song, and Kenneth Loparo. Community mitigation: Adata-driven system for covid-19 risk assessment in a hierarchical manner. In
CIKM , pages 2909–2916, 2020.
[58] Yanfang Ye, Shifu Hou, Yujie Fan, Yiyue Qian, Yiming Zhang, Shiyu Sun, Qian
Peng, and Kenneth Laparo. 𝛼-satellite: An ai-driven system and benchmark
datasets for hierarchical community-level risk assessment to help combat covid-
19.arXiv preprint arXiv:2003.12232 , 2020.
[59] Yiming Zhang, Yiyue Qian, Yujie Fan, Yanfang Ye, Xin Li, Qi Xiong, and Fudong
Shao. dstyle-gan: Generative adversarial network based on writing and photog-
raphy styles for drug identification in darknet markets. In ACSAC , 2020.
[60] Yiming Zhang, Yiyue Qian, Yanfang Ye, and Chuxu Zhang. Adapting distilled
knowledge for few-shot relation reasoning over knowledge graphs. In SDM ,
pages 666–674. SIAM, 2022.
[61] Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. Calibrate
before use: Improving few-shot performance of language models. In ICML , pages
12697–12706. PMLR, 2021.