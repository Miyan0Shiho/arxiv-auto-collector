# Automated Label Placement on Maps via Large Language Models

**Authors**: Harry Shomer, Jiejun Xu

**Published**: 2025-07-29 18:00:22

**PDF URL**: [http://arxiv.org/pdf/2507.22952v1](http://arxiv.org/pdf/2507.22952v1)

## Abstract
Label placement is a critical aspect of map design, serving as a form of
spatial annotation that directly impacts clarity and interpretability. Despite
its importance, label placement remains largely manual and difficult to scale,
as existing automated systems struggle to integrate cartographic conventions,
adapt to context, or interpret labeling instructions. In this work, we
introduce a new paradigm for automatic label placement (ALP) that formulates
the task as a data editing problem and leverages large language models (LLMs)
for context-aware spatial annotation. To support this direction, we curate
MAPLE, the first known benchmarking dataset for evaluating ALP on real-world
maps, encompassing diverse landmark types and label placement annotations from
open-source data. Our method retrieves labeling guidelines relevant to each
landmark type leveraging retrieval-augmented generation (RAG), integrates them
into prompts, and employs instruction-tuned LLMs to generate ideal label
coordinates. We evaluate four open-source LLMs on MAPLE, analyzing both overall
performance and generalization across different types of landmarks. This
includes both zero-shot and instruction-tuned performance. Our results
demonstrate that LLMs, when guided by structured prompts and domain-specific
retrieval, can learn to perform accurate spatial edits, aligning the generated
outputs with expert cartographic standards. Overall, our work presents a
scalable framework for AI-assisted map finishing and demonstrates the potential
of foundation models in structured data editing tasks. The code and data can be
found at https://github.com/HarryShomer/MAPLE.

## Full Text


<!-- PDF content starts -->

Automated Label Placement on Maps via Large Language Models
Harry Shomer∗
University of Texas at Arlington
Arlington, Texas
harry.shomer@uta.eduJiejun Xu
HRL Laboratories
Malibu, California
jxu@hrl.edu
Abstract
Label placement is a critical aspect of map design, serving as a form
of spatial annotation that directly impacts clarity and interpretabil-
ity. Despite its importance, label placement remains largely manual
and difficult to scale, as existing automated systems struggle to
integrate cartographic conventions, adapt to context, or interpret
labeling instructions. In this work, we introduce a new paradigm
for automatic label placement (ALP) that formulates the task as a
data editing problem and leverages large language models (LLMs)
for context-aware spatial annotation. To support this direction, we
curate MAPLE, the first known benchmarking dataset for evalu-
ating ALP on real-world maps, encompassing diverse landmark
types and label placement annotations from open-source data. Our
method retrieves labeling guidelines relevant to each landmark
type leveraging retrieval-augmented generation (RAG), integrates
them into prompts, and employs instruction-tuned LLMs to gen-
erate ideal label coordinates. We evaluate four open-source LLMs
on MAPLE, analyzing both overall performance and generalization
across different types of landmarks. This includes both zero-shot
and instruction-tuned performance. Our results demonstrate that
LLMs, when guided by structured prompts and domain-specific re-
trieval, can learn to perform accurate spatial edits, aligning the
generated outputs with expert cartographic standards. Overall,
our work presents a scalable framework for AI-assisted map fin-
ishing and demonstrates the potential of foundation models in
structured data editing tasks. The code and data can be found at
https://github.com/HarryShomer/MAPLE.
CCS Concepts
•Computing methodologies →Machine learning .
Keywords
Automated Label Placement, Map Finishing, Large Language Model
ACM Reference Format:
Harry Shomer and Jiejun Xu. 2025. Automated Label Placement on Maps
via Large Language Models. In Proceedings of 1st Workshop on AI for Data
Editing at KDD’25 (AI4DE@KDD’25). ACM, New York, NY, USA, 10 pages.
https://doi.org/XXXXXXX.XXXXXXX
∗Research done while an intern at HRL Laboratories
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
AI4DE@KDD’25, Toronto, ON, Canada
©2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-1-4503-XXXX-X/2018/06
https://doi.org/XXXXXXX.XXXXXXX1 Introduction
Maps are an omnipresent and vital part of our everyday lives. They
provide an intuitive interface for understanding the world around
us, allowing us to plan, navigate, and coordinate our future move-
ments. Due to their historical and practical significance, the dis-
cipline of cartography has emerged as a way of formalizing and
studying the mapmaking process. The primary goal of cartography
is to produce maps that are both accurate and easy to interpret.
One central problem in cartography is label placement [ 7,12].
Figure 1 illustrates the concept of label placement by comparing
unlabeled maps (left) with their labeled counterparts (right). Po-
sitioned near each landmark is a label placed in a contextually
appropriate manner. Placing these labels is a subtle but essential
aspect of map design. It requires balancing spatial proximity to the
referenced feature with overall readability and visual clarity. To
ensure consistency and legibility, cartographers typically follow
detailed guidelines that prescribe label positioning based on the
type of landmark and surrounding context. However, manual la-
bel placement is often tedious and time-consuming, as it requires
examining each feature individually and applying context-specific
rules. This challenge has also been recognized at the national scale:
the National Geospatial-Intelligence Agency (NGA), responsible
for producing mission-critical maps, recently issued a $708 mil-
lion contract for large-scale geospatial data labeling to support
automation in map production [ 23]. This investment underscores
the complexity and scale of the labeling problem in modern car-
tographic workflows, particularly for mission-critical geospatial
applications.
To address these challenges, automatic label placement (ALP)
[2,25,27,33] has emerged as a computational method for deter-
mining label positions algorithmically. Many solutions exist that
consider the problem of ALP on maps, including rule-based en-
gines embedded in Geographic Information Systems (GIS) soft-
ware, optimization-based formulations (e.g., integer programming
or heuristic search), and more recent machine learning methods
such as deep reinforcement learning, generative models, and graph-
based neural networks. These systems aim to maximize label cov-
erage while minimizing visual conflicts such as overlaps and mis-
alignment. Some approaches optimize label positions based on
geometric criteria, while others attempt to learn placement pat-
terns from annotated datasets. However, these methods tend to be
limited in several respects. First , they are often inflexible, requir-
ing significant manual configuration or retraining to adapt to new
map requirements or domain-specific conventions. Second , they
struggle to simultaneously handle the many constraints found in
real-world labeling scenarios, especially on dense or heterogeneous
maps. Third , and perhaps most importantly, they lack the ability to
interpret and apply textual labeling guidelines, such as those found
in cartographic guidebooks, that describe nuanced preferences forarXiv:2507.22952v1  [cs.HC]  29 Jul 2025

AI4DE@KDD’25, August 3, 2025, Toronto, ON, Canada Shomer et al.
Figure 1: Illustration of label placement on maps. The two
left subfigures show unlabeled maps, while the two right
subfigures show the corresponding labeled versions. The
zoomed-in insets highlight examples of labels positioned
near landmarks. The goal of our work is to develop an auto-
mated and scalable approach for placing labels at appropri-
ate locations on the map, respecting labeling guidelines, and
adapting to the surrounding context.
different landmark types, contextual relationships, or visual hier-
archies. As a result, these systems fall short in producing label
placements that are both scalable and semantically aligned with
human-designed cartographic standards.
These observations motivate us to ask – Can we design a method
for ALP that can efficiently consider a set of human-readable guide-
lines in its reasoning process? . To tackle this problem, we attempt
to use Large Language Models (LLMs) [ 15,32], to determine the
label placement. We consider the use of LLMs due to their ability
to flexibly reason on a variety of situations in a relatively efficient
manner. To incorporate the relevant labeling instructions from
a set of human-readable guidelines, we consider the use of Re-
trieval Augmented Generation (RAG) [ 20]. Our RAG pipeline is
designed to retrieve the relevant instructions for a given landmark
to aid the LLM in placing the label. To better orient the LLMs to
the task of ALP, we also try instruction tuning LLMs for the task
of map labeling via LoRA [ 18]. We also observe that there is no
open-source dataset benchmarking method for automatic
label placement on maps . To remedy this issue, we design a new
dataset for this problem, MAPLE –MapAutomatic PLacement
Evaluation. MAPLE contains 100 maps from three major cities and
over 1000 individual landmarks that require labeling. The data is ex-
tracted from the popular OpenStreetMap [ 16]. We further consider
the use of an external set of labeling guidelines to guide the label
placement on these maps. Our contributions can be summarized as
the following:
(1)We create the first open-source dataset for benchmarking
automatic label placement on maps – MAPLE .
(2)We design new strategies for both retrieving relevant guide-
lines for a landmark using RAG and instruction tuning LLMs
for ALP.(3)Through extensive experiments, we benchmark the ability
of four different open-source LLMs on our new dataset. We
further test their ability by type of landmark and various
prompt designs.
2 Background and Related Work
Automatic label placement is a longstanding challenge in cartog-
raphy and information visualization, involving the positioning of
textual annotations on maps and diagrams to maximize readabil-
ity and minimize overlap. Traditional methods have evolved over
time, incorporating rule-based heuristics, optimization techniques,
and, more recently, machine learning approaches. We categorize
existing work into the following four primary areas.
Rule-based Approaches : Rule-based approaches have histor-
ically served as the cornerstone of automated label placement in
cartography and geographic information systems (GIS). These sys-
tems formalize long-standing cartographic design principles, such
as minimizing label overlaps, maximizing proximity to labeled fea-
tures, and preserving legibility, into a sequence of deterministic
heuristics. One of the most influential early frameworks for these
rules is based on the cartographic theory developed by Imhof [19],
who outlined qualitative principles for label prioritization, type-
face choice, and positioning relative to points, lines, and areas.
Contemporary GIS software packages, such as Esri’s ArcGIS Pro,
implement these principles through the Maplex Label Engine, a
proprietary module that offers advanced controls for conflict res-
olution, label placement strategy (e.g., curved versus horizontal
text), and feature-specific behavior (e.g., roads versus rivers) [ 9].
Similarly, the open-source QGIS project includes the PAL labeling
engine, which provides a flexible rule-based system with config-
urable priorities and spatial constraints [ 28]. Major web -mapping
platforms like Google Maps employ sophisticated rule -based en-
gines to prioritize feature types, detect collisions, and index spatial
data to automatically place labels during panning and zooming op-
erations [ 14]. These engines enable dynamic label positioning that
adapts to map scale and content density and are widely adopted
in professional cartographic workflows. While rule-based systems
are highly effective for standard mapping tasks, they inherently
lack semantic understanding of the features they label. They op-
erate on geometric constraints, without incorporating contextual
knowledge about landmark importance, category, or inter-feature
relationships. Our approach fundamentally differs from traditional
rule-based systems by leveraging large language models (LLMs) to
interpret and apply labeling instructions dynamically. This enables
greater adaptability and semantic awareness, allowing for more
nuanced label placements that are sensitive to both cartographic
conventions and the specific contextual properties of a given map.
Optimization and Mathematical Programming : Beyond
rule-based systems, optimization-based approaches offer a more
global perspective on the label placement problem by formulating it
as a constrained optimization task. These methods aim to discover
optimal or near-optimal label configurations that satisfy multiple
constraints, such as non-overlap and proximity to anchor features,
while maximizing objectives like label coverage or legibility. A wide
range of classical optimization paradigms have been explored in
this space, including Integer Linear Programming (ILP), Quadratic

Automated Label Placement on Maps via LLMs AI4DE@KDD’25, August 3, 2025, Toronto, ON, Canada
Programming, Simulated Annealing, and Genetic Algorithms [ 5].
These methods treat each label as a variable whose placement is
subject to constraints, and they search for a configuration that
minimizes a global cost function or maximizes layout utility. For in-
stance, metaheuristic techniques like the Hybrid Discrete Artificial
Bee Colony (HDABC) algorithm have shown promise in navigating
the vast search space of potential label positions efficiently [ 4]. A
particularly comprehensive treatment of these approaches is pro-
vided Niedermann [25], which systematically studies the use of
exact and approximate optimization techniques for various types
of map features, including point, line, and area labeling. Their work
introduces a modular framework that supports different mathe-
matical formulations, including Mixed-Integer Programming and
Constraint Satisfaction, and presents empirical evaluations across
multiple real-world datasets. Notably, they highlight the trade-offs
between optimality and computational feasibility, showing that
while exact methods often yield high-quality results, they can be
impractical for dense or large-scale maps without specialized heuris-
tics or problem decomposition. While these optimization techniques
offer significant improvements over purely rule-based systems, they
remain fundamentally geometric in nature. They typically do not
incorporate semantic, contextual, or user-driven considerations
into the placement process. Our work departs from these methods
by leveraging LLMs to synthesize placement instructions based
on both spatial attributes and semantic metadata, enabling a more
adaptable, context-aware approach that goes beyond geometric
optimization.
Deep Learning for Visual Label Prediction Recent advances
in deep learning have opened new directions for automating label
and layout generation tasks by learning spatial patterns directly
from data. In contrast to rule-based or optimization approaches,
deep learning methods model layout prediction as a supervised or
reinforcement learning task, using neural networks to learn spa-
tial relationships, feature importance, and layout aesthetics from
labeled examples. One early direction involves using convolutional
neural networks (CNNs) and generative models to synthesize la-
bel positions. For example, Oucheikh and Harrie [27] explored the
use of GANs (Generative Adversarial Networks) [ 13] trained on
expert-labeled maps to predict plausible label layouts. They show
promising results in reproducing stylistic elements of cartographic
design. Another line of work models the label placement problem
as a graph reasoning task. For example, Qu et al . [29] proposed a
Graph Transformer architecture that treats labels and landmarks as
nodes and learns to place them based on their spatial and semantic
relationships. Reinforcement learning has also been applied to label
placement through a learning-based optimization lens. In particular,
Bobák et al . [2] introduced a multi-agent deep reinforcement learn-
ing (MADRL) framework where each label acts as an autonomous
agent. These agents learn through trial-and-error to maximize over-
all label completeness and minimize collisions, outperforming con-
ventional rule-based and heuristic strategies in dense feature maps.
Beyond cartographic applications, recent work has demonstrated
the potential of large language models (LLMs) for general-purpose
visual layout generation. LayoutGPT [ 11] treats layout generation
as a visual planning task, using in-context learning with LLMs to
produce complex web-style layouts from textual prompts. Similarly,
Design2Code [ 31] benchmarks the ability of multimodal LLMs suchas GPT-4V and Gemini Vision Pro to convert visual designs into
HTML/CSS code, showing that LLMs can reason about layout struc-
ture and spatial alignment with minimal task-specific supervision.
Our approach builds on this line of work by introducing a retrieval-
augmented prompting method that incorporates landmark-specific
labeling guidelines into a language model to predict ideal label
coordinates. By combining spatial inputs (e.g., visual coordinates)
with semantic metadata (e.g., landmark types), our method extends
deep learning beyond layout reproduction toward context-aware,
convention-compliant label planning in map finishing.
Cognitive Studies on Label Preferences In addition to al-
gorithmic correctness, effective label placement also depends on
how humans perceive and interpret spatial arrangements. A grow-
ing body of research in cartographic design and visual cognition
highlights that factors such as alignment, spacing, and grouping
significantly influence the usability and readability of labeled maps.
For example, Scheuerman et al. [ 30] conducted user studies show-
ing that participants consistently preferred label placements with
clearer alignment and visual symmetry, even when these conflicted
with purely geometric optima. Similarly, Bobák et al. [ 3] found that
users overwhelmingly favored labels placed directly above point
features, contradicting traditional top-right placement conventions
long assumed to be optimal. Such findings suggest that traditional
evaluation metrics like overlap minimization or proximity alone
are insufficient for ensuring perceptual quality.
These insights expose a key limitation of many rule-based and
optimization-driven systems: while they may produce technically
valid layouts, they often fail to account for human-centered criteria
such as aesthetic balance or visual hierarchy. Efforts to incorporate
user feedback or style preferences into labeling systems exist but
are often domain-specific and require manual configuration or re-
training. Although our method does not explicitly model human
preferences, this line of work underscores the need for more adap-
tive, semantically informed approaches to layout generation. By
moving beyond rigid rules or fixed optimization objectives, we aim
to create a system that can more flexibly respond to diverse labeling
goals, potentially including human-centric instructions in future
iterations. The growing body of perceptual research provides a
valuable foundation for such extensions and reinforces the need to
consider not just what is correct, but also what is visually effective.
3 Dataset Construction
In this section we detail the construction of a new dataset for bench-
marking the task of automatic label placement on maps, MAPLE –
MapAutomatic PLacement Evaluation. The dataset is constructed
from OpenStreetMap (OSM) [ 16] and contains 100 maps from mul-
tiple cities. Each map contains, on average, 13 different landmarks
that require labeling, with a total of 1276 total landmarks. An exam-
ple of two maps is shown in Figure 1. To guide the label placement
for each landmark, we consider the use of a publicly available label
placement guideline.
The dataset construction process contains three main steps: (a)
Collecting the raw data from OSM, (b)Determining the label lo-
cation for each landmark, (c)Identifying and processing a set of
labeling guidelines. In the rest of this section, we detail each step.

AI4DE@KDD’25, August 3, 2025, Toronto, ON, Canada Shomer et al.
Figure 2: The pipeline for extracting the label text from a map. For a map we first detect the location of all text in the map. Then
for each piece of recognized text, we perform optical character recognition (OCR) to convert it to a machine-readable format.
3.1 Data Collection using OpenStreetMap
The first step to building our new dataset MAPLE is to collect a set
of maps. We consider a map to be a small area that contains roughly
10-20 different landmarks. For each map, we require the following
pieces of information to test any automatic label placement method:
(1) An image of the map with the landmark labels included.
(2)An image of the map without the labels included. This al-
lows us to generate new maps with different algorithms.
(3)The name and location of each landmark in the map. The
location is expressed as a set of 𝑘coordinates, which is neces-
sary to account for the variety of different shapes a landmark
can take.
(4)The location of the corresponding text for each landmark on
the map (i.e., “the label”).
(5)The type of landmark (e.g., an “office” or “shop”). This is nec-
essary as labeling instructions often differ based on landmark
type.
For retrieving this information, we use OpenStreetMap (OSM) [ 16],
which is a free and publicly available platform that provides accu-
rate map data, including detailed information about landmarks in a
given region.
Once we identify a single region that we wish to extract, we
perform the following steps. First, we use QGIS [ 28] to retrieve an
image of the map with and without the labels of each landmark.
QGIS is a free geographic information software that is used for
viewing or manipulating various types of map data. We use QGIS
for this step as it is highly customizable, allowing us to retrieve
images of maps without the labels. This is not possible with the
native OSM software. Second, we query the OSM API [ 16] using
the region’s coordinates to extract metadata for each landmark.
This includes their name, geographic location, and type. By default,
OSM uses the EPSG coordinate system to identify each landmark
and map. However, since each map is only a small region, the use of
a global coordinate system is unnecessary. As such, we replace the
coordinates of each landmark with its pixel values for that map’s
image. This allows us a much simpler and easier identification
system for determining the location of landmarks for a single map.
Finally, to focus the dataset and reduce complexity, we restrict our
extraction to a curated set of commonly occurring landmark types.In total, we include 7 landmark categories, as summarized in Table 1,
which also reports the number of instances collected for each type.
Table 1: Dataset Statistics by Landmark Type.
Type # Landmarks
Tourism 176
Shop 209
Amenity 572
Leisure 94
Office 62
Building 491
Place 15
The above procedure is repeated 100 times for different areas.
To enhance the diversity of the maps we extract, we consider maps
from three different locations. This includes Seattle, Los Angeles,
and San Francisco. The number of maps from each city, along with
the number of total landmarks, are shown in Table 2.
Table 2: Dataset Statistics by City.
City # Maps # Landmarks
Los Angeles 34 605
San Francisco 33 562
Seattle 33 453
3.2 Determining the Correct Label Location
In the Section 3.1 we discussed how the raw data is retrieved from
OSM. However, one piece of information that we can not retrieve
via OSM is the location of each landmark’s label on the map. This
is because the labeling system used by QGIS (i.e., PAL) [ 28] is not
exposed via their API. For example, in Figure 1 we can see that the
map contains the label “Azure Apartments” to mark the location of
that landmark. The precise location on the map of where the label
“Azure Apartment” is vital, as it serves as the ground truth for any
ALP method. However this location is not accessible from OSM. As
such, in order to evaluate any ALP method, we must determine a

Automated Label Placement on Maps via LLMs AI4DE@KDD’25, August 3, 2025, Toronto, ON, Canada
Figure 3: The pipeline for determining the true label location for a single landmark. We first (1) extract the area around the
landmark in question (2) find all recognized text (shown in Figure 2) in that area (3) filter for text found in the landmark’s
name (4) union the location of all identified text to form the final label location.
strategy for determining the ground truth location of each landmark’s
label .
To overcome this problem, we consider a text detection and
recognition pipeline to appropriately determine the label for each
landmark. Specifically, given a single map, we perform the following
steps: (a)Detect all text in the map, (b)For each piece of detected
text, we use Optical Character Recognition (OCR) to convert it to
a machine-readable format, (c)Assign each piece of recognized
text to the proper landmark, (d)Determine the ground truth label
location for a landmark as the union of all detected text assigned
to it. We now detail each step in this pipeline.
Text Detection : We first must determine the location of all the text
in a given map. To achieve this, we use DBNet++ [ 21]. DBNet++ is
a text detector which proposes a differentiable binarization module
to enhance text detection capabilities. The output of this algorithm
is a bounding box that describes the location of any piece of text
identified in the map.
Text Recognition : Now that we’ve identified where all the text
on a map is located, we must now determine what that text says.
Currently, the text is simply part of an image, which while read-
able for a human, is not by a machine. To recognize the text we
use ABINet [ 10]. ABINet uses a vision transformer along with a
bidirectional language model to recognize a piece of text. We run
ABINet on each piece of detected text for a map. An example of the
text detection and recognition pipeline for a single map is shown
in Figure 2.
Text Assignment : We now have the location and actual text of
all the words on a map. However, how do we assign each word to
the correct landmark ? For example, in Figure 2 we can see that the
words “Columbus Avenue” are right next to the landmark for the“North Beach Branch Library”. How do we determine which words
are the correct text for the library’s label? To achieve this we first
assume that the label must be nearby it’s landmark. Specifically we
assume that it must be no more that 𝑝pixels from the boundary
of the landmark. This is reasonable as labels almost always either
intersect with the landmark itself or are very close by. In prac-
tice we find that p=50px works well. Next, we extract all detected
text that intersect with the landmark. For example, in Figure 2 for
“North Beach Branch Library” this might be the words [“Columbus”,
“Avenue”, “North”, “Beach”, “Branch”, “Library”, “Mason”, “Street”].
We note that a single detected word can only be assigned to one
landmark. We then check if the intersecting words are contained
in the landmark’s name. In practice, we use fuzzy matching instead
of exact matching, as the output of any OCR method may contain
some errors. We specifically use Levenshtein distance with an 80%
distance threshold. We give an example of this process in Figure 3
in steps 1-3.
Ground Truth Label Determination : For every landmark in
a map, the previous step gives us a set of nearby words that are
contained in the landmark’s name. Furthermore we have the lo-
cation of each word in the image, in the form of a bounding box,
via text detection. We can then use that information to determine
the ground truth locations of the landmark’s label by simply com-
bining the location of each word. Let us assume that a landmark
is assigned𝑛words with locations [𝑤1,𝑤2,···,𝑤𝑛]. The ground
truth location of the label, 𝑙is given by the union of each such that
𝑙=∪𝑛
𝑖=1𝑤𝑖.
In practice we are not able to determine the ground truth label
location for all landmarks. This is because either the recognized
text: (a) wasn’t detected, (b) is far away from the landmark, (c) is too
different from it’s true value (due to poor text recognition). With
that said, in practice we find that this pipeline is highly effective

AI4DE@KDD’25, August 3, 2025, Toronto, ON, Canada Shomer et al.
in determining the label location for most landmarks. Specifically,
we found that for about 87% of maps , we are able to determine a
label location, thus validating our overall pipeline.
3.3 Labeling Guidelines
In the last two sections we discussed how we constructed our
dataset from OpenStreetMap [ 16]. This includes, the collection
of the individual maps, landmark location, and our pipeline for
determining the location of each landmark’s label. However, an
important question to consider is how do we guide any algorithm to
properly determine the location of a label ?
As previously discussed in Section 2 many existing approaches
either use rule-based approaches [ 28], optimization techniques [ 25],
or visual label prediction [ 11]. However, a drawback of these ap-
proaches is that they all rely on hard constraints, and with any
change in the underlying label placement guidelines, the approach
itself needs to be either modified. As such, we consider a different
and more flexible approach to determining label placement. Instead,
we consider the use of human-readable guidelines for label
placement . These guidelines can they be used by a model to guide
their reasoning when deciding where to place a label. Such guide-
lines are common in practice, as they are often used by professional
organization to standardize their procedure for map labeling.
For the choice of guidelines, we use those published by the Na-
tional Geospatial-Intelligence Agency (NGA) [ 24]. These guidelines
are used by the United States to create accurate maps for national
security operations. Later in Section 4.2 we describe how these
guidelines are processed and stored by our proposed methods.
4 Methods
In the previous section we introduced a new dataset, MAPLE ,
for benchmarking automatic label placement (ALP). To guide the
correct label placement, MAPLE includes a set of textual labeling
guidelines that describe how different landmarks should be labeled.
However, as discussed earlier in Section 2, previous approaches
to ALP are unable to flexibly use a set of textual guidelines to
steer their decision making process. In the following subsection we
introduce our proposed solution for overcoming this issue.
4.1 Overall Design
To enable the task of ALP given a set of textual guidelines, we
consider the use of Large Language Models (LLMs) [ 15,32]. This
is due to LLMs strong ability to understand textual information
and perform various reasoning tasks. To incorporate the use of
the external labeling guidelines, we use LLMs in conjunction with
Retrieval Augmented Generation (RAG) [ 20]. Specifically, our RAG
pipeline first stores the instructions in a vector database. Then when
labeling a specific landmark, we query the database to retrieve the
instructions that are personalized to that landmark. The information
about the landmark and the retrieved guidelines is then passed to
an LLM, which outputs the suggested coordinates of our label. An
overview of the framework is given in Figure 4.
In designing this framework, there are a few questions that must
be considered. (1)How do we process the labeling guidelines for
storage in a vector database? (2)How do we retrieve the relevant
instructions for a specific landmark? (3)How do we most optimallyprompt the LLM for our task? (4)Can we fine-tune the LLMs for
better downstream performance? In the following subsections we
answer these questions.
4.2 Vector Database Construction
In order to retrieve the pertinent instructions via RAG, we must
have a way of storing the instructions. Vector databases [ 17] are
a common method of doing so. They operate by storing each doc-
ument as a vector, where the vector is often an encoded version
of the underlying data. By storing the data as vectors, we enable
quick retrieval through vector similarity measures.
In order to store the guidelines in a vector database, we first
must decide at what granularity the vectors should be. A common
method in RAG is to chunk the data into fixed chunks, where each
chunk contains 𝐶tokens. However, a concern with this method is
that many instructions may have a length longer or shorter than
𝐶. As such, using a fixed chunk strategy may result in us splitting
certain labeling instructions into multiple chunks or combining
multiple instructions into one vector. Therefore, a fixed chunk size
strategy is not optimal for our task .
Instead, we choose to encode each section as its own vector.
As such, each entry in our database corresponds to one section
of instructions. This ensures that a single vector contains all the
labeling instructions of only one type of landmark. Given the text
of each section, they are encoded using the nomic-embed-text [ 26]
text embedder.
4.3 Prompting LLMs for Labeling
Using an LLM in conjunction with RAG requires two important
considerations. First, we must design a strategy for retrieving the
most accurate set of instructions from our vector database for each
landmark. Second, careful consideration needs to be given to the
prompt design for our task. This is crucial, as the prompt can often
have a strong impact on the performance in many tasks.
First, to retrieve the correct instructions, we use both the name
and type of the landmark. This information is encoded using the
nomic-embed-text text embedder [ 26]. To identify relevant instruc-
tions, we perform a vector similarity search against our instruction
database and return the top 𝑘most similar entries. These instruc-
tions are then re-ranked based on their relevance to the specific
landmark context, considering factors such as keyword overlap,
instruction specificity, or optional metadata if available. The text
of the top𝑘instructions is then concatenated and appended to the
prompt for use by the LLM.
With the corresponding instructions, we are now ready to prompt
the LLM to determine the location to place the label. The design of
the prompt is shown in Figure 5. We denote the landmark-specific
components in red. This includes the retrieved instructions, the
name and type of the landmark, the location of the landmark, and
how we format the location. As a reminder, the location is com-
posed of a set of coordinates used to mark the boundary of the
location and is retrieved from OpenStreetMap. Crucially, we try
multiple strategies for formatting the location. This is because a
specific LLM may be suited to understand certain formats over
others when determining the location on an image. For example,
previous work has shown that LLMs can generate CSS for the task

Automated Label Placement on Maps via LLMs AI4DE@KDD’25, August 3, 2025, Toronto, ON, Canada
Figure 4: Overall framework for performing ALP using LLM+RAG. We first retrieve the correct instructions using both the
name and type of the landmark. This information is then included in a prompt to the LLM, along with the location of the
landmark, for the LLM to determine where to place the label.
of visual planning [ 11]. We try four different formatting strate-
gies, where the coordinates are represented as: List, JSON, CSS,
XML. For example, in the list format, the coordinates may look like
“[(100,150),(250,300),(100,400),(250,500)]”. Later, in Section 5.2,
we find that the type of format can indeed have a large impact on
performance.
Figure 5: Prompt used to ask LLM for labeling. Those values
in red are specific to each landmark.
4.4 Instruction-tuning LLMs for Labeling
In the previous section, we describe a method for prompting LLMs
to place the label of landmarks given a set of retrieved instructions.
However, a concern is that current LLMs may not be suitably op-
timized for this task. As such, using the existing LLMs may result
in poor performance. Due to this, we consider tuning each LLM to
better align it with our task.
We specifically consider instruction tuning each LLM for our
task [ 34]. Instruction tuning involves giving an LLM a set of instruction-
response pairs, where the LLM takes the instruction as input and
must learn to give the appropriate response. In our case, the in-
struction will be the prompt shown in Figure 5, and the response is
the output (X, Y) coordinates of where to place the label. We shown
an example in Figure 6 where the LLM must learn to respond to
theInstruction with the correct Response .
In order to tune each LLM, we use LoRA [ 18]. Instead of mod-
ifying the original weights of the LLM, LoRA instead learns an
additional set of low rank weight matrices that are then added
to the original weights of the LLM. This allows for much more
efficient tuning due to only having to learn much fewer weights.
We further consider the quantized version of LoRA, QLoRA [ 6], to
further enhance the training efficiency.
Figure 6: Format of instruction tuning strategy. The LLM is
trained to respond to the instruction with the correct (X, Y)
coordinates, which correspond to the ground truth location
of the label.
Figure 7: Format of the prompt when including neighboring
landmarks. This is appended to the original prompt in Fig-
ure 5.
4.5 Including Neighboring Context
Up until now, when prompting the LLM, we only include infor-
mation about the landmark itself. However, it may be that a given
landmark is situated close to several other landmarks. This can
have the effect of altering the labeling instructions. This is intuitive,
as it could be that the location where the label should be placed
may be occupied by another landmark or another label itself. For
example, in Figure 1 we can see that “Azure Apartment” is very
close to “Happy Trails”. This can affect the label placement of both
landmarks so as they don’t interfere with one another.

AI4DE@KDD’25, August 3, 2025, Toronto, ON, Canada Shomer et al.
Table 3: Overall Results (RMSE) by LLM, format, and strategy.
Format W/o Tuning With Tuning
Llama3.1 Gemma2 Qwen3 Phi-4 Llama3.1 Gemma2 Qwen3 Phi-4
List 97.4 96.1 88.1 81.2 32.8 32.3 33.6 28.4
CSS 165.0 117.0 114.7 225.1 41.8 37.3 41.8 35.3
JSON 142.8 95.6 80.0 82.2 37.7 30.1 36.4 29.5
XML 121.4 101.4 104.5 85.3 31.8 31.7 32.8 29.7
To account for the neighbors of a given landmark, we include
them when: (1)retrieving the labeling instructions from the data-
base, (2)prompting the LLM to place the label. When retrieving the
instructions, this is simply done by including the names and types
of all nearby landmarks. When prompting the LLM, the type and
location of all nearby landmarks are further included in the prompt.
The LLM is further instructed to consider these nearby landmarks
when determining where to place the label. We detail the specific
prompt in Figure 7. In practice we only include those neighbors
that are 50px or closer to the landmark being labeled.
5 Experiments
In this section, we conduct extensive experiments to validate the
effectiveness of the proposed method on the MAPLE dataset. Specif-
ically, we attempt to determine: (RQ1) How well can various SOTA
open-source LLMs perform? (RQ2) Does the performance vary by
the type of format used to represent the coordinates and by the
type of landmark? (RQ3) Can fine-tuning each LLM help improve
performance? (RQ4) Can adding information about neighboring
landmarks enhance the performance?
5.1 Experimental Settings
Dataset : The MAPLE dataset contains 1276 maps in total. They
are split into training, validation, and test sets using an 80/10/20%
random split, respectively. The final number of landmarks for train,
validation, and test is 883, 126, 267, respectively. Note that the train-
ing and validation maps are only used when instruction-tuning the
LLMs.
Models : We test four different open-source and SOTA LLMs. This
includes: Llama3.1 (8B) [ 15], Gemma2 (9B) [ 32], Qwen3 (8B) [ 32],
Phi-4 (14B) [1].
Tuning Strategy : For each model, we also consider instruction
tuning them to enhance their applicability towards the task of label
placement. This is done through the use of QLoRA [ 6] for enhanced
efficiency. We train for 5 epochs, with a learning rate of 1e-5 and a
weight decay of 1e-5.
Evaluation : To evaluate the prediction quality, we consider the dis-
tance between the predicted location and the label. This is done by
comparing the predicted location ˆ𝑦𝑖and the centroid of the ground
truth label𝑦𝑖. Using these values for each landmark, we compute
the RMSE across all samples 𝑖=1to𝑁.5.2 Main Results
The main results are shown in Table 4. They are broken down by:
LLM, with and w/o tuning, and by the location format. Inspecting
the results, we can make a few observations. (1)Some LLMs perform
better than others. When not tuning, we can see that Qwen3 and
Phi-4 tend to perform best, while with tuning, Phi-4 and Gemma 2
are best. (2)Instruction tuning has a large positive effect on perfor-
mance. Comparing the best performance of each LLM between the
two strategies, their performance decreases by almost 200% . This
indicates that the pre-trained LLM weights are likely not optimal
for our task and require some additional supervision. (3)We also
observe that the location format has a very large effect on perfor-
mance. In Figure 8, we show the mean performance by format type
across all LLMs. We can see that regardless of whether we tune
the LLM, the performance varies considerably. Specifically, we find
that the performance tends to be the most consistently strong when
using the List format. On the other hand, formatting the coordinates
via CSS tends to be the worst. Interestingly, however, when tuning,
XML tends to perform best. These results underline that how we
represent the location of each landmark does indeed matter and is
an important consideration for ALP using LLMs.
Figure 8: Mean performance across LLMs by type of coordi-
nate format.
5.3 Results by Landmark Type
We further display the results by type of landmark. As shown earlier
in Table 1, there are 7 different types of landmarks in our dataset.
In order to understand the strengths and weaknesses of current
LLMs for ALP, we must determine if it is better suited for labeling
certain types of landmarks over others.
We show the results for the two best LLMs, Qwen3 and Phi-4
in Table 4. The results are displayed with and w/o tuning. We first
observe that performance decreases across every type when tuning,
suggesting that it benefits all types of landmarks. Second, both LLMs

Automated Label Placement on Maps via LLMs AI4DE@KDD’25, August 3, 2025, Toronto, ON, Canada
tend to struggle on both “Leisure” and “Building” landmarks. This is
true even when tuning, despite the fact that “Building” landmarks
have the second most number of samples in MAPLE. On the other
hand, both LLMs tend to perform very well on those landmarks of
type “Shop” and “Office”. This suggests that the quality of ALP using
LLMs can vary quite drastically by the type of landmark. Lastly,
both LLMs are fairly consistent in their performance by type, with
their overall trends being quite similar.
Table 4: Results (RMSE) by type of landmark. Coordinates
are formatted as a list.
Type Qwen3 Phi-4
w/o tune with tune w/o tune with tune
Tourism 46.9 24.5 39.6 16.5
Shop 26.7 13.9 27.7 8.7
Amenity 60.4 22.9 61.1 23
Leisure 137.7 52.4 127 36.3
Office 21.9 8.9 23.5 5.7
Building 116.2 41.6 111.2 39.9
Place 41.4 22.3 40.3 10.3
5.4 Results with Neighboring Context
We also experiment with adding the neighbors of a given landmark
as described in Section 4.5. In Table 5 we show the mean results with
and w/o the neighboring context. We further display the results
with and w/o tuning. For simplicity, we only test the List coordinate
format, as it is the format that the LLMs perform most consistently
well on.
Interestingly, we observe that there is often no benefit to this ad-
ditional context. Specifically, outside of two scenarios (highlighted
inbold ), the performance always slightly decreases. This raises
the question of whether the neighboring landmarks are indeed
helpful for labeling. It may also suggest that our current strategy
for considering neighboring context is suboptimal and may need
some refinement.
Table 5: Mean results (RMSE) across LLMs by w/o and with
neighboring context. Coordinates are formatted as a list.
LLM W/o Tuning With Tuning
std + neighbors std + neighbors
Llama3.1 97.4 104.5 32.8 36.9
Gemma2 96.1 101.9 32.3 30.5
Qwen3 88.1 90.3 33.6 35.7
Phi-4 81.2 81.0 28.4 29.5
6 Conclusion
In this paper we study the problem of automatic label placement
(ALP) on maps. We find that no open-source datasets exist for
benchmarking this task. To remedy this issue, we propose a new
dataset, MAPLE, that contain 100 maps from three cities with over
1000 landmarks to be labeled. MAPLE also includes a set of la-
beling guidelines, meant to provide instructions on how to labelvarious types of landmarks correctly. We experiment with using
Large Language Models (LLMs) for this task where the relevant
labeling instructions are retrieved using Retrieval Augmented Gen-
eration (RAG). We experiment with multiple prompting strategies
along with instruction tuning. We show that LLMs can indeed per-
form ALP, however their performance differs by LLM and type
of landmark. We further find that instruction tuning can help to
dramatically improve the performance of LLMs for our task. For fu-
ture work, we hope to experiment with other prompting strategies
including in-context learning [ 8]. We also hope to make use of the
visual layout of the map itself through the use of vision language
models (VLMs) [22].

AI4DE@KDD’25, August 3, 2025, Toronto, ON, Canada Shomer et al.
References
[1]Marah Abdin, Jyoti Aneja, Harkirat Behl, Sébastien Bubeck, Ronen Eldan, Suriya
Gunasekar, Michael Harrison, Russell J Hewett, Mojan Javaheripi, Piero Kauff-
mann, et al .2024. Phi-4 technical report. arXiv preprint arXiv:2412.08905 (2024).
[2]Petr Bobák, Ladislav Čmolík, and Martin Čadík. 2023. Reinforced Labels: Multi-
agent deep reinforcement learning for point-feature label placement. IEEE Trans-
actions on Visualization and Computer Graphics 30, 9 (2023), 5908–5922.
[3]Petr Bobák, Ladislav Čmolík, and Martin Čadík. 2024. From Top-Right to User-
Right: Perceptual Prioritization of Point-Feature Label Positions. arXiv preprint
arXiv:2407.11996 (2024).
[4]Wen Cao, Jiaqi Xu, Yong Zhang, Siqi Zhao, Chu Xu, and Xiaofeng Wu. 2023.
A hybrid discrete artificial bee colony algorithm based on label similarity for
solving point-feature label placement problem. ISPRS International Journal of
Geo-Information 12, 10 (2023), 429.
[5]Jon Christensen and Joe Marks. 1995. An empirical study of algorithms for point
feature label placement. (1995).
[6]Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. 2023.
Qlora: Efficient finetuning of quantized llms. Advances in neural information
processing systems 36 (2023), 10088–10115.
[7]Donald W. Hamer Center for Maps and Geospatial Information. 2025. Maps
& Geospatial. https://www.e-education.psu.edu/geog486/node/557 Accessed:
2025-05-02.
[8]Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Jingyuan Ma, Rui Li, Heming Xia,
Jingjing Xu, Zhiyong Wu, Tianyu Liu, et al .2022. A survey on in-context learning.
arXiv preprint arXiv:2301.00234 (2022).
[9]Esri. 2024. Label with the Maplex Label Engine - ArcGIS Pro . https://pro.arcgis.com/
en/pro-app/latest/help/mapping/text/label-with-the-maplex-label-engine.htm
Accessed April 27, 2025.
[10] Shancheng Fang, Hongtao Xie, Yuxin Wang, Zhendong Mao, and Yongdong
Zhang. 2021. Read like humans: Autonomous, bidirectional and iterative language
modeling for scene text recognition. In Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition . 7098–7107.
[11] Weixi Feng, Wanrong Zhu, Tsu-jui Fu, Varun Jampani, Arjun Akula, Xuehai He,
Sugato Basu, Xin Eric Wang, and William Yang Wang. 2023. Layoutgpt: Compo-
sitional visual planning and generation with large language models. Advances in
Neural Information Processing Systems 36 (2023), 18225–18250.
[12] Kenneth Field. 2018. Cartography. Esri Press.
[13] Ian J Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley,
Sherjil Ozair, Aaron Courville, and Yoshua Bengio. 2014. Generative adversarial
nets. Advances in neural information processing systems 27 (2014).
[14] Google. 2025. Google Maps API v3 Reference. https://developers.google.com/
maps/documentation/javascript/reference Accessed: 2025-05-03.
[15] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek
Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex
Vaughan, et al .2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783
(2024).
[16] Mordechai Haklay and Patrick Weber. 2008. Openstreetmap: User-generated
street maps. IEEE Pervasive computing 7, 4 (2008), 12–18.
[17] Yikun Han, Chunjiang Liu, and Pengfei Wang. 2023. A comprehensive survey
on vector database: Storage and retrieval technique, challenge. arXiv preprint
arXiv:2310.11703 (2023).
[18] Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu
Wang, Weizhu Chen, et al .2022. LoRA: Low-Rank Adaptation of Large Language
Models. In International Conference on Learning Representations .
[19] Eduard Imhof. 1975. Positioning names on maps. The American Cartographer 2,
2 (1975), 128–144.
[20] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
et al.2020. Retrieval-augmented generation for knowledge-intensive nlp tasks.
Advances in neural information processing systems 33 (2020), 9459–9474.
[21] Minghui Liao, Zhaoyi Wan, Cong Yao, Kai Chen, and Xiang Bai. 2020. Real-time
scene text detection with differentiable binarization. In Proceedings of the AAAI
conference on artificial intelligence , Vol. 34. 11474–11481.
[22] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual in-
struction tuning. Advances in neural information processing systems 36 (2023),
34892–34916.
[23] National Geospatial-Intelligence Agency. 2024. Sequoia: NGA’s largest data
labeling effort to date. https://www.nga.mil/news/NGA_announces_$708M_
data_labeling_RFP.html Accessed: 2025-05-02.
[24] National Geospatial-Intelligence Agency/Foundation GEOINT Group. 2022. NGA
Standardization Document, Data Product Specification (DPS). https://www.nga.
mil/resources/Print_on_Demand_(PoD).html.
[25] Benjamin Niedermann. 2017. Automatic Label Placement in Maps and Figures:
Models, Algorithms and Experiments . Ph. D. Dissertation. Dissertation, Karlsruhe,
Karlsruher Institut für Technologie (KIT), 2017.
[26] Zach Nussbaum, John X Morris, Brandon Duderstadt, and Andriy Mulyar. 2024.
Nomic embed: Training a reproducible long context text embedder. arXiv preprintarXiv:2402.01613 (2024).
[27] Rachid Oucheikh and Lars Harrie. 2024. A feasibility study of applying generative
deep learning models for map labeling. Cartography and Geographic Information
Science 51, 1 (2024), 168–191.
[28] QGIS Project. 2024. QgsLabelingEngine Class Reference - QGIS API Documentation .
https://api.qgis.org/api/classQgsLabelingEngine.html Accessed April 27, 2025.
[29] Jingwei Qu, Pingshun Zhang, Enyu Che, Yinan Chen, and Haibin Ling. 2024.
Graph Transformer for Label Placement. IEEE Transactions on Visualization and
Computer Graphics (2024).
[30] Jaelle Scheuerman, Jason L Harman, Rebecca R Goldstein, Dina Acklin, and
Chris J Michael. 2023. Visual preferences in map label placement. Discover
Psychology 3, 1 (2023), 27.
[31] Chenglei Si, Yanzhe Zhang, Zhengyuan Yang, Ruibo Liu, and Diyi Yang. 2024.
Design2code: How far are we from automating front-end engineering? arXiv
e-prints (2024), arXiv–2403.
[32] Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy
Hardin, Surya Bhupatiraju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari,
Alexandre Ramé, et al .2024. Gemma 2: Improving open language models at a
practical size. arXiv preprint arXiv:2408.00118 (2024).
[33] Huafei Yu, Tinghua Ai, Min Yang, Rachid Oucheikh, Bo Kong, Hao Wu, Zhenyu
Zhang, and Lars Harrie. 2024. A Deep Learning Segmentation Approach for Road
Label Placement. Available at SSRN 4940850 (2024).
[34] Shengyu Zhang, Linfeng Dong, Xiaoya Li, Sen Zhang, Xiaofei Sun, Shuhe Wang,
Jiwei Li, Runyi Hu, Tianwei Zhang, Fei Wu, et al .2023. Instruction tuning for
large language models: A survey. arXiv preprint arXiv:2308.10792 (2023).