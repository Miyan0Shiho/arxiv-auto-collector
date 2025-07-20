# Introducing the Swiss Food Knowledge Graph: AI for Context-Aware Nutrition Recommendation

**Authors**: Lubnaa Abdur Rahman, Ioannis Papathanail, Stavroula Mougiakakou

**Published**: 2025-07-14 11:12:30

**PDF URL**: [http://arxiv.org/pdf/2507.10156v1](http://arxiv.org/pdf/2507.10156v1)

## Abstract
AI has driven significant progress in the nutrition field, especially through
multimedia-based automatic dietary assessment. However, existing automatic
dietary assessment systems often overlook critical non-visual factors, such as
recipe-specific ingredient substitutions that can significantly alter
nutritional content, and rarely account for individual dietary needs, including
allergies, restrictions, cultural practices, and personal preferences. In
Switzerland, while food-related information is available, it remains
fragmented, and no centralized repository currently integrates all relevant
nutrition-related aspects within a Swiss context. To bridge this divide, we
introduce the Swiss Food Knowledge Graph (SwissFKG), the first resource, to our
best knowledge, to unite recipes, ingredients, and their substitutions with
nutrient data, dietary restrictions, allergen information, and national
nutrition guidelines under one graph. We establish a LLM-powered enrichment
pipeline for populating the graph, whereby we further present the first
benchmark of four off-the-shelf (<70 B parameter) LLMs for food knowledge
augmentation. Our results demonstrate that LLMs can effectively enrich the
graph with relevant nutritional information. Our SwissFKG goes beyond recipe
recommendations by offering ingredient-level information such as allergen and
dietary restriction information, and guidance aligned with nutritional
guidelines. Moreover, we implement a Graph-RAG application to showcase how the
SwissFKG's rich natural-language data structure can help LLM answer
user-specific nutrition queries, and we evaluate LLM-embedding pairings by
comparing user-query responses against predefined expected answers. As such,
our work lays the foundation for the next generation of dietary assessment
tools that blend visual, contextual, and cultural dimensions of eating.

## Full Text


<!-- PDF content starts -->

Introducing the Swiss Food Knowledge Graph:
AI for Context-Aware Nutrition Recommendation
Lubnaa Abdur Rahman
Graduate School for Cellular and
Biomedical Sciences
University of Bern
Bern, Switzerland
lubnaa.abdurrahman@unibe.chIoannis Papathanail
University of Bern
Bern, Switzerland
ioannis.papathanail@unibe.chStavroula Mougiakakou
University of Bern
Bern, Switzerland
stavroula.mougiakakou@unibe.ch
Abstract
Artificial intelligence has driven significant progress in the nutrition
field, especially through multimedia-based automatic dietary as-
sessment. However, existing automatic dietary assessment systems
often overlook critical non-visual factors, such as recipe-specific
ingredient substitutions that can significantly alter nutritional con-
tent, and rarely account for individual dietary needs, including
allergies, restrictions, cultural practices, and personal preferences.
In Switzerland, while food-related information is available, it re-
mains fragmented, and no centralized repository currently inte-
grates all relevant nutrition-related aspects within a Swiss context.
To bridge this divide, we introduce the Swiss Food Knowledge
Graph (SwissFKG), the first resource, to our best knowledge, to
ever unite recipes, ingredients, and their substitutions with nutri-
ent data, dietary restrictions, allergen information, and national
nutrition guidelines under one graph. We establish a Large Lan-
guage Model (LLM)-powered enrichment pipeline for populating
the graph, whereby we further present the first benchmark of four
off-the-shelf (<70 B parameter) LLMs for food knowledge augmen-
tation. Our results demonstrate that LLMs can effectively enrich
the graph with relevant nutritional information. Our SwissFKG
goes beyond recipe recommendations by offering ingredient-level
information such as allergen and dietary restriction information,
and guidance aligned with nutritional guidelines. Moreover, we
implement a Graph-Retrieval Augmented Generation (Graph-RAG)
application to showcase how the SwissFKG’s rich natural-language
data structure can help LLM answer user-specific nutrition queries,
and we evaluate LLM-embedding pairings by comparing user-query
responses against predefined expected answers. As such, our work
lays the foundation for the next generation of dietary assessment
tools that blend visual, contextual, and cultural dimensions of eat-
ing.
CCS Concepts
•Computing methodologies →Natural language processing .
Keywords
Food Knowledge Graph, Personalized Nutrition, Large Language
Models, Graph-RAG
1 Introduction
Nutrition plays a critical role in addressing pressing health and
environmental challenges [ 50]. Non-communicable diseases (NCDs)
currently account for roughly 75% of all deaths each year, with
cardiovascular diseases (CVDs) being the world’s leading causeof mortality [ 73]. Diabetes prevalence is projected to climb to 853
million by 2050 [ 26]. Food allergies are also on the rise: an estimated
4.3% of the global population lives with food allergies [ 15]. At the
same time, growing health and environmental concerns are driving
dietary shifts toward plant-based eating [ 68]. In Switzerland, these
global patterns are reflected locally: CVDs accounted for 27.5% of
all deaths in 2022 [ 57], diabetes affects nearly 500,000 people [ 65],
and an estimated 4–8% of the population lives with food allergies
[4]. Concurrently, efforts to reduce meat consumption are gaining
momentum [79].
The field of nutrition has witnessed remarkable growth in re-
cent years, with Artificial Intelligence (AI) playing a pivotal role
in automating processes such as dietary assessment. Image-based
automatic dietary assessment systems have shown immense po-
tential over the past years, from everyday dietary monitoring to
hospitalized setups [ 12,35,36,49,79]. However, a fundamental
limitation of these tools lies in their inability to capture non-visual
information, particularly differences in recipes and ingredient-level
substitutions when it comes to home-cooked meals and obstruc-
tions within layered foods [ 3]. Composed meals often vary widely
in their nutritional profiles depending on the specific ingredients
used, which cannot be deduced from images alone and require fur-
ther context. This poses a significant risk, particularly when dietary
assessments inform critical decisions like insulin dosing recommen-
dations, where inaccurate nutritional estimates may lead to serious
health implications, such as severe hypoglycemic episodes [47].
To overcome these limitations, dietary systems should adopt a
holistic framework integrating recipes, along with contextual fac-
tors like user preferences, allergen sensitivities, cultural practices,
and health goals. Traditional "one-size-fits-all" dietary guidelines
often overlook these personal nuances, reducing their real-world
effectiveness. Personalized nutrition bridges this gap by creating
tailored plans while respecting individual needs, habits, and pref-
erences like dietary restrictions (DR) with growing evidence high-
lighting positive impacts on health outcomes [58].
More recently, there has been a surge in leveraging Large Lan-
guage Models (LLMs) to power nutrition recommendation systems
[48]. LLMs open up exciting possibilities for natural, conversational
interaction with users, making it easier to ask questions and re-
ceive dietary guidance, a feature lacking from existing automatic
dietary assessment systems. However, LLMs are trained on vast,
often unreliable, and culturally nonspecific internet data, making
their recommendations sometimes untrustworthy and even poten-
tially harmful [43, 51]. Although LLMs have been applied to tasksarXiv:2507.10156v1  [cs.AI]  14 Jul 2025

Abdur Rahman L. et al.
like recipe generation [ 27], many do not constrain outputs to veri-
fied knowledge or assess their credibility [ 59], possibly leading to
nutritionally unsound or harmful recommendations and eroding
user trust [6].
Despite growing demand for nutritional awareness, nutritional
information remains fragmented across sources. Diverse cultural,
personal, and regional food practices call for flexible, unified guide-
lines that honor individual needs and local traditions. This un-
derscores the need for a centralized, adaptable system to provide
coherent, culturally sound, and personalized dietary recommenda-
tions. Building on these, we introduce the Swiss Food Knowledge
Graph (SwissFKG), the first comprehensive and cohesive repository
combining recipes, nutrient values sourced from the Swiss Food
Composition Database (FCDB) [ 64], allergen information, guide-
lines from the Swiss Food Pyramid (SFP) [ 14], and DRs. To enhance
our graph, we propose a semi-automated enrichment pipeline lever-
aging LLMs. We further present a Graph-Retrieval Augmented
Generation (Graph-RAG) pipeline tailored to question-answering
(QA) over the SwissFKG. We evaluate LLMs and embedding models
on various enrichment tasks, as well as their combined performance
within the Graph-RAG framework over the SwissFKG. Through
this work, we lay the foundation for a general-population use case
while paving the way toward a global Food Knowledge Graph (FKG)
built on prior efforts [ 19,21,31] to deliver verified, personalized
nutrition-related recommendations that can be further integrated
into existing automatic dietary systems.
2 Related Works
2.1 Knowledge Graphs and Conversational
Agents
In a landscape increasingly shaped by LLMs, their growing adop-
tion, driven by conversational abilities [ 63], has led to widespread
integration into applications with LLMs passing exams in multiple
fields [ 28,78]. Notably, chatbots [ 25] are leveraging LLMs for QA
tasks, while recommendation systems [ 34] are incorporating them
to allow users to express queries naturally. Notably, their use has
expanded into critical domains such as healthcare, where conversa-
tional AI agents powered by LLMs assist in diagnosis and health
guidance [ 2]. However, deploying LLMs for sensitive recommen-
dations, particularly in healthcare, poses substantial risks if the
underlying knowledge is not rigorously validated. Usage of LLMs,
to date, remains a popular topic of discourse due to their black-box
nature [ 5,10] and the lack of credible audits of the web-scale data
on which they are trained [ 71]. These limitations leave LLMs prone
to errors and hallucinations [ 7], with LLMs often providing confi-
dent justifications for incorrect answers [ 5]. To reduce such errors,
techniques like Chain-of-Thought prompting (CoT) [ 72] were in-
troduced to encourage step-by-step reasoning in LLMs, followed
by Graph-RAG [ 20] to ground responses in credible knowledge.
Graph-RAG leverages well-constructed Knowledge Graphs (KGs)
to retrieve verifiable information, thereby improving the credibility
and transparency of LLM-based answers.
As such, there is an increasing interest in developing KGs that in-
tegrate disparate knowledge sources into unified repositories. KGs
are graph-structured repositories representing entities as nodes
and their relationships as edges [ 22], enabling the expression ofrich, near-natural-language relationships. Generally, building a KG
involves extracting and merging facts from various structured or
unstructured sources, reconciling entities, and establishing rela-
tionships guided by an underlying ontology. KGs have found appli-
cations across numerous AI tasks, particularly in personalization,
reasoning, and recommendation, notably within disease manage-
ment [ 60]. In recommendation systems specifically, KGs serve as
valuable sources of supplementary information, linking users not
only with the items they interact with, but also with the detailed
attributes of those items. This facilitates more informed sugges-
tions compared to collaborative filtering methods alone [ 18,54].
Consequently, KGs provide a structured foundation that allows AI
systems to draw upon encoded domain knowledge, enabling more
consistent and context-aware outcomes. Various methodologies for
constructing KGs have been explored, ranging from fully manual
approaches to semi-automated methods, where LLMs enrich KG
data, to fully automated techniques in which LLMs autonomously
define both the structure and content of the graph [24, 29, 76, 77].
2.2 Food Knowledge Graphs
In the food and nutrition domain, foundational ontologies and
knowledge bases form the structured backbone for downstream
applications. Early overviews document efforts to standardize food
concepts and resolve vocabulary gaps across datasets [ 30,67,70].
Among these, FoodOn classifies products, ingredients, and sources
by organism, anatomical part, and processing method, enabling
global traceability and quality control [ 11]. Ontologies also under-
pin semantic recommendation systems, aligning food categories
and nutrient relations with user profiles, yet they typically omit
cooking steps and nuanced dietary preferences, leaving room for
more recipe- and user-aware models [39, 46].
Several projects build on this foundation to construct recipe-
centric KGs. FoodKG integrates crowdsourced recipes with the
USDA FCDB and maps ingredients to FoodOn classes, supporting
nutritional QA and recommendations [ 21]. Likewise, Recipe1M+
has been used to link recipes, ingredients, and nutrient facts, but
still treats each ingredient as a fixed node without modeling equiv-
alents or substitutes [ 31,38]. Others have even been more spe-
cific and have created national FKGs [ 19]. A parallel stream em-
beds nutritional and health considerations directly into FKGs. Diet-
management ontologies encode food groups and nutrition tar-
gets, for instance, assisting diabetic menu planning [ 41]. RecipeKG
goes further by computing healthiness scores against sugar-and-fat
thresholds, enabling queries like “find a quick snack under 100 calo-
ries” and driving healthy recommendations [ 8]. However, these re-
sources assume that meeting nutritional targets guarantees dietary
appropriateness, neglecting critical factors such as food allergies,
intolerance, and individual taste preferences. In [ 9], the authors
use a case-based reasoning system with manually encoded substi-
tution rules (e.g., butter →margarine for vegan variants) to tweak
recipes to user needs. More recently, graph-embedding approaches
rank substitutes by both co-occurrence similarity and a diet-aware
substitution suitability metric that factors in allergen removal or
calorie reduction [61].

Introducing the Swiss Food Knowledge Graph
3 Materials and Methods
3.1 Data collection
Recipes. We crawled and curated 1,000 recipes from well-known
Swiss culinary websites. Along the way, we filtered out flawed
entries with invalid ingredients, missing instructions, or exact du-
plicates. If multiple recipes shared a name but differed in content,
we kept them all and assigned unique IDs. To guarantee structural
consistency, we designed a custom JSON schema optimized for the
KG construction and manually mapped all key entities. Recipes
gave a solid basis to our FKG by providing recipe macro-nutrition,
a list of ingredients, utensils, instructions, seasons, cuisines, and
certain keywords for tagging recipes.
Food databases. To enhance the graph at the ingredient level, we
enrich it with nutrient profiles for each ingredient. For nutritional
profiling, we primarily utilized the Swiss FCDB, which provides
comprehensive macro- and micronutrient data for 1,146 food items
commonly consumed in Switzerland [ 64]. In cases where a spe-
cific ingredient was not available in the Swiss FCDB, the USDA’s
FoodData Central was used as a secondary reference source [ 69].
It features extensive, well-documented nutritional data for over
10,000 items, including branded and experimental foods. To support
ingredient-level substitutions, we incorporated data from Foodsubs
[1], which lists substitution options for 3,177 unique ingredients.
We further included glycemic index (GI) values for ingredients in
the KG to cater for the diabetic population. Since GI values were
missing from both the Swiss and USDA FCDBs, we sourced this
data from FoodStruct [ 16], which compiles GI information from
scientific publications and provides values for 615 individual food
items or ingredients.
Guidelines and Recommendations. The integration of dietary rec-
ommendations is essential for a comprehensive FKG. Switzerland
defines 14 allergen categories as part of its food regulations, out-
lined in the regulations on Allergen Information [ 13]. These cover
the mandatory labeling of common allergens like gluten, eggs, milk,
nuts, and soy [ 4]. However, the regulation only describes the cat-
egories; it does not provide a complete mapping from individual
ingredients to their respective allergen category. We also wanted
to include the SFP guidelines published by the Federal Food Safety
and Veterinary Office [ 14]. These define nine main food groups
according to the Swiss pyramid. While some of this information
was already included in the Swiss FCDB, it wasn’t available for all
the ingredients we were working with.
3.2 Data Enrichment
3.2.1 Translation of Non-English Content. Half of the recipes in
our dataset were already in English, and for these, we also had
French versions which we to evaluate translations. The rest were
written in French, German, or Italian. We translated all non-English
content (primarily French) using an LLM. To evaluate the LLM
translation capabilities, we used the Crosslingual Optimized Metric
for Evaluation of Translation (COMET)[ 55] score, an automatic
metric that compares machine translations against human refer-
ences. Unlike simpler methods, COMET considers meaning, fluency,
and alignment with both the source and the reference text. We used
thewmt22-comet-da model weights, focusing on ingredient listsand instructions, which are key components for FKG construction.
COMET encodes three texts ( s,t,r) into vectors, combines them
with their differences, and outputs a score correlating strongly with
human evaluations. This ensures translations are both faithful to
the source and fluent compared to references.
The COMET score is computed as follows:
h𝑠=Enc(s) (1)
h𝑡=Enc(t) (2)
h𝑟=Enc(r) (3)
COMETscore(s,t,r)=FFN([h𝑠,h𝑡,h𝑟,h𝑠−h𝑡,h𝑡−h𝑟])(4)
where
sis the original text,
tis the LLM’s English translation,
ris a reference translation,
Enc(·)converts text into vectors,
FFN(·)is the regression head that predicts
the quality of translation.
3.2.2 Recipes: Cuisines, Diets, and Seasons’ suitability. To build rich
entities and relationships, we used recipe names, descriptions, and
keywords. We applied a predefined list of the four seasons and 18
DR, based on WHO [ 74], with religious diets further divided into
multiple categories. We also used a list of 100 international cuisines
to map recipes where possible. For example, “Vegan Swiss Summer
Breadrolls” implied Swiss cuisine, summer suitability, and a vegan
(and therefore vegetarian) diet. All recipes were additionally labeled
as “unrestricted” to indicate they are suitable for those without DR.
3.2.3 Ingredients: Text Splitting, Allergens, SFP Categories, DRs.
Text Splitting. We found inconsistencies in the crawled recipe
data, including irregular ingredient names, utensils listed as ingre-
dients, non-standard terms, and misplaced or missing measurement
units. To clean this, we used an LLM to normalize ingredient names
to singular, canonical forms; extract preparation details into a notes
field; correctly assign quantities and units; and move utensils to a
separate field. For instance, ingredient text “a lemon, zest grated,
and 1/2 juiced” becomes: Ingredient: lemon; Quantity: 1; Unit: –;
Notes: zest grated; ½juiced. We then used a pattern matching ap-
proach based on [ 53] to score the output. Given a predicted output
string𝑎and the expected string 𝑏- which we consider as ground
truth, we define the similarity score as
Similarity(𝑎,𝑏)=2𝑀
|𝑎|+|𝑏|(5)
where
𝑎is the predicted output string,
𝑏is the ground-truth string,
𝑀is the total number of matching characters found by
recursively identifying longest common substrings.
|𝑎|and|𝑏|denote the lengths of strings 𝑎and𝑏, respectively.

Abdur Rahman L. et al.
This score ranges from 0 (no overlap) to 1 (perfect match). By
computing Similarity(𝑎,𝑏), we obtain a single metric that reflects
how closely each model’s output aligns with the expected split.
DR, Allergens, SFP Categories. To enrich the KG with Swiss nu-
tritional metadata, we used LLMs to infer DR to the 18 defined
categories, assign allergens based on the 14 groups and map in-
gredients to the 9 SFP categories. Recipe-level DR were further
enhanced based on the intersection of ingredient-level labels. As
an evaluation metric, we used F1 scores adapted to the label type.
For multi-label fields (e.g., allergen or pyramid categories), we com-
puted a set-based F1 score using eq. (6), while for binary diet flags,
we used the standard F1-score per category in eq. (8), whereby the
final diet score was computed as the mean F1 across all 18 DR labels.
𝐹1= 
1 if𝑆true=𝑆pred=∅
0 if|𝑆true∩𝑆pred|=0
2·𝑃·𝑅
𝑃+𝑅otherwise(6)
𝑃=|𝑆true∩𝑆pred|
|𝑆pred|, 𝑅 =|𝑆true∩𝑆pred|
|𝑆true|(7)
𝐹1=2·Precision·Recall
Precision+Recall(8)
Precision =𝑇𝑃
𝑇𝑃+𝐹𝑃,Recall =𝑇𝑃
𝑇𝑃+𝐹𝑁(9)
where:
𝑆trueis the set of true (ground-truth) labels ,
𝑆predis the set of predicted labels ,
𝑇𝑃(true positives) = # of labels correctly predicted ,
𝐹𝑃(false positives) = # of extra labels predicted ,
𝐹𝑁(false negatives) = # of true labels missed .
3.2.4 Ingredient nutritional content and substitution. While most
recipes included energy and nutrient values, we extended this by
attributing nutritional content to each ingredient (and substitute
ingredient). We first queried the Swiss FCDB [ 64] and the USDA’s
FoodData Central [ 69] when needed using exact matches. If no di-
rect match was found, we used embedding-based cosine similarity
to identify semantically close ingredients, allowing us to recover in-
formation despite naming variations, synonyms, or minor spelling
differences. In our case, the cosine similarity measures how close
two ingredients are in meaning based on their vector representa-
tions. Given two ingredient embeddings ®ing.1and®ing.2, the cosine
similarity is computed as:
Cosine Similarity(®ing.1,®ing.2)=®ing.1·®ing.2
∥®ing.1∥∥®ing.2∥(10)
For evaluation, we framed the task as a retrieval problem and
used a binary correctness criterion per query. For each query, we
computed cosine similarities between its embedding and all candi-
date item embeddings, and selected the candidate with the highest
similarity score as the predicted match. Although ties did not occur
in our case, they can be handled by returning all top candidates andchecking if the ground truth is among them, or resolving manually.
The accuracy can be given by:
Accuracy =1
𝑁𝑁∑︁
𝑖=11 𝑋𝑖=𝑌𝑖(11)
where
𝑁is the total number of queries,
𝑋𝑖is the predicted match for query 𝑖(highest-similarity item),
𝑌𝑖is the ground-truth item for query 𝑖,
1(·)is the indicator function (1 if true, 0 otherwise).
3.3 Knowledge Graph
We built the SwissFKG based on the available data and according
to the ontology shown in fig. 1. The KG includes the following
node types: Recipe ,Instruction ,Ingredient (including substitute-
ingredient andcomposite substitute variants), DR,Season ,Cuisine ,
Utensil ,Allergen Category , and SFP Category . Node properties
are shown in the "{}". The relationships connecting these entities
are structured as follows:
•CONTAINS : Connects a Recipe to its constituent Ingredi-
ents, with associated quantity, unit, and preparation notes.
•IS_SUITABLE_FOR : Links a Recipe or an Ingredient to
theDRs they comply with.
•IS_FOR_SEASON : Links a Recipe to aSeason based on
its contextual suitability.
•IS_PART_OF : Links a Recipe to a specific Cuisine .
•USES : Connects a Recipe to required Utensil s, with op-
tional quantity and descriptive notes. Connects an Instruc-
tion to the Ingredient s used within that step, including
quantity, unit, and notes.
•HAS : Connects a Recipe to its stepwise Instruction s.
•ALLERGEN_OF : Connects an Ingredient to its associated
Allergen Category (as defined in Swiss Annex 6).
•CLASSIFIED_AS : Connects an Ingredient to aSFP Cate-
gory , reflecting its nutritional classification.
•SUBSTITUTED_BY : Links an Ingredient to a substitute
Ingredient , along with the quantity ratio and any substi-
tution notes.
•HAS_COMPOSITE_SUBSTITUTE : Links an Ingredient
to aComposite Substitute node, which aggregates multi-
ple ingredients.
•COMPOSED_OF : Connects a Composite Substitute to
its underlying Ingredient s, with associated substitution
quantity and notes.
3.4 Graph-RAG over SwissFKG
To further demonstrate the practical applicability of the Swiss-
FKG, we implemented a Graph-RAG pipeline [ 20]. The knowl-
edge in the graph is represented as structured triplets in the form
(subject, predicate, object); for example, (Recipe:ApplePie, CON-
TAINS_INGREDIENT, Ingredient:Apples), where each triplet ex-
presses the factual relationship between two entities. We enrich
this representation by incorporating both entity properties (e.g., the
nutritional composition of apples) and relationship properties (e.g.,
the quantity of apples used in that relationship). To enable semantic

Introducing the Swiss Food Knowledge Graph
Figure 1: Knowledge Graph Nodes, Node Properties, and Relationships
and structural retrieval, we generate graph embeddings [ 33] that
capture the meaning of entities, their relationships, and the broader
graph topology. These embeddings allow us to efficiently retrieve
relevant information based on user queries, supporting accurate
and context-aware answer generation through the LLM. When a
user submits a query, our retrieval mechanism first uses an LLM to
extract key concepts, keywords, and potential synonyms from the
input. These extracted elements seed a search over graph nodes. In
parallel, we embed the user query and use cosine similarity, with a
cut-off of 0.5, against the graph embeddings to retrieve contextu-
ally aligned knowledge from the graph. We re-rank the retrieved
knowledge and keep the 10 most relevant ones based on the cosine
similarity to prevent exceeding the context length. By comparing
the embeddings of the user query with those of the graph content,
we identify and retrieve the most relevant knowledge from the
graph. The final step involves passing this retrieved graph infor-
mation to an LLM, which synthesizes the structured data into a
coherent, natural-language answer. We further evaluated the per-
formance of the Graph-RAG pipeline across different combinations
of LLM and embedding models. For this, we used the following
metric to compare the final output to the expected output:
Accuracy =1
𝑁𝑁∑︁
𝑖=11(𝑌𝑖⊆𝑋𝑖) (12)where
𝑁is the total number of questions evaluated,
𝑋𝑖is the model’s response to question 𝑖,
𝑌𝑖is the expected answer to question 𝑖,
1(·)is the indicator function, which returns 1 if the
condition𝑌𝑖⊆𝑋𝑖is true, and 0 otherwise.
3.5 Experimental Setup
All experiments were conducted on NVIDIA RTX A6000 GPU, with
LLMs and embedding models served through Ollama (version 0.9.0)
[45]. To ensure a fair comparison, we used the same prompt and
fixed generation settings across all LLMs: zero temperature for
deterministic output, a fixed random seed for reproducibility, a
4096-token context window, and restrictive sampling (top- 𝑝=0,
top-𝑘=1). For models that support internal reasoning, the thinking
mode was explicitly disabled. For LLM system prompts, the CoT
prompting technique was used. Additionally, for all enrichment
tasks, LLMs were asked to respect a specific JSON format when pro-
viding outputs with examples provided as part of system prompts
demonstrating sample input and expected output.
To address incomplete or erroneous data in our initial knowl-
edge sources and prepare them for integration into the FKG, we
leveraged LLMs to enrich content according to our target ontology.

Abdur Rahman L. et al.
To balance performance, multilingual support, and efficiency un-
der limited resources, we selected recent (2024–2025) open-source
LLMs under 70B parameters. These models performed well on
benchmarks, especially instruction-following, and offered a strong
trade-off between capability and cost for the enrichment task. We
used the following models: Phi-4 (14B) [40],Mistral Small 3.2
(24B) [42],Gemma3 (27B) [17], and Qwen3 (30B-A3B) [52]. For
identifying semantically similar nutrition terms, suggesting ingre-
dient substitutions, and mapping GI values, we experimented with
the following embedding models: textttAll MiniLM 33M [ 56],Nomic
Embed [44], Mxbai Embed Large [32].
We evaluated the LLM and embedding models on various enrich-
ment tasks using a curated expected output considered as ground
truth covering 10% of the data: 100 recipes and 200 ingredients. For
translation quality, we extracted English–French recipes and had a
native bilingual speaker verify translations across three semantic
targets: name, instructions, and ingredients. For ingredient-related
tasks, including text splitting, mapping DR, allergens, SFP cate-
gories, and ingredient matching, we constructed dedicated ground
truths for the ingredient entries. To demonstrate SwissFKG’s appli-
cability as a proof of concept, we combined the two best-performing
LLM models from the enrichment tasks with the different embed-
ding models in a Graph-RAG pipeline. We evaluate this setup on
50 curated questions generated from the KG to measure the effec-
tiveness of prompting LLMs within our approach.
4 Results
4.1 Data Enrichment
Figure 2: Performance of LLMs on enrichment tasks. COMET
scores reported for translation tasks, Similarity for text split-
ting task and F1 scores for Ingredient-to-allergen, -SFP, and
-DR mappings4.1.1 Translation of Non-English Content. Table 1 reports average
COMET scores per model. Gemma3 (27B) consistently outper-
formed others in all aspect, with scores of 0.8061 ,0.8376 , and
0.8761 indicating strong alignment with reference translations in
food-specific contexts. Mistral Small 3.2 (24B) followed closely, par-
ticularly on instructions and ingredients. Phi-4 (14B) performed
reasonably but dropped on instructions. Notably, Qwen3 (30B-A3B)
yielded the lowest scores overall, though it still maintained reliabil-
ity with all scores above 0.78.
Model Name Instructions Ingredients
Gemma3 (27B) 0.8061 0.8376 0.8761
Mistral Small 3.2 (24B) 0.8037 0.8364 0.8715
Phi-4 (14B) 0.8047 0.8156 0.8694
Qwen3 (30B-A3B) 0.7831 0.8329 0.8660
Table 1: COMET scores for translation tasks
4.1.2 Ingredients: Text Splitting, Allergens, SFP Categories, DRs.
Text Splitting. Table 2 gives an insight to how well the models
were in following the given instructions for splitting text and fol-
lowing examples provided to them. Gemma3 (27B) achieved the
highest similarity score of 0.9578 .Phi-4 (14B) followed closely with
scores above 0.91, while Qwen3 (30B-A3B) also performed reliably.
Mistral Small 3.2 (24B) showed weaker performance here.
Model Text Splitting
Gemma3 (27B) 0.9578
Mistral Small 3.2 (24B) 0.8628
Phi-4 (14B) 0.9197
Qwen3 (30B-A3B) 0.9178
Table 2: Similarity scores for ingredient text splitting
DR, Allergens, SFP Categories. As shown in Table 3, Mistral
Small 3.2 (24B) achieved the best results in both allergen and
DR mappings with an F1 score of 0.947 and0.868 respectively,
demonstrating good generalization capabilities for the nutrition
domain. For SFP mapping, Phi-4 (14B) performed best with an F1
score of 0.8, despite its smaller size. Gemma3 (27B) andQwen3 (30B-
A3B) performed generally well for allergen detection, while for SFP
Categorization and DRs, results among LLMs were very close. The
majority of errors occurred when the mappings should have been
none, i.e., no category should have been attributed. For example,
with ingredient "icing sugar", the expected allergen should have
been None - yet all LLMs except Mistral Small 3.2 (24B) assumed
an allergen of category 1: gluten-containing products. Looking at
DR mappings across LLMs, the diabetic diet had the highest error
rates, ranging from 61-74%. On the other hand, gluten-free and
some religious diets were among the easiest for LLMs to map, with
error rates as low as 7%.

Introducing the Swiss Food Knowledge Graph
Model Allergen SFP DR
Gemma3 (27B) 0.923 0.78 0.794
Mistral Small 3.2 (24B) 0.947 0.795 0.868
Phi-4 (14B) 0.810 0.8 0.803
Qwen3 (30B-A3B) 0.912 0.765 0.790
Table 3: F1 scores for Ingredient-to -Allergen, -SFP, and -DR
mapping tasks
4.1.3 Ingredient nutritional content and substitution. Using cosine
similarity as the retrieval criterion, we found that, as shown in
Table 4, Mxbai Embed Large achieved the highest accuracy at
0.66, slightly outperforming All Minilm 33M , followed by Nomic
Embed . Looking deeper, ingredient matches using All MiniLM 33M
often resulted in low cosine similarity scores (<0.5), despite cor-
rect alignments. For example, the ingredient "Nori" was correctly
matched with "Seaweed, Nori, dried" with a similarity of 0.47, while
Nomic Embed yielded a higher score of 0.78 for the same pair.
Model Accuracy
All Minilm 33M 0.64
Mxbai Embed Large 0.66
Nomic Embed 0.61
Table 4: Accuracy of closest ingredient matching
4.2 Knowledge Graph
To construct the KG, we leveraged the top-performing models from
prior enrichment tasks - Gemma3 (27B) [ 17] for translation tasks
and text splitting, Mistral Small 3.2 (24B) [ 42] for allergens and
DRs mappings, Phi-4 (14B) [ 40] for SFP Category mappings, and
ingredient name matching was done using Mxbai Embed Large [ 32].
All the data together resulted in a SwissFKG with 5,896 nodes and
62,499 relations. As shown in Table 5, the node types with the largest
counts were Ingredient (including substitutions) (2,548), Instruction
(2,176), and Recipe (1,000). The breakdown of the relationships of the
graph is given in Table 6, whereby we noted a low number of recipes
and cuisines being linked, highlighting missing cuisine tagging in a
lot of instances from the crawled data. Our SwissFKG comprises 893
recipes that contain ingredients that are allergens, while the rest
are without. Additionally, the top 5 most used ingredients were salt,
pepper, olive oil, pepper, water, and sugar. Allergens that appeared
the most were of categories 7 (Dairy/Milk), 1 (Glutens), and 8 (Nuts).
Among the 1000 recipes, 584 are vegetarian friendly, 467 gluten-
free, over 500 were suitable for religious restrictions, and above 300
were low/free of lactose.
4.3 Graph-RAG over SwissFKG
Table 7 presents the performance of the two best overall performing
LLMs (based on enrichment tasks) when integrated into the Graph-
RAG pipeline using different embedding models over the SwissFKG
dataset. Among the tested LLMs, Gemma3 (27B) achieved the
highest accuracy of 0.80 when paired with the Mxbai Embed
Large embedding model. Both Gemma3 (27B) and Mistral Small 3.2Label Count
Ingredient 2548
Instruction 2176
Recipe 1000
Utensil 91
Cuisine 21
DietRestriction 19
AllergenCategory 14
CompositeSubstitute 14
SwissFoodPyramidCategory 9
Season 4
Table 5: Entity label distribution in the annotated dataset.
Source Labels Relationship Type Target Labels Count
Ingredient IS_SUITABLE_FOR DietRestriction 33694
Recipe CONTAINS Ingredient 10099
Instruction USES Ingredient 6073
Recipe IS_SUITABLE_FOR DietRestriction 6008
Ingredient CLASSIFIED_AS SwissFoodPyramidCategory 2301
Recipe HAS Instruction 2176
Ingredient ALLERGEN_OF AllergenCategory 1037
Recipe USES Utensil 548
Recipe IS_FOR Season 271
Ingredient SUBSTITUTED_BY Ingredient 158
Recipe IS_PART_OF Cuisine 67
CompositeSubstitute COMPOSED_OF Ingredient 46
Ingredient HAS CompositeSubstitute 21
Table 6: Counts of relationship types between node labels
(24B) performed the best using the Mxbai Embed Large embedding
model, and the worst with All MiniLM 33M. Swapping embedding
models with the same LLM led to up to a 16% difference in accuracy
for Gemma3 (27B) and 10% for Mistral Small 3.2 (24B). Furthermore,
we note that with the All MiniLM 33M embedding model, no knowl-
edge could be retrieved for 8 questions with Gemma3 (27B) and 13
questions with Mistral Small 3.2 (24B).
Model Name Embedding Model Accuracy
Gemma3 (27B) All MiniLM 33M 0.64
Gemma3 (27B) Mxbai Embed Large 0.80
Gemma3 (27B) Nomic Embed 0.72
Mistral Small 3.2 (24B) All MiniLM 33M 0.66
Mistral Small 3.2 (24B) Mxbai Embed Large 0.76
Mistral Small 3.2 (24B) Nomic Embed 0.74
Table 7: Accuracy of QA through Graph-RAG over SwissFKG
5 Discussion
Our comprehensive evaluation of LLM-based data enrichment and
Graph-RAG over the SwissFKG reveals several key insights into
the interplay between model scale, pretraining alignment, retrieval
representations, and domain-specific performance. Contrary to
common assumptions that larger parameter counts inherently yield
superior performance, our results demonstrate that model scale is
not the sole determinant of success across enrichment tasks. As

Abdur Rahman L. et al.
fig. 2 illustrates, Gemma3 (27B) achieved the highest COMET scores
on translation of names, instructions, and ingredients (Table 1), re-
flecting the strength of its multilingual pretraining. Mistral Small
3.2 (24B), despite being smaller than Gemma3 (27B), outperformed
all others on allergen mapping and DR classification. Phi-4 (14B),
even though being the smallest model, achieved the best alignment
with SFP categories. These insights underscore the fact that certain
models are more apt for certain tasks, and following instructions,
irrespective of their size. Despite explicit instructions to avoid un-
stated assumptions, LLMs made certain assumptions in allergen and
SFP mappings. That misstep underscores how some LLMs might
fail to internalize task directives and might not follow given ex-
amples. Even when All MiniLM correctly matches ingredients, it
often assigns low similarity scores, likely due to tightly clustered
embeddings that obscure true matches,suggesting that fine-tuning
on food-specific data, rather than using a fixed cutoff like 0.5, could
improve performance.
The poor performance of most LLMs in tagging the diabetic DR
likely stems from clinical guidelines favoring personalized patterns,
nutrient goals, and portion control over fixed lists, making the
“diabetic diet” highly variable, and from the sparse or conflicting
examples in general web-trained models due to the lack of stan-
dardized guidelines [ 62]. The low number of relationships linked
recipes to cuisines flags a key area for improvement within the
graph. Beyond the enrichment stage, Graph-RAG experiments over
our SwissFKG, even though with promising results of 80% over
50 questions, highlights the critical role of embedding represen-
tations in retrieval-augmented inference. The lower performance
particularly in the zero-information retrieval incidents (section 4.3),
further underscores the need for careful embedding selection when
constructing Graph-RAG pipelines relying on graph embeddings.
While our approach yielded promising results, several limitations
merit attention. Model size remains a constraint, as we evaluated
only non-domain-specific off-the-shelf LLMs up to 30B parameters;
larger or domain-adapted models may further enhance enrichment
quality. Our choice of metrics, COMET, character similarity, F1,
and cosine accuracy, captures only part of the picture, overlooking
hallucination rates, semantic drift, human alignment, and probable
structural inconsistencies with the graph schema. The current re-
trieval strategy, relying on static embeddings and cosine similarity,
could be enhanced by incorporating hybrid methods [ 37,75] that
combine dense semantic and sparse lexical retrieval or adaptive
techniques [ 23,66] that dynamically adjust thresholds and retrieval
based on query context, to improve relevance and robustness. Fi-
nally, automated enrichment, while efficient, still requires expert
oversight for low-confidence or novel mappings to safeguard data
integrity.
As future works, we propose to further expand the model reper-
toire and possibly fine-tune models to be more domain-specific,
particularly embedding models, deploying a dynamic ingestion
pipeline to keep nutritional data current, and introducing a human-
in-the-loop interface for validating uncertain graph entries. Further-
more, extending the Graph-RAG pipeline with multimodal inputs
and integrating multimodal LLMs for QA could enable processing of
meal photographs and evolve into a more capable nutritional agent.
User-centered evaluation with realistic professional queries could
help refine and stress-test the system for reliable deployment inreal-world settings. A promising direction for the future is integrat-
ing the SwissFKG into personalized dietary systems by combining
QA over the knowledge along with automatic dietary assessment
systems. Such integration could enable real-time, context-aware
nutritional guidance, thus bridging the gap between static food
databases and dynamic individual consumption patterns. As the
need for FKGs grows, our work contributes to ongoing efforts to
standardize and harmonize FKGs and can serve as a foundation for
assembling these into a unified global FKG, fostering cross-border
collaboration in nutrition research. While this work showcases a
general-use case for public nutrition recommendations, we aim
to extend the graph’s applications to advanced domains like clin-
ical nutrition, food chemistry, sustainability, and functional food
research.
6 Conclusion
In this work, we introduced SwissFKG, the first FKG, to the best
of our knowledge, that integrates Swiss recipes, ingredients, nutri-
tional profiles, allergen classifications, DR, and national nutrition
guidelines into a unified repository. At the core of SwissFKG is
a semi-automated enrichment pipeline powered by low-resource
(<70B parameters), open-source LLMs, demonstrating satisfactory
results across enrichment tasks. We further demonstrated the practi-
cal value of SwissFKG by using it as the backbone for a Graph-RAG
pipeline that enables LLMs to answer user queries grounded in
structured food knowledge with up to 80% accuracy over 50 ques-
tions. Our work lays an important foundation for future research
towards building a global FKG and paves the way for integration
with personalized dietary systems, where nutritional insights from
SwissFKG could enhance automatic dietary assessment systems for
personalized precision nutrition.
Acknowledgments
This work was partly supported by the European Commission and
the Swiss Confederation - State Secretariat for Education, Research
and Innovation (SERI) within the projects 101057730 Mobile Arti-
ficial Intelligence Solution for Diabetes Adaptive Care (MELISSA)
and 101080117 Preventing Obesity through Biologically and Behav-
iorally Tailored Interventions for You (BETTER4U).

Introducing the Swiss Food Knowledge Graph
References
[1]2025. FoodSubs: Ingredient Substitutions and Ingredient Synonyms. https:
//www.foodsubs.com/. Accessed: 2025-07-09.
[2] Mahyar Abbasian, Iman Azimi, Amir M Rahmani, and Ramesh Jain. 2023. Con-
versational health agents: A personalized llm-powered agent framework. arXiv
preprint arXiv:2310.02374 (2023).
[3] Lubnaa Abdur Rahman, Ioannis Papathanail, Lorenzo Brigato, Elias K Spanakis,
and Stavroula Mougiakakou. 2024. Food Recognition and Nutritional Apps. In
Diabetes Digital Health, Telehealth, and Artificial Intelligence . Elsevier, 73–83.
[4] aha! Swiss Allergy Centre. 2024. Food allergy. https://www.aha.ch/swiss-allergy-
centre/allergies-intolerances/food-allergies/food-allergy. Last update: 13 August
2024.
[5] Rohan Ajwani, Shashidhar Reddy Javaji, Frank Rudzicz, and Zining Zhu. 2024.
LLM-generated black-box explanations can be adversarially helpful. arXiv
preprint arXiv:2405.06800 (2024).
[6] Grace Ataguba and Rita Orji. 2025. Exploring Large Language Models for Per-
sonalized Recipe Generation and Weight-Loss Management. ACM Transactions
on Computing for Healthcare (2025).
[7]Yejin Bang, Ziwei Ji, Alan Schelten, Anthony Hartshorn, Tara Fowler, Cheng
Zhang, Nicola Cancedda, and Pascale Fung. 2025. Hallulens: Llm hallucination
benchmark. arXiv preprint arXiv:2504.17550 (2025).
[8]Charalampos Chelmis and Bedirhan Gergin. 2021. A Knowledge Graph for
Semantic-Driven Healthiness Evaluation of Recipes.
[9] Amélie Cordier, Valmi Dufour-Lussier, Jean Lieber, Emmanuel Nauer, Fadi Badra,
Julien Cojan, Emmanuelle Gaillard, Laura Infante-Blanco, Pascal Molli, Amedeo
Napoli, et al .2014. Taaable: a case-based system for personalized cooking.
Successful Case-based Reasoning Applications-2 (2014), 121–162.
[10] Murillo Edson de Carvalho Souza, Murillo Edson De Carvalho Souza, and Li
Weigang. 2025. Unveiling the Black Box: The Significance of XAI in Making
LLMs Transparent. Authorea Preprints (2025).
[11] Damion M Dooley, Emma J Griffiths, Gurinder S Gosal, Pier L Buttigieg, Robert
Hoehndorf, Matthew C Lange, Lynn M Schriml, Fiona SL Brinkman, and
William WL Hsiao. 2018. FoodOn: a harmonized food ontology to increase
global food traceability, quality control and data integration. npj Science of Food
2, 1 (2018), 23.
[12] Takumi Ege and Keiji Yanai. 2017. Estimating food calories for multiple-dish
food photos. In 2017 4th IAPR Asian Conference on Pattern Recognition (ACPR) .
IEEE, 646–651.
[13] Federal Department of Home Affairs (FDHA), Switzerland. 2016. Verordnung des
EDI vom 16. Dezember 2016 betreffend Information über Lebensmittel (OIDAl) –
Annex 6. https://www.fedlex.admin.ch/eli/cc/2017/158/fr#annex_6. SR817.022.16;
accessed June 26 2025.
[14] Federal Office of Public Health (BLV). 2025. Schweizer Ernährungsempfehlun-
gen. https://www.blv.admin.ch/blv/en/home/lebensmittel-und-
ernaehrung/ernaehrung/empfehlungen-informationen/schweizer-
ernaehrungsempfehlungen.html. Accessed June 26, 2025.
[15] Hua Feng, Xiujuan Xiong, Zhuo Chen, Qunying Xu, Zhongwei Zhang, Nan Luo,
and Yongning Wu. 2023. Prevalence and influencing factors of food allergy in
global context: a meta-analysis. , 320–352 pages.
[16] FoodStruct Nutrition & Medical Content Team. 2025. FoodStruct: Encyclopedia
of Food & Nutrition. https://foodstruct.com/. Accessed: 2025-07-09.
[17] Google DeepMind. 2025. Gemma3: A family of lightweight, state-of-the-art open
multimodal models with 128K context window, supporting 140+ languages and
function calling. https://deepmind.google/models/gemma/gemma-3/. Released
March 12, 2025.
[18] Qingyu Guo, Fuzhen Zhuang, Chuan Qin, Hengshu Zhu, Xing Xie, Hui Xiong,
and Qing He. 2020. A survey on knowledge graph-based recommender systems.
IEEE Transactions on Knowledge and Data Engineering 34, 8 (2020), 3549–3568.
[19] Saransh Kumar Gupta, Lipika Dey, Partha Pratim Das, and Ramesh Jain.
2024. Building FKG. in: a Knowledge Graph for Indian Food. arXiv preprint
arXiv:2409.00830 (2024).
[20] Haoyu Han, Yu Wang, Harry Shomer, Kai Guo, Jiayuan Ding, Yongjia Lei, Ma-
hantesh Halappanavar, Ryan A Rossi, Subhabrata Mukherjee, Xianfeng Tang,
et al.2024. Retrieval-augmented generation with graphs (graphrag). arXiv
preprint arXiv:2501.00309 (2024).
[21] Steven Haussmann, Oshani Seneviratne, Yu Chen, Yarden Ne’eman, James
Codella, Ching-Hua Chen, Deborah L McGuinness, and Mohammed J Zaki.
2019. FoodKG: a semantics-driven knowledge graph for food recommendation.
InThe Semantic Web–ISWC 2019: 18th International Semantic Web Conference,
Auckland, New Zealand, October 26–30, 2019, Proceedings, Part II 18 . Springer,
146–162.
[22] Aidan Hogan, Eva Blomqvist, Michael Cochez, Claudia d’Amato, Gerard De Melo,
Claudio Gutierrez, Sabrina Kirrane, José Emilio Labra Gayo, Roberto Navigli,
Sebastian Neumaier, et al .2021. Knowledge graphs. ACM Computing Surveys
(Csur) 54, 4 (2021), 1–37.
[23] Hsin-Ling Hsu and Jengnan Tzeng. 2025. DAT: Dynamic alpha tuning for hybrid
retrieval in retrieval-augmented generation. arXiv preprint arXiv:2503.23013(2025).
[24] Yujia Hu, Tuan-Phong Nguyen, Shrestha Ghosh, and Simon Razniewski.
2025. Enabling LLM Knowledge Analysis via Extensive Materialization.
arXiv:2411.04920 [cs.CL] https://arxiv.org/abs/2411.04920
[25] Shih-Hong Huang, Ya-Fang Lin, Zeyu He, Chieh-Yang Huang, and Ting-Hao Ken-
neth Huang. 2024. How does conversation length impact User’s satisfaction? A
case study of length-controlled conversations with LLM-powered Chatbots. In
Extended Abstracts of the CHI Conference on Human Factors in Computing Systems .
1–13.
[26] International Diabetes Federation. 2025. Global Diabetes Data & Insights. https:
//diabetesatlas.org/data-by-location/global/ Data by location: Global; accessed 7
July 2025.
[27] Ponrawin Kansaksiri, Pongpipat Panomkhet, and Natthanet Tantisuwichwong.
2023. Smart cuisine: Generative recipe & chatgpt powered nutrition assistance
for sustainable cooking. Procedia Computer Science 225 (2023), 2028–2036.
[28] Yuki Kataoka, Sachiko Yamamoto-Kataoka, Ryuhei So, and Toshi A. Furukawa.
2023. Beyond the Pass Mark: Accuracy of ChatGPT and Bing in the National
Medical Licensure Examination in Japan. JMA Journal 6, 4 (16 Oct. 2023), 536–538.
https://doi.org/10.31662/jmaj.2023-0043
[29] Vamsi Krishna Kommineni, Birgitta König-Ries, and Sheeba Samuel. 2024. From
human experts to machines: An LLM supported approach to ontology and knowl-
edge graph construction. arXiv:2403.08345 [cs.CL] https://arxiv.org/abs/2403.
08345
[30] Agnieszka Ławrynowicz, Anna Wróblewska, Weronika T Adrian, Bartosz Kul-
czyński, and Anna Gramza-Michałowska. 2022. Food recipe ingredient substitu-
tion ontology design pattern. Sensors 22, 3 (2022), 1095.
[31] Zhenfeng Lei, Anwar Ul Haq, Adnan Zeb, Md Suzauddola, and Defu Zhang. 2021.
Is the suggested food your desired?: Multi-modal recipe recommendation with
demand-based knowledge graph. Expert Systems with Applications 186 (2021),
115708.
[32] Xianming Li and Jing Li. 2023. AnglE-optimized Text Embeddings. arXiv preprint
arXiv:2309. 71 (2023).
[33] Bingchen Liu and Xin Li. 2025. Large Language Models for Knowledge Graph
Embedding Techniques, Methods, and Challenges: A Survey. arXiv preprint
arXiv:2501.07766 (2025).
[34] Qidong Liu, Xiangyu Zhao, Yuhao Wang, Yejing Wang, Zijian Zhang, Yuqi Sun,
Xiang Li, Maolin Wang, Pengyue Jia, Chong Chen, et al .2024. Large language
model enhanced recommender systems: Taxonomy, trend, application and future.
arXiv preprint arXiv:2412.13432 (2024).
[35] Ya Lu, Thomai Stathopoulou, Maria F Vasiloglou, Lillian F Pinault, Colleen Kiley,
Elias K Spanakis, and Stavroula Mougiakakou. 2020. goFOODTM: an artificial
intelligence system for dietary assessment. Sensors 20, 15 (2020), 4283.
[36] Jinge Ma, Xiaoyan Zhang, Gautham Vinod, Siddeshwar Raghavan, Jiangpeng He,
and Fengqing Zhu. 2024. MFP3D: Monocular Food Portion Estimation Leveraging
3D Point Clouds. arXiv preprint arXiv:2411.10492 (2024).
[37] Priyanka Mandikal and Raymond Mooney. 2024. Sparse Meets Dense: A Hybrid
Approach to Enhance Scientific Document Retrieval. arXiv:2401.04055 [cs.IR]
https://arxiv.org/abs/2401.04055
[38] Javier Marın, Aritro Biswas, Ferda Ofli, Nicholas Hynes, Amaia Salvador, Yusuf
Aytar, Ingmar Weber, and Antonio Torralba. 2021. Recipe1m+: A dataset for
learning cross-modal embeddings for cooking recipes and food images. IEEE
Transactions on Pattern Analysis and Machine Intelligence 43, 1 (2021), 187–203.
[39] Dexon Mckensy-Sambola, Miguel Ángel Rodríguez-García, Francisco García-
Sánchez, and Rafael Valencia-García. 2021. Ontology-based nutritional recom-
mender system. Applied Sciences 12, 1 (2021), 143.
[40] Microsoft Research. 2024. Phi-4: A 14B-parameter language model with 16K
context window. https://huggingface.co/microsoft/phi-4. Released under the
MIT License; arXiv technical report also available.
[41] Weiqing Min, Chunlin Liu, Leyi Xu, and Shuqiang Jiang. 2022. Applications of
knowledge graphs for food science and industry. Patterns 3, 5 (2022).
[42] Mistral AI. 2025. Mistral -Small -3.2-24B-Instruct -2506: minor update improving
instruction-following, reducing repetition, and enhancing function calling. https:
//huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506. Apache2.0
license; knowledge cut-off 2023-10-01; context window 128K tokens.
[43] Paweł Niszczota and Iga Rybicka. 2023. The credibility of dietary advice for-
mulated by ChatGPT: Robo-diets for people with food allergies. Nutrition 112
(2023), 112076.
[44] Zach Nussbaum, John X. Morris, Brandon Duderstadt, and Andriy Mulyar. 2024.
Nomic Embed: Training a Reproducible Long Context (8192) Text Embedder.
https://www.nomic.ai/blog/posts/nomic-embed-text-v1. Open -source model
with open weights, training code, and data; 8192 -token context; Apache 2.0
license.
[45] Ollama Contributors. 2025. Ollama: A lightweight, extensible framework for run-
ning large language models locally. https://github.com/ollama/ollama. GitHub
repo (MITlicense).
[46] Ishita Padhiar, Oshani Seneviratne, Shruthi Chari, Dan Gruen, and Deborah L
McGuinness. 2021. Semantic modeling for food recommendation explanations. In

Abdur Rahman L. et al.
2021 IEEE 37th International Conference on Data Engineering Workshops (ICDEW) .
IEEE, 13–19.
[47] Maria Panagiotou, Ioannis Papathanail, Lubnaa Abdur Rahman, Lorenzo Brigato,
Natalie S Bez, Maria F Vasiloglou, Thomai Stathopoulou, Bastiaan E de Galan,
Ulrik Pedersen-Bjergaard, Klazine van der Horst, et al .2023. A Complete AI-
Based System for Dietary Assessment and Personalized Insulin Adjustment
in Type 1 Diabetes Self-management. In International Conference on Computer
Analysis of Images and Patterns . Springer, 77–86.
[48] Ilias Papastratis, Dimitrios Konstantinidis, Petros Daras, and Kosmas Dimitropou-
los. 2024. AI nutrition recommendation using a deep generative model and
ChatGPT. Scientific Reports 14, 1 (2024), 14620.
[49] Ioannis Papathanail, Lubnaa Abdur Rahman, Lorenzo Brigato, Natalie S Bez,
Maria F Vasiloglou, Klazine van der Horst, and Stavroula Mougiakakou. 2023.
The nutritional content of meal images in free-living conditions—automatic
assessment with goFOODTM. Nutrients 15, 17 (2023), 3835.
[50] Ioannis Papathanail, Jana Brühlmann, Maria F Vasiloglou, Thomai Stathopoulou,
Aristomenis K Exadaktylos, Zeno Stanga, Thomas Münzer, and Stavroula
Mougiakakou. 2021. Evaluation of a novel artificial intelligence system to moni-
tor and assess energy and macronutrient intake in hospitalised older patients.
Nutrients 13, 12 (2021), 4539.
[51] Valentina Ponzo, Ilaria Goitre, Enrica Favaro, Fabio Dario Merlo, Maria Vittoria
Mancino, Sergio Riso, and Simona Bo. 2024. Is Chatgpt an effective tool for
providing dietary advice? Nutrients 16, 4 (2024), 469.
[52] Qwen Team, Alibaba Cloud. 2025. Qwen3: Think Deeper, Act Faster. https:
//qwenlm.github.io/blog/qwen3/. Introduces 8 open -weighted models (dense &
MoE) ranging from 0.6B–235B params, supports 128K context window, hybrid
"thinking" mode, Apache2.0 license.
[53] John W Ratcliff, David E Metzener, et al .1988. Pattern matching: The gestalt
approach. Dr. Dobb’s Journal 13, 7 (1988), 46.
[54] Shaina Raza, Mizanur Rahman, Safiullah Kamawal, Armin Toroghi, Ananya
Raval, Farshad Navah, and Amirmohammad Kazemeini. 2024. A comprehensive
review of recommender systems: Transitioning from theory to practice. arXiv
preprint arXiv:2407.13699 (2024).
[55] Ricardo Rei, Craig Stewart, Ana C Farinha, and Alon Lavie. 2020. COMET: A
neural framework for MT evaluation. arXiv preprint arXiv:2009.09025 (2020).
[56] Nils Reimers, Iryna Gurevych, and the Sentence -Transformers Team. 2020.
all-MiniLM -L6-v2: Efficient sentence embedding model (384 -dim vectors)
fine-tuned on 1 billion pairs. https://huggingface.co/sentence-transformers/all-
MiniLM-L6-v2. Apache2.0 license; trained via contrastive learning on 1B sen-
tence pairs; maps sentences/paragraphs to a 384-dim dense space.
[57] Thomas Rosemann, Andrea Bachofner, and Oliver Strehle. 2024. Kardiovaskuläre
Erkrankungen in der Schweiz – Prävalenz und Versorgung [Cardiovascular
diseases in Switzerland - Prevalence and care]. Praxis (Bern 1994) 113, 3 (March
2024), 57–66.
[58] Konstantinos Rouskas, Mary Guela, Marianna Pantoura, Ioannis Pagkalos, Maria
Hassapidou, Elena Lalama, Andreas FH Pfeiffer, Elise Decorte, Veronique Cor-
nelissen, Saskia Wilson-Barnes, et al .2025. The Influence of an AI-Driven
Personalized Nutrition Program on the Human Gut Microbiome and Its Health
Implications. Nutrients 17, 7 (2025), 1260.
[59] Santiago Ruiz-Rincón and Ixent Galpin. 2024. SnapChef: AI-powered Recipe
Suggestions. (2024).
[60] Fatemeh Sarani Rad, Rasha Hendawi, Xinyi Yang, and Juan Li. 2024. Personalized
Diabetes Management with Digital Twins: A Patient -Centric Knowledge Graph
Approach. Journal of Personalized Medicine 14, 4 (March 2024), 359. https:
//doi.org/10.3390/jpm14040359
[61] Sola S Shirai, Oshani Seneviratne, Minor E Gordon, Ching-Hua Chen, and Debo-
rah L McGuinness. 2021. Identifying ingredient substitutions using a knowledge
graph of food. Frontiers in Artificial Intelligence 3 (2021), 621766.
[62] Beata Sińska and Alicja Kucharska. 2023. Dietary guidelines in diabetes–why
are they so difficult to follow? Pediatric Endocrinology Diabetes and Metabolism
29, 3 (2023), 125–127.
[63] Guangzhi Sun, Xiao Zhan, and Jose Such. 2024. Building better ai agents: A
provocation on the utilisation of persona in llm-based conversational agents. In
Proceedings of the 6th ACM Conference on Conversational User Interfaces . 1–6.
[64] Swiss Federal Commission on Nutrition (NAE). 2025. Nährwertdaten-
banken. https://www.nae.admin.ch/nae/en/home/fachinformationen/
naehrwertdatenbanken.htm. Accessed June 26, 2025.
[65] Swiss Medical Network. n.d.. Diabetology. https://www.swissmedical.net/en/
diabetology Accessed: 2025-06-24.
[66] Chihiro Taguchi, Seiji Maekawa, and Nikita Bhutani. 2025. Efficient Context
Selection for Long-Context QA: No Tuning, No Iteration, Just Adaptive- 𝑘.arXiv
preprint arXiv:2506.08479 (2025).
[67] Katherine Thornton, Kenneth Seals-Nutt, Mika Matsuzaki, and Damion Dooley.
2024. Reuse of the FoodOn ontology in a knowledge base of food composition
data. Semantic Web 15, 4 (2024), 1195–1206.
[68] Maria Tziva, SO Negro, Agni Kalfagianni, and Marko P Hekkert. 2020. Un-
derstanding the protein transition: The rise of plant-based meat substitutes.
Environmental innovation and societal transitions 35 (2020), 217–231.[69] U.S. Department of Agriculture, Agricultural Research Service. 2019. FoodData
Central. https://fdc.nal.usda.gov/. Launched April 2019; metadata updated April
21,2025.
[70] Francesco Vitali, Rosario Lombardo, Damariz Rivero, Fulvio Mattivi, Pietro
Franceschi, Alessandra Bordoni, Alessia Trimigno, Francesco Capozzi, Giovanni
Felici, Francesco Taglino, et al .2018. ONS: an ontology for a standardized descrip-
tion of interventions and observational studies in nutrition. Genes & nutrition 13
(2018), 1–9.
[71] Kun Wang, Guibin Zhang, Zhenhong Zhou, Jiahao Wu, Miao Yu, Shiqian Zhao,
Chenlong Yin, Jinhu Fu, Yibo Yan, Hanjun Luo, et al .2025. A comprehensive
survey in llm (-agent) full stack safety: Data, training and deployment. arXiv
preprint arXiv:2504.15585 14, 8 (2025).
[72] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi,
Quoc V Le, Denny Zhou, et al .2022. Chain-of-thought prompting elicits reason-
ing in large language models. Advances in neural information processing systems
35 (2022), 24824–24837.
[73] World Health Organization. 2024. Noncommunicable diseases. https://www.
who.int/news-room/fact-sheets/detail/noncommunicable-diseases Fact sheet,
last updated 23 December 2024.
[74] World Health Organization, Western Pacific Regional Office. 2021. Dietary
Restrictions Infographic. Technical poster (digital), https://www.who.int/docs/
default-source/wpro---documents/posters/food-safety/dietary-restrictions-
infographic---digital---english-wpro.pdf?sfvrsn=5b52bc9f_3. Accessed:
2025-07-09.
[75] Haoyu Zhang, Jun Liu, Zhenhua Zhu, Shulin Zeng, Maojia Sheng, Tao Yang,
Guohao Dai, and Yu Wang. 2024. Efficient and Effective Retrieval of Dense-Sparse
Hybrid Vectors using Graph-based Approximate Nearest Neighbor Search. arXiv
preprint arXiv:2410.20381 (2024).
[76] Jian Zhang, Bifan Wei, Shihao Qi, haiping Zhu, Jun Liu, and Qika Lin. 2025. GKG-
LLM: A Unified Framework for Generalized Knowledge Graph Construction.
arXiv:2503.11227 [cs.AI] https://arxiv.org/abs/2503.11227
[77] Lingfeng Zhong, Jia Wu, Qian Li, Hao Peng, and Xindong Wu. 2023. A Compre-
hensive Survey on Automatic Knowledge Graph Construction. ACM Comput.
Surv. 56, 4, Article 94 (Nov. 2023), 62 pages. https://doi.org/10.1145/3618295
[78] Pengfei Zhou, Weiqing Min, Chaoran Fu, Ying Jin, Mingyu Huang, Xiangyang
Li, Shuhuan Mei, and Shuqiang Jiang. 2025. FoodSky: A food-oriented large
language model that can pass the chef and dietetic examinations. Patterns 6, 5
(2025), 101234. https://doi.org/10.1016/j.patter.2025.101234
[79] Samuel Zumthurm, Ioannis Papathanail, Lubnaa Abdur Rahman, Lorenzo Brigato,
Stavroula Mougiakakou, and Aline Stämpfli. 2025. Reducing meat consumption
using a diet-related written prompt and the Swiss food pyramid: A field study.
Food Quality and Preference 126 (2025), 105416.