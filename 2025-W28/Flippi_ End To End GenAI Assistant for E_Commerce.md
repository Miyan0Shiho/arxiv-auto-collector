# Flippi: End To End GenAI Assistant for E-Commerce

**Authors**: Anand A. Rajasekar, Praveen Tangarajan, Anjali Nainani, Amogh Batwal, Vinay Rao Dandin, Anusua Trivedi, Ozan Ersoy

**Published**: 2025-07-08 08:50:47

**PDF URL**: [http://arxiv.org/pdf/2507.05788v1](http://arxiv.org/pdf/2507.05788v1)

## Abstract
The emergence of conversational assistants has fundamentally reshaped user
interactions with digital platforms. This paper introduces Flippi-a
cutting-edge, end-to-end conversational assistant powered by large language
models (LLMs) and tailored for the e-commerce sector. Flippi addresses the
challenges posed by the vast and often overwhelming product landscape, enabling
customers to discover products more efficiently through natural language
dialogue. By accommodating both objective and subjective user requirements,
Flippi delivers a personalized shopping experience that surpasses traditional
search methods. This paper details how Flippi interprets customer queries to
provide precise product information, leveraging advanced NLP techniques such as
Query Reformulation, Intent Detection, Retrieval-Augmented Generation (RAG),
Named Entity Recognition (NER), and Context Reduction. Flippi's unique
capability to identify and present the most attractive offers on an e-commerce
site is also explored, demonstrating how it empowers users to make
cost-effective decisions. Additionally, the paper discusses Flippi's
comparative analysis features, which help users make informed choices by
contrasting product features, prices, and other relevant attributes. The
system's robust architecture is outlined, emphasizing its adaptability for
integration across various e-commerce platforms and the technological choices
underpinning its performance and accuracy. Finally, a comprehensive evaluation
framework is presented, covering performance metrics, user satisfaction, and
the impact on customer engagement and conversion rates. By bridging the
convenience of online shopping with the personalized assistance traditionally
found in physical stores, Flippi sets a new standard for customer satisfaction
and engagement in the digital marketplace.

## Full Text


<!-- PDF content starts -->

Flippi: End To End GenAI Assistant for E-Commerce
Anand A. Rajasekar*
Praveen Tangarajan*
Anjali Nainani
Amogh Batwal
Vinay Rao Dandin
Anusua Trivedi
Ozan Ersoy
Flipkart US R&D Center
Bellevue, Washington, USA
ABSTRACT
The emergence of conversational assistants has fundamentally re-
shaped user interactions with digital platforms. This paper intro-
duces Flippi-a cutting-edge, end-to-end conversational assistant
powered by large language models (LLMs) and tailored for the e-
commerce sector. Flippi addresses the challenges posed by the vast
and often overwhelming product landscape, enabling customers
to discover products more efficiently through natural language
dialogue. By accommodating both objective and subjective user re-
quirements, Flippi delivers a personalized shopping experience that
surpasses traditional search methods. This paper details how Flippi
interprets customer queries to provide precise product information,
leveraging advanced NLP techniques such as Query Reformulation,
Intent Detection, Retrieval-Augmented Generation (RAG), Named
Entity Recognition (NER), and Context Reduction. Flippi’s unique
capability to identify and present the most attractive offers on an
e-commerce site is also explored, demonstrating how it empow-
ers users to make cost-effective decisions. Additionally, the paper
discusses Flippi’s comparative analysis features, which help users
make informed choices by contrasting product features, prices, and
other relevant attributes. The system’s robust architecture is out-
lined, emphasizing its adaptability for integration across various
e-commerce platforms and the technological choices underpinning
its performance and accuracy. Finally, a comprehensive evaluation
framework is presented, covering performance metrics, user satis-
faction, and the impact on customer engagement and conversion
rates. By bridging the convenience of online shopping with the per-
sonalized assistance traditionally found in physical stores, Flippi
sets a new standard for customer satisfaction and engagement in
the digital marketplace.
1 INTRODUCTION
Conversational commerce has rapidly evolved, enabling consumers
to shop, purchase, and seek assistance through interactive plat-
forms such as live chat, voice assistants, and messaging apps. These
conversational tools are pivotal in e-commerce, guiding customers
to discover suitable products, make informed purchasing decisions,
and access post-purchase support.
Historically, online shopping has been dominated by search bar-
driven navigation, where users begin with broad queries-such as
product type and basic requirements like brand or features. How-
ever, traditional platforms often present significant challenges thatdetract from the shopping experience. Chief among these is choice
overload: the sheer volume of available products can make it diffi-
cult for users to find what they want. Compounding this are search
algorithms that sometimes misinterpret queries, leading to irrel-
evant results. Filtering and sorting tools may lack the necessary
granularity or intuitiveness, making it hard for users to efficiently
narrow their options.
A further challenge is the lack of personalized recommendations,
which are increasingly essential for aligning product suggestions
with individual tastes and preferences. Static product displays fail
to deliver the interactive, engaging experiences that users now ex-
pect, often resulting in decision fatigue. Product pages are typically
dense with information-ranging from seller-provided descriptions
to customer reviews-making it difficult for users to extract relevant
details, especially on mobile devices with limited screen space. For
value-driven customers seeking the best offers, the decision-making
process becomes even more complex. Thus, developing a compre-
hensive conversational assistant that supports users throughout
their e-commerce journey-from product discovery to customer
support-is both necessary and impactful.However, building such
a system presents unique challenges. Multi-turn conversations re-
quire the assistant to understand and retain the context of previous
interactions, ensuring a natural conversational flow and preventing
repetitive exchanges. The ability to ask targeted follow-up ques-
tions is also crucial for a complete conversational experience.
Technical integration with existing e-commerce systems is an-
other hurdle. The architecture must be streamlined to interact seam-
lessly with various components, such as search engines, product
detail fetchers, and customer support and experience (CX) systems.
The complexity of integrating with diverse engineering elements
can be substantial.
This paper presents an end-to-end data science architecture for
modeling e-commerce assistants. We detail the current architec-
ture, the design decisions behind it, and the individual modules
and their evaluations. The paper is organized as follows: Section 2
reviews related work; Section 3 outlines the proposed framework;
Section 4 discusses the evaluation methodology; Section 5 presents
results; Section 6 examines Flippi’s production impact; and Section
7 concludes with future directionsarXiv:2507.05788v1  [cs.CL]  8 Jul 2025

KDD2025LLM4ECommerce, August 2025, Toronto, Canada Praveen and Anand, et al.
2 RELATED WORK
The rapid advancement of natural language processing (NLP) and
machine learning has driven remarkable growth in conversational
agents and chatbots, particularly within e-commerce [Lim et al .
2022; Mariani et al .2023]. Early e-commerce applications leveraged
conversational interfaces to improve shopping experiences, such
as chatbot-based recommenders that utilized user preferences and
browsing history for personalized suggestions [Liu et al .2019], and
dialogue generation models that enabled more intuitive product
discovery [Zhang et al .2022]. Further studies have explored the
influence of chatbot language style on user trust [Cheng et al .2022],
negotiation capabilities in conversational commerce [Priyanka et al .
2023], and factors affecting customer engagement among younger
demographics [Rumagit et al. 2023].
The emergence of Large Language Models (LLMs) has marked
a paradigm shift in NLP, enabling significant progress in knowl-
edge acquisition and generation [Minaee et al .2024; Naveed et al .
2024; Zhao et al .2023]. Trained on extensive corpora using self-
supervised objectives [Gui et al .2023], models such as BERT [De-
vlin et al .2018], GPT-3.5 [Brown et al .2020; Sanh et al .2021],
GPT-4 [OpenAI et al .2024], LLaMA [Touvron et al .2023], and Mis-
tral [Jiang et al .2023] have advanced the generation of human-like
text and deepened linguistic understanding.
Despite their strengths, adapting LLMs to specific domains and
user preferences remains challenging [Havrilla et al .2024; Kirk et al .
2024]. Approaches such as domain-specific fine-tuning [Hu et al .
2021; Xu et al .2023] and context-aware prompting [Liu et al .2021;
Sahoo et al .2024] have been proposed, though fine-tuning can be
resource-intensive. Retrieval Augmented Generation (RAG) offers
a scalable alternative, augmenting LLMs with external knowledge
to improve factuality and reduce hallucinations [Gao et al .2024;
Lewis et al .2020; Siriwardhana et al .2022]. Advanced prompt engi-
neering techniques, further enhances LLM reasoning within RAG
frameworks [Chu et al. 2023; Wei et al. 2023].
Evaluating LLM-driven systems requires more than traditional
NLP metrics, prompting the development of frameworks that assess
qualitative aspects such as bias, toxicity, factual consistency, and
hallucination rates [Chang et al .2023; Guo et al .2023; Ribeiro et al .
2020; Zheng et al .2023]. Human-in-the-loop evaluation remains
essential for judging the quality and utility of responses in complex
interactive systems [Kiela et al .2021; Lewis et al .2020; Schiller
2024].
Recent adoption of generative AI has led to the development
of advanced e-commerce assistants, incorporating features like
conversational product discovery, personalized recommendations,
product comparisons, and support for order-related queries. These
systems leverage large product catalogs, customer reviews, and
community Q&A to create interactive shopping experiences. How-
ever, challenges persist in managing hallucinations, maintaining
context over multi-turn dialogues, and leveraging structured prod-
uct data for accurate responses. Achieving deep personalization
and robust long-term context retention remains difficult for many
solutions.
Building on these foundations, this paper introduces Flippi, an
end-to-end generative AI assistant for e-commerce. Flippi integrates
advanced NLP techniques—Query Reformulation, Intent Detection,RAG, Named Entity Recognition, and Context Reduction—within
a modular, scalable architecture. By addressing key challenges in
product discovery, personalization, offer identification, and product
comparison, Flippi provides a comprehensive framework for reli-
able, user-centric conversational assistance in digital marketplaces.
3 PROPOSED ARCHITECTURE
The interaction between the customer and the assistant begins
when the customer inputs their question into the assistant’s chat
window, thereby starting a session. To facilitate a continuous and
coherent conversation, each user query, along with the complete
session context (previous turns of dialogue), is forwarded to the
Standalone Query (SAQ) module. The primary function of the SAQ
module is to leverage the full session history and the current user
query to generate a contextually independent query that accurately
reflects the customer’s complete and most recent intent. This re-
formulated, standalone query is subsequently directed to a coarse
intent classification model, which determines the appropriate pro-
cessing pipeline, or "flow," for the query. The Flippi framework is
designed around several core flows, primarily including Product
Search, Decision Assistance (for product-specific Q&A), Offers (for
promotion-related inquiries), and Customer Experience (CX, for
post-purchase and customer service issues), with the architecture
allowing for scalability to additional use cases.
If the coarse intent model identifies the user’s intent as requir-
ing a direct response (e.g., handling chitchat, general e-commerce
platform FAQs), decision assistance, or product comparison, the
ArgsLLM module is activated. The main functionality of this mod-
ule is to gather all necessary arguments required for the Decision
Assistance and Product Comparison flows. Additionally, it is capa-
ble of directly generating a response to the user’s query for certain
intents. The overall process is illustrated in Figure 1. At a high
level: the Search flow aims to retrieve relevant product sets based
on customer interest. The Decision Assistance (DA) flow answers
product-related questions for a specific item using catalog infor-
mation and user-generated content (UGC) like reviews. The Offers
flow addresses queries about current or upcoming promotions and
discounts. The CX flow manages customer service inquiries re-
lated to post-purchase events. Finally, other ancillary flows provide
templated responses for queries outside the e-commerce scope or
answer general FAQs pertaining to the e-commerce platform.
3.1 Standalone Query Module (SAQ)
The SAQ module is the first and one of the most crucial compo-
nents of the Flippi pipeline. It facilitates continuous conversation
between user and assistant. The SAQ module receives each user’s
query along with the complete session context. The session context
includes all the previous queries and responses exchanged between
the user and the assistant. The primary function of the SAQ mod-
ule is to generate an independent query that accurately reflects
the user’s current intent while incorporating the important and
relevant information from the context. This revised query should
encapsulate all the necessary information, allowing the conversa-
tional agent to provide a more targeted and relevant response.
Another function of this component is to perform product disam-
biguation. Product disambiguation is the process of identifying the

Flippi: End To End GenAI Assistant for E-Commerce KDD2025LLM4ECommerce, August 2025, Toronto, Canada
Figure 1: High level architecture
Sample 1 Sample 2
ConversationUser: oppo mobile
Bot: Products that match your query are
OPPO A78 5G (Glowing Black), OPPO A78 (Aqua Green)
and OPPO A38 (Glowing Gold)
Bot: What budget do you have in mind?
User: 15,000
Bot: Products that match your query are
OPPO A38 (Glowing Gold), OPPO A17 (Lake Blue)
and OPPO A58 (Dazzling Green)
Bot: Any particular feature you are looking for in a phone?
User: Motorola mobileUser: PETER ENGLAND suit for wedding under 6000
Bot: Products that match your query are
PETER ENGLAND SLIM FIT Solid Suit, Raymond Men
Solid Suit and VAN HEUSEN Men 2 PC Suit Solid Suit]
Bot: Do you have a specific size in mind?
User: No, what is the color of VAN HEUSEN suit?
SAQ Motorola mobile within 15,000 What is the color of VAN HEUSEN Men 2 PC Suit Solid
Suit?
Table 1: Example user conversations with standalone queries
correct product that the user is referring to when there are multiple
products in the context. The SAQ module uses the context of the
conversation to disambiguate the user’s request and add the cor-
rect product name to the reformulated query. E.g., if the user says
"second product", the SAQ module will identify the right product
from the bot output and add the product name to the final query.
Table 1 consists of example conversations and the corresponding
standalone queries. Currently, this component is powered by an
LLM call.
3.2 Coarse Intent Model
The primary functionality of this model is to understand the intent
behind the query and trigger the appropriate flow to handle the
user request. Currently, this model takes the given SAQ query asinput and predicts one of the eight distinct intents which then is
leveraged as a routing mechanism for downstream flows.
Here are the following eight intents and their definitions:
(1)answer_product_specific_questions : This intent pertains
to queries that revolve around detailed information about a
product such as its specifications, warranty, payments and
availability. The Decision Assistant flow accomplishes this
task by utilizing data from both the product catalog and user-
generated content, such as customer reviews. For instance,
any question about a product’s features or functionality will
be addressed through this intent.
(2)search_for_products : This intent caters queries aiming to
find or explore products on the e-commerce platform. By
calling upon a search API, the intent can further clarify

KDD2025LLM4ECommerce, August 2025, Toronto, Canada Praveen and Anand, et al.
the user’s needs by posing follow-up questions to better
understand their requirements.
(3)compare_products : This intent is designed to handle queries
that involve comparing two or more products or their fea-
tures. For example, a query like “RAM of iPhone 14 vs Sam-
sung S23” would be directed to this intent, enabling the user
to make a more informed decision about the product they
wish to purchase.
(4)return_direct_response : This intent deals with queries that
are either greetings, casual conversations or questions seek-
ing general information about product features. These queries
may include questions such as “What is an OLED display”,
“What is the difference between a processor and RAM” or
“What is an operating system”. The main focus of these
queries is to provide general information related to shopping.
(5)answer_offer_related_questions : This intent handles queries
concerning current or upcoming offers. This includes general
offers, big sale events, and promotions on specific products.
Queries related to discounts and other promotional activities
are directed here.
(6)post_purchase : This intent handles customer queries that
occur after a purchase has been completed. These may in-
clude issues with orders, tracking delivery, refund, or can-
cellation of a product, and other post-purchase concerns.
Queries of this nature are typically directed to the customer
service department
(7)get_answer_from_faq : This intent addresses commonly
asked, non-product specific questions that are specific to
ecommerce platform. These may include generic questions
about payment methods, return policies, exchange policies,
or pre-purchase queries. Examples include “How to order?”,
“Where can I find the refund policy” or “How to apply an
offer code?”.
(8)not_applicable : This category includes any queries that fall
outside the scope of online shopping or do not fit within the
categories mentioned above. Queries that are not related to
general conversation, shopping, or product-specific queries
are categorized as not applicable. Examples may include
"how to calculate the area of a circle” etc.
Different BERT based architectures were tried for this text classi-
fication and we were able to achieve high classification accuracy
of around 95%. Given high traffic requirements, we preferred the
smaller and low latency DistillBERT model. Data augmentation
and MixUp approaches were tried to further improve the model.
This model was improved by data augmentation techniques such as
continuous sampling using active learning and relabeling by LLMs.
Results are discussed in Section 4.
3.3 Product Search and Followup
Product search and discovery are core use cases for Flippi. This flow
incorporates multiple components and leverages the e-commerce
platform’s existing search infrastructure. A store classifier, fine-
tuned on production queries, is used to identify the relevant product
category or store. This flow also includes enhancements beyond
standard search to better interpret subjective queries and handle
queries that yield no initial results. First, the SAQ output undergoesa ’Query Cleanup’ step to normalize it for the platform’s search
service. The cleaned query is then passed to the search service,
which returns a list of products (e.g., up to 24 products, with a
subset, such as 8, displayed initially with a ’View More’ option). As
customers are presented with products pertaining to their initial
query, an accompanying question related to the product is posed
to gain a more comprehensive understanding of their preferences.
Generating effective follow-up questions for the assistant ad-
heres to several key principles:
(1)Relevance to the initial query and current context:
Follow-up questions and their suggested values should be
directly pertinent to the user’s stated needs and the ongo-
ing conversation. Example: If a user queries "running shoes
for wide feet under $80", follow-up questions should focus on
relevant attributes like "preferred cushioning level" or "trail vs.
road running", rather than suggesting high-end racing flats or
features common in shoes well over the specified budget.
(2)Grounding in available product assortment: Proposed
values or features in follow-up questions must correspond to
actual products or characteristics available on the e-commerce
platform. Example: When discussing "cotton t-shirts", if the
platform primarily stocks common colors, the follow-up should
not suggest "artisanal hand-dyed indigo" as a color option if
such products are not offered.
(3)Avoidance of repetition: The same follow-up question or
attribute type should generally not be repeated within the
same conversational thread for a given search, especially if
the user has already declined or ignored a similar prompt.
Example: If the assistant previously asked "Are you looking for
a specific brand?" and the user responded "No, show me all op-
tions", it should avoid asking about brands again immediately
unless the search context significantly changes.
(4)Adaptability to context shifts: The assistant must recog-
nize and adapt to changes in the user’s focus or product inter-
est within a single session. Example: If a user initially searches
for "digital cameras" and later asks "what about tripods for
these models?", subsequent follow-up questions should shift
to tripod-specific attributes (e.g., "material preference for the
tripod?", "maximum height needed?") rather than persisting
with camera features.
(5)Conciseness in interaction: The number of follow-up ques-
tions posed before providing or refining results should be
limited to prevent user fatigue and streamline the discovery
process. Example: After an initial set of products is displayed,
posing more than two or three clarifying questions before show-
ing updated results is generally advisable to maintain engage-
ment.
For generating these follow-up questions, a context-aware prompt
was developed for an LLM to autonomously suggest relevant at-
tributes. This prompt aims to generate pertinent follow-up ques-
tions with appropriate value suggestions, while avoiding repetition.
However, grounding these suggestions to the e-commerce plat-
form’s actual inventory requires integrating outputs from a service
providing popular product facets (features and their common val-
ues). This approach faces challenges: the facets service may not

Flippi: End To End GenAI Assistant for E-Commerce KDD2025LLM4ECommerce, August 2025, Toronto, Canada
provide features for all product categories, or the range of values
for a feature might be limited. Additionally, the LLM might some-
times struggle to generate relevant suggestions even when provided
with features and values (e.g., suggesting a premium brand for a
low-budget query). These challenges may be mitigated by improve-
ments to the facets service and by fine-tuning smaller, specialized
language models for this task.
3.4 ArgsLLM
ArgsLLM is a LLM-powered component responsible for extracting
and resolving complete product names from the Standalone Query
(SAQ) and conversation context. These extracted names are used by
downstream pipelines to retrieve relevant product information and
answer customer queries. ArgsLLM also handles customer queries
that do not have a shopping intent, making it essential for both
product-specific and general interactions.
This component is triggered when the detected coarse intent is one
of the following, with ArgsLLM behaving as described:
•answer_product_specific_questions :ArgsLLM extracts
the complete product name from SAQ, using the detected
intent, the query itself, and a list of up to eight suggested
products shown in the previous turn. Since customers rarely
mention full product names, these suggestions are crucial
for accurate extraction.
•compare_products :ArgsLLM identifies all product names
referenced in the comparison query, again relying on the
suggested product list to resolve incomplete names.
•return_direct_response :ArgsLLM generates brief, courte-
ous, and neutral replies for chitchat, or general knowledge
queries, ensuring continued customer engagement.
If no products were suggested in the previous turn, or if the product
of interest is missing from the suggested list, ArgsLLM extracts and
returns the product name directly from the SAQ. If no product name
is present in the query, it returns an empty response, which further
triggers the follow-up module to prompt the user for the relevant
product name. This design ensures robust handling of diverse user
intents and query scenarios.
3.5 Decision Assistant (DA)
The main function of this flow is to answer user queries regarding
a specific product. The queries can be on specifications, warranty,
availability and other product related queries. In general these
queries can be answered from either product catalog or user gener-
ated data like customer reviews, seller FAQs etc. Once the coarse
intent model predicts the intent as DA, then the query is passed to
ArgsLLM which gets the exact product name either based on the
query and list of products returned from the latest search query
in the session. Given the query and product_id, the user query is
then passed to the DA intent model that decides if the query can be
answered from the product catalog data, UGC data or other sources.
Once the DA flow is triggered all the subsequent queries are directly
routed to DA flow until the DA intent model predicts the query as
discovery or other main intents.
In situations where the DA intent model interprets the user’s in-
tent as ’answer_product_specific_questions’ , the model will retrieve
Figure 2: DA architecture
all the product specifications from the product catalog. For instance,
if the product in question is a smartphone, the specifications might
include the type of processor it uses (e.g., Snapdragon). From this
pool of specifications, the top 20 most relevant specs are selected,
based on their potential to answer the user’s query. These chosen
specifications, together with the user’s query, serve as the con-
text for a generative model (specifically and LLM) to formulate
a fitting response to the user’s query. If the generative model is
unable to provide an answer, a template response is given, and
a ’not_answerable’ flag is returned. This triggers the UGC flows,
which involve selecting and presenting the top 3 FAQs and reviews
that are most likely to address the user’s query.
The selection of the most pertinent reviews and FAQs is facili-
tated by a bi-encoder-based semantic text similarity (STS) model.
This model, trained using contrastive loss on proprietary e-commerce
UGC FAQ data, is designed to assess the semantic similarity be-
tween texts. The same model, adapted for this specific domain, also
functions as a context reducer, picking out the relevant specifica-
tions from the comprehensive list. This method of context reduction
has proven to be nearly as effective as the fully supervised Dense
Retriever (DPR) model. An added benefit is that it does not require
retraining for different product categories, as the STS model is
trained on data from a wide variety of categories.
3.6 Summarization
Summarization is essential for distilling complex product descrip-
tions and user queries into concise, easily understood summaries,

KDD2025LLM4ECommerce, August 2025, Toronto, Canada Praveen and Anand, et al.
thereby enhancing user comprehension and decision-making. By
focusing on relevant information, summarization reduces cognitive
load and improves the overall user experience.
This component takes two inputs: the Standalone Query (SAQ)
and detailed product information (including product name, features,
and customer reviews). Using an LLM, we generate a summary
tailored to the contextualized user query, ensuring coverage of
product highlights and relevant pros and cons.
To ensure summaries are both concise and highly relevant to user
queries, the summarization process employs targeted guidelines.
Key product attributes are prioritized. A comprehensive summary is
then constructed, addressing all pertinent objective and subjective
facets of the product in relation to the query, while actively min-
imizing redundancy. This involves presenting positive or neutral
aspects in a few descriptive sentences (e.g., up to three) to guide
the user, alongside a brief mention of any relevant negative points
(e.g., up to two) if identified in the source material. The overarching
goal is to produce engaging, user-centric summaries that empower
effective decision-making.
3.7 Compare
This flow is designed to assist users in comparing the features of
two or more products. Given a compare query for example "which
phone has a better camera?", we get the products retrieved by the
latest search query in the current session and fetch their product
details. We then apply context reduction to select top 20 most
relevant feature required to address the query for each product and
send this data to the LLM prompting it to generate a comparison
summary and final verdict for the query asked. The comparison
summary here generates descriptive summary with the pros and
cons of the feature(s) that the customer is interested in. The verdict
is a single sentence based on the comparison summary to help the
customer decide which product is better.
3.8 CX/Post Purchase
For post-purchase inquiries, the current system architecture primar-
ily directs users to the e-commerce platform’s established customer
support channels. This typically involves guiding the customer
to access their order history and initiate a support interaction re-
garding a specific purchase through the platform’s existing help
interfaces or dedicated support assistants.
3.9 Other Flows
In case user queries are not related to shopping context or if its
regarding general FAQs on the e-commerce portal we have a set
of flows which either provide pre-designed responses or answers
these queries based on the FAQs.
4 EVALUATION METHODS AND METRICS
One of the challenges inherent in conversational chat bot systems
revolves around the evaluation of conversation quality with cus-
tomers. In response to this challenge, we have devised a metric i.e.
Answerability [Gupta et al .2022] which evaluates our assistant’s
response quality at each turn of conversation and finally computes
a metric based on the whole session.For models which are part of the pipeline, specific metrics are
allocated to comprehensively evaluate its performance over time.
These metrics serve as benchmarks to assess the effectiveness and
efficiency of each model’s functioning throughout its operational
duration within the pipeline.
4.1 SAQ
We track two metrics to evaluate the performance of SAQ, Query
Modification Accuracy (QMA) and Product disambiguation Ac-
curacy (PDA). The former metric deals with correctness of the
generated query, in terms of retaining relevant features and not hal-
lucinating unnecessary information. The second metric evaluates
the correctness of the generated product name when user query
warrants product disambiguation. In order to evaluate QMA, we
follow two approaches:
•LLM based - We developed a GPT-4 prompt to evaluate the
accuracy of the reformulated query in terms of correct in-
tent, budget, features and product category given the entire
conversation history,
•Human based - We share the output of the top perform-
ing prompts on the evaluation dataset to human labelers to
estimate the accuracy.
We also evaluate the accuracy of product disambiguation with the
help of human labelers.
4.2 Coarse Intent Model
Since this is a classification model, we track traditional classification
metrics like model accuracy, F1-score, Precision and Recall for
evaluations.
4.3 Summarization
The summarization component of our chatbot is evaluated by as-
sessing the quality of product summaries generated in response to
user queries. To ensure a comprehensive assessment, we employ
both human and tune LLM to match the perform of two main steps:
(1)Product Relevancy: The evaluator determines whether the
product described in the summary matches the user’s query.
If the summary addresses the correct product, the “Product
Relevancy” label is marked as True; otherwise, it is marked
as False. For example, a summary about a refrigerator in
response to a smartphone query would be marked as False.
(2)Summary Quality: The summary is further assessed on
the following criteria:
(a)Factual Accuracy: The accuracy of product details in the
summary is checked. If all facts are correct, the label is
Pass; otherwise, it is Fail.
(b)Query Relevance: The degree to which the summary ad-
dresses the user’s query is rated as Fully Relevant, Partially
Relevant, or Irrelevant. Partially Relevant and Irrelevant
summaries are further analyzed to identify specific gaps.
This structured approach enables us to systematically identify
strengths and areas for improvement in the summarization compo-
nent.

Flippi: End To End GenAI Assistant for E-Commerce KDD2025LLM4ECommerce, August 2025, Toronto, Canada
4.4 Compare
The comparison evaluation is conducted using a large language
model (GPT-4), which is provided with the user query, product
context and specifications, the generated comparison summary,
and a verdict. All key aspects are extracted from both the user
query and the comparison summary. Each aspect is then evaluated
according to two main criteria:
(1)Relevancy: Each aspect in the comparison summary is as-
sessed for its connection to the user’s query. An aspect is
labeled as:
•Relevant if it is directly mentioned in the query,
•Partially Relevant if it can be reasonably inferred from the
query, or
•Irrelevant if it is not present or implied in the query.
(2)Correctness: This criterion is divided into two parts:
•Verdict Correctness: Evaluates whether the verdict accu-
rately compares the specified aspect in the context of the
user query. The verdict is labeled as Correct if the compar-
ison is accurate, Incorrect if not, or N/Aif the aspect is not
addressed.
•Comparison Correctness: Assesses whether the comparison
summary appropriately addresses or compares the speci-
fied aspect, considering the full context. This is labeled as
either Correct orIncorrect .
This structured approach ensures a rigorous and transparent
evaluation of both the relevance and accuracy of each aspect within
the comparison component.
4.5 ArgsLLM
The ArgsLLM component was evaluated through a two-stage pro-
cess involving both human and LLM-based annotations. Initially,
human annotators and a larger LLM independently labeled model re-
sponses using the Standalone Query (SAQ), coarse intent, suggested
product list, and generated output. Each response was marked as
either good orbad, with corrections provided for badresponses.
Human annotators further categorized cases based on the presence
and matching of product names in the query and content.
To scale and standardize evaluation, the evaluator LLM was sub-
sequently tuned—leveraging advanced prompting strategies such
as chain-of-thought reasoning—to closely match human annotation
quality. Once aligned, the tuned evaluator LLM was systematically
employed to assess model responses and provide detailed feed-
back, enabling efficient identification of failure points and targeted
improvements for the ArgsLLM generation models.
For each coarse intent, clear labeling guidelines were followed.
Foranswer_product_specific_questions , a response was labeled good
if it correctly handled product names as indicated by the query
and context; otherwise, it was labeled bad. For compare_products ,
responses were labeled good if all relevant product names were
accurately reflected, partially good if only some were included,
andbadif key names were missing. For return_direct_response , re-
sponses were labeled good if they were succinct, polite, inoffensive,
and neutral.
This iterative evaluation framework, combining human expertise
and advanced LLM-based assessment, enabled scalable, high-qualityevaluation and systematic enhancement of ArgsLLM’s generative
capabilities.
5 EXPERIMENTS AND RESULTS
5.1 SAQ
SAQ results obtained from LLM and human based evaluations are
tabulated in Table 2. Our LLM based approach is strict and penalizes
the module even for small mistakes. However, we are able to exper-
iment and evaluate with a shorter turnaround time compared to
human labeling. We chose the top performing prompts and perform
human evaluation to obtain both QMA and PDA. The prompt with
the highest QMA is chosen for production.
Table 3 contains the turn wise accuracy of SAQ module. As the
turn count increases, the accuracy of SAQ drops significantly. This
is expected since the conversation gets more complex as the user
converse for more turns. Only 449 sessions out of 4000 i.e ≈10%of
the sessions contain 5 turns.
QMA PDA
LLM based 0.8949 -
Human based 0.9234 0.9523
Table 2: LLM and Human based evaluation
Turn QMA No. of datapoints
1 0.9865 4000
2 0.9057 1675
3 0.873 1063
4 0.8559 673
5 0.8396 449
Table 3: Turn wise accuracy of SAQ module with support
5.2 Coarse Intent Model
Table 4 presents the accuracy metrics for various BERT base archi-
tectures tested on our dataset. We opted for the DistillBERT model
due to its compatibility with our traffic and latency requirements.
Model Test Accuracy
FK-Bert 94.70%
FK-Bert + Mixup + Aug 94.69%
BERT 94.43%
DistillBert 94.89%
DistillBert + Mixup + Aug 94.75%
Table 4: Test accuracy of different models
Over time, we enhanced the model with the use of Active Learn-
ing and data adjustments. The final accuracy of the coarse intent
model on our test dataset is approximately 97%. The accuracy de-
tails for the detailed intent model can be found in Table 5.
5.3 Summarization, Compare & ArgsLLM
Our summarization module demonstrated strong performance,
achieving an overall factuality of 88.47% and a combined full and
partial relevancy of 94.27% based on human annotations. Product
relevancy for summarization was rated at 97.05% by human eval-
uators. For the compare module, GPT-4 evaluation yielded mean
scores of 94.16% for relevancy, 89.44% for comparison correctness,

KDD2025LLM4ECommerce, August 2025, Toronto, Canada Praveen and Anand, et al.
Intent Precision Recall F1-Score
answer_offer_related_questions 0.9664 0.9839 0.9751
answer_product_specific_questions 0.4545 0.8333 0.5882
compare_products 0.5652 0.9455 0.7075
get_answer_from_faq 0.6875 0.7529 0.7187
not_applicable 0.7194 0.7542 0.7364
post_purchase 0.9895 0.9759 0.9827
return_direct_response 0.8557 0.8706 0.8631
search_for_products 0.9962 0.9867 0.9914
accuracy 0.9734
macro avg 0.7793 0.8879 0.8204
weighted avg 0.9771 0.9734 0.9748
Table 5: Intent Accuracy
and88.18% for verdict correctness. The ArgsLLM component was
rated as good in90.24% of cases by human annotators, while GPT-4
evaluation reported a good response rate of 92.33% .
5.4 DA-QnA
Table 6 presents the accuracy of the DA question answering model
across seven distinct Business Units. The results clearly shows
that the model consistently maintains an average accuracy rate of
approximately 90% or higher, demonstrating its effectiveness.
BU/Category Helpful Answer Rate
Mobiles 89.65%
Clothing 88.7%
Large 94.2%
BGM 90.2%
Home 90.9%
Furniture 88.1%
Electronics 91.3%
Table 6: DA-QnA Metrics
5.5 Answerability Evaluation
Ground truth labels for the Answerability metric are obtained
through human annotators and LLMs are also leveraged for as-
sessing the bot’s responses against user queries. Answerability is
assessed at both turn and session levels, focusing on relevancy
(Highly relevant, Partially relevant, Irrelevant) at the turn level,
and overall successfullness (Successful, Unsuccessful) according to
predefined labeling guidelines. Turnwise labels are used to calculate
turn-level and session-level Answerability metrics. Session-level
answerability demonstrated a consistent improvement of approxi-
mately 32% over the evaluation period
6 IMPACT
The Flippi system was first deployed through an A/B experimenta-
tion framework. Since its initial deployment, the system has under-
gone numerous iterative development cycles, with each iteration
yielding enhancements over its predecessor. During a major e-
commerce sales event, Flippi was exposed to the entire user base
through specific platform integrations, and concurrently A/B tested
on primary user interfaces such as Search, Browse, and Category
Landing Pages. Following this period of broad exposure, the sys-
tem’s presence in certain platform-wide touchpoints has been ad-
justed, currently encompassing approximately 15% of that initialscope, as part of ongoing strategic experimentation. The develop-
ment of Flippi is an active process, characterized by continuous
A/B testing and feature refinement to optimize user experience and
system efficacy.
Engaged MAU 1.3M
Last Touch Conversion ∼1.0%
Thumbs-up share 68%
Table 7: L0 Output Metricsand Thumbs-Up Share for Experi-
ence
Key Performance Indicators (KPIs) for Flippi include Engaged
Monthly Active Users (MAU), Channel Conversion, and a user
satisfaction and retention metric. An "Engaged MAU" is defined as
a unique user who initiates at least one query with the assistant
within a month. Table 7 presents key output metrics, alongside a
user satisfaction indicator.
7 CONCLUSION AND FUTURE WORK
Conversational assistants have fundamentally transformed the land-
scape of online shopping, and Flippi stands at the forefront of this
revolution. By delivering a contextual, intuitive, and highly respon-
sive shopping experience, Flippi empowers customers to make con-
fident, informed purchase decisions with unprecedented ease. Its
advanced natural language pipeline enables precise interpretation
of customer queries, ensuring that users receive the most relevant
and timely information available. Seamless integration with spe-
cialized assistants—including decision support, offer discovery, and
expert help—further elevates Flippi into a truly end-to-end solution
for digital commerce.
Flippi’s proven effectiveness is reflected in strong user engage-
ment and satisfaction metrics, yet the journey toward excellence
continues. Future enhancements will focus on refining individual
system components to drive even greater accuracy in search rele-
vance, intent prediction, and self-serve answer quality—key levers
for raising our already impressive thumbs-up rate beyond 68%. Ex-
ploring the deployment of in-house trained models promises to
further strengthen Flippi’s conversational intelligence, enabling
richer, more nuanced interactions that go beyond the capabilities of
prompt-tuned commercial LLMs. Expanding language support to
include regional and vernacular languages will unlock new oppor-
tunities for engagement and inclusivity, broadening Flippi’s reach
across diverse customer segments.
Finally, a deeper investigation into Flippi’s long-term impact
on customer loyalty and retention will yield valuable insights into
the sustained value of conversational AI in e-commerce. As Flippi
continues to evolve, it sets a new benchmark for customer-centric
innovation, shaping the future of digital shopping and redefining
what customers can expect from intelligent, conversational com-
merce.
REFERENCES
Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla
Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sand-
hini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child,
Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse,
Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark,
Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario
Amodei. 2020. Language Models are Few-Shot Learners. CoRR abs/2005.14165
(2020). arXiv:2005.14165 https://arxiv.org/abs/2005.14165

Flippi: End To End GenAI Assistant for E-Commerce KDD2025LLM4ECommerce, August 2025, Toronto, Canada
Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Linyi Yang, Kaijie Zhu, Hao Chen,
Xiaoyuan Yi, Cunxiang Wang, Yidong Wang, Wei Ye, Yue Zhang, Yi Chang, Philip S.
Yu, Qiang Yang, and Xing Xie. 2023. A Survey on Evaluation of Large Language
Models. arXiv:2307.03109 [cs.CL]
Yuxin Cheng, Yuwei Jiang, and Xitong Guo. 2022. Chatbot language style and its
influence on user trust and engagement. International Journal of Information
Management 62 (2022), 102433. https://doi.org/10.1016/j.ijinfomgt.2021.102433
Zheng Chu, Jingchang Chen, Qianglong Chen, Weijiang Yu, Tao He, Haotian Wang,
Weihua Peng, Ming Liu, Bing Qin, and Ting Liu. 2023. A Survey of Chain of Thought
Reasoning: Advances, Frontiers and Future. arXiv:2309.15402 [cs.CL]
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. BERT: Pre-
training of Deep Bidirectional Transformers for Language Understanding. CoRR
abs/1810.04805 (2018). arXiv:1810.04805 http://arxiv.org/abs/1810.04805
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei
Sun, Meng Wang, and Haofen Wang. 2024. Retrieval-Augmented Generation for
Large Language Models: A Survey. arXiv:2312.10997 [cs.CL]
Jie Gui, Tuo Chen, Jing Zhang, Qiong Cao, Zhenan Sun, Hao Luo, and Dacheng Tao.
2023. A Survey on Self-supervised Learning: Algorithms, Applications, and Future
Trends. arXiv:2301.05712 [cs.LG]
Zishan Guo, Renren Jin, Chuang Liu, Yufei Huang, Dan Shi, Supryadi, Linhao Yu, Yan
Liu, Jiaxuan Li, Bojian Xiong, and Deyi Xiong. 2023. Evaluating Large Language
Models: A Comprehensive Survey. arXiv:2310.19736 [cs.CL]
Pranav Gupta, Anand A. Rajasekar, Amisha Patel, Mandar Kulkarni, Alexander Sunell,
Kyung Kim, Krishnan Ganapathy, and Anusua Trivedi. 2022. Answerability: A cus-
tom metric for evaluating chatbot performance. In Proceedings of the 2nd Workshop
on Natural Language Generation, Evaluation, and Metrics (GEM) , Antoine Bosselut,
Khyathi Chandu, Kaustubh Dhole, Varun Gangal, Sebastian Gehrmann, Yacine
Jernite, Jekaterina Novikova, and Laura Perez-Beltrachini (Eds.). Association for
Computational Linguistics, Abu Dhabi, United Arab Emirates (Hybrid), 316–325.
https://doi.org/10.18653/v1/2022.gem-1.27
Alex Havrilla, Yuqing Du, Sharath Chandra Raparthy, Christoforos Nalmpantis, Jane
Dwivedi-Yu, Maksym Zhuravinskyi, Eric Hambro, Sainbayar Sukhbaatar, and
Roberta Raileanu. 2024. Teaching Large Language Models to Reason with Re-
inforcement Learning. arXiv:2403.04642 [cs.LG]
Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang,
Lu Wang, and Weizhu Chen. 2021. LoRA: Low-Rank Adaptation of Large Language
Models. arXiv:2106.09685 [cs.CL]
Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Deven-
dra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume
Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock,
Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El
Sayed. 2023. Mistral 7B. arXiv:2310.06825 [cs.CL]
Douwe Kiela, Max Bartolo, Yixin Nie, Divyansh Kaushik, Atticus Geiger, Zhengxuan
Wu, Bertie Vidgen, Grusha Prasad, Amanpreet Singh, Pratik Ringshia, Zhiyi Ma,
Tristan Thrush, Sebastian Riedel, Zeerak Waseem, Pontus Stenetorp, Robin Jia,
Mohit Bansal, Christopher Potts, and Adina Williams. 2021. Dynabench: Rethinking
Benchmarking in NLP. CoRR abs/2104.14337 (2021). arXiv:2104.14337 https:
//arxiv.org/abs/2104.14337
Robert Kirk, Ishita Mediratta, Christoforos Nalmpantis, Jelena Luketina, Eric Hambro,
Edward Grefenstette, and Roberta Raileanu. 2024. Understanding the Effects of
RLHF on LLM Generalisation and Diversity. arXiv:2310.06452 [cs.LG]
Patrick S. H. Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Se-
bastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented Generation for
Knowledge-Intensive NLP Tasks. CoRR abs/2005.11401 (2020). arXiv:2005.11401
https://arxiv.org/abs/2005.11401
Weng Marc Lim, Satish Kumar, Sanjeev Verma, and Rijul Chaturvedi. 2022. Alexa,
what do we know about conversational commerce? Insights from a systematic
literature review. Psychology & Marketing 39, 3 (2022), 587–605. https://doi.org/10.
1002/mar.21654
Feng Liu, Ruiming Tang, Xutao Li, Weinan Zhang, Yunming Ye, Haokun Chen, Huifeng
Guo, and Yuzhou Zhang. 2019. Deep Reinforcement Learning based Recommenda-
tion with Explicit User-Item Interactions Modeling. arXiv:1810.12027 [cs.IR]
Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham
Neubig. 2021. Pre-train, Prompt, and Predict: A Systematic Survey of Prompting
Methods in Natural Language Processing. arXiv:2107.13586 [cs.CL]
Marcello M. Mariani, Novin Hashemi, and Jochen Wirtz. 2023. Artificial intelligence
empowered conversational agents: A systematic literature review and research
agenda. Journal of Business Research 153 (2023), 336–361. https://doi.org/10.1016/j.
jbusres.2022.08.011
Shervin Minaee, Tomas Mikolov, Narjes Nikzad, Meysam Chenaghlu, Richard Socher,
Xavier Amatriain, and Jianfeng Gao. 2024. Large Language Models: A Survey.
arXiv:2402.06196 [cs.CL]
Humza Naveed, Asad Ullah Khan, Shi Qiu, Muhammad Saqib, Saeed Anwar, Muham-
mad Usman, Naveed Akhtar, Nick Barnes, and Ajmal Mian. 2024. A Comprehensive
Overview of Large Language Models. arXiv:2307.06435 [cs.CL]
OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Flo-
rencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, ShyamalAnadkat, Red Avila, Igor Babuschkin, Suchir Balaji, Valerie Balcom, Paul Baltescu,
Haiming Bao, Mohammad Bavarian, Jeff Belgum, Irwan Bello, Jake Berdine, Gabriel
Bernadett-Shapiro, Christopher Berner, Lenny Bogdonoff, Oleg Boiko, Madelaine
Boyd, Anna-Luisa Brakman, Greg Brockman, Tim Brooks, Miles Brundage, Kevin
Button, Trevor Cai, Rosie Campbell, Andrew Cann, Brittany Carey, Chelsea Carlson,
Rory Carmichael, Brooke Chan, Che Chang, Fotis Chantzis, Derek Chen, Sully
Chen, Ruby Chen, Jason Chen, Mark Chen, Ben Chess, Chester Cho, Casey Chu,
Hyung Won Chung, Dave Cummings, Jeremiah Currier, Yunxing Dai, Cory De-
careaux, Thomas Degry, Noah Deutsch, Damien Deville, Arka Dhar, David Dohan,
Steve Dowling, Sheila Dunning, Adrien Ecoffet, Atty Eleti, Tyna Eloundou, David
Farhi, Liam Fedus, Niko Felix, Simón Posada Fishman, Juston Forte, Isabella Ful-
ford, Leo Gao, Elie Georges, Christian Gibson, Vik Goel, Tarun Gogineni, Gabriel
Goh, Rapha Gontijo-Lopes, Jonathan Gordon, Morgan Grafstein, Scott Gray, Ryan
Greene, Joshua Gross, Shixiang Shane Gu, Yufei Guo, Chris Hallacy, Jesse Han, Jeff
Harris, Yuchen He, Mike Heaton, Johannes Heidecke, Chris Hesse, Alan Hickey,
Wade Hickey, Peter Hoeschele, Brandon Houghton, Kenny Hsu, Shengli Hu, Xin Hu,
Joost Huizinga, Shantanu Jain, Shawn Jain, Joanne Jang, Angela Jiang, Roger Jiang,
Haozhun Jin, Denny Jin, Shino Jomoto, Billie Jonn, Heewoo Jun, Tomer Kaftan,
Łukasz Kaiser, Ali Kamali, Ingmar Kanitscheider, Nitish Shirish Keskar, Tabarak
Khan, Logan Kilpatrick, Jong Wook Kim, Christina Kim, Yongjik Kim, Jan Hendrik
Kirchner, Jamie Kiros, Matt Knight, Daniel Kokotajlo, Łukasz Kondraciuk, Andrew
Kondrich, Aris Konstantinidis, Kyle Kosic, Gretchen Krueger, Vishal Kuo, Michael
Lampe, Ikai Lan, Teddy Lee, Jan Leike, Jade Leung, Daniel Levy, Chak Ming Li,
Rachel Lim, Molly Lin, Stephanie Lin, Mateusz Litwin, Theresa Lopez, Ryan Lowe,
Patricia Lue, Anna Makanju, Kim Malfacini, Sam Manning, Todor Markov, Yaniv
Markovski, Bianca Martin, Katie Mayer, Andrew Mayne, Bob McGrew, Scott Mayer
McKinney, Christine McLeavey, Paul McMillan, Jake McNeil, David Medina, Aalok
Mehta, Jacob Menick, Luke Metz, Andrey Mishchenko, Pamela Mishkin, Vinnie
Monaco, Evan Morikawa, Daniel Mossing, Tong Mu, Mira Murati, Oleg Murk,
David Mély, Ashvin Nair, Reiichiro Nakano, Rajeev Nayak, Arvind Neelakantan,
Richard Ngo, Hyeonwoo Noh, Long Ouyang, Cullen O’Keefe, Jakub Pachocki, Alex
Paino, Joe Palermo, Ashley Pantuliano, Giambattista Parascandolo, Joel Parish, Emy
Parparita, Alex Passos, Mikhail Pavlov, Andrew Peng, Adam Perelman, Filipe de
Avila Belbute Peres, Michael Petrov, Henrique Ponde de Oliveira Pinto, and Michael.
2024. GPT-4 Technical Report. arXiv:2303.08774 [cs.CL]
Shukla Priyanka, Abhishek Kumar, and Rajesh Singh. 2023. Product negotiation in
conversational commerce: A study of chatbot negotiation capabilities. In Proceedings
of the 2023 International Conference on Conversational Agents . ACM, 45–54.
Marco Tulio Ribeiro, Tongshuang Wu, Carlos Guestrin, and Sameer Singh. 2020. Beyond
Accuracy: Behavioral Testing of NLP Models with CheckList. In Proceedings of the
58th Annual Meeting of the Association for Computational Linguistics , Dan Jurafsky,
Joyce Chai, Natalie Schluter, and Joel Tetreault (Eds.). Association for Computational
Linguistics, Online, 4902–4912. https://doi.org/10.18653/v1/2020.acl-main.442
Renny Rumagit, Fanny Tumbel, and Merinda Pandowo. 2023. Chatbots and customer
engagement among Generation Z: An empirical study in e-commerce. Journal of
Theoretical and Applied Electronic Commerce Research 18, 1 (2023), 1–15. https:
//doi.org/10.3390/jtaer18010001
Pranab Sahoo, Ayush Kumar Singh, Sriparna Saha, Vinija Jain, Samrat Mondal, and
Aman Chadha. 2024. A Systematic Survey of Prompt Engineering in Large Language
Models: Techniques and Applications. arXiv:2402.07927 [cs.AI]
Victor Sanh, Albert Webson, Colin Raffel, Stephen H. Bach, Lintang Sutawika, Zaid
Alyafeai, Antoine Chaffin, Arnaud Stiegler, Teven Le Scao, Arun Raja, Manan Dey,
M. Saiful Bari, Canwen Xu, Urmish Thakker, Shanya Sharma, Eliza Szczechla,
Taewoon Kim, Gunjan Chhablani, Nihal V. Nayak, Debajyoti Datta, Jonathan
Chang, Mike Tian-Jian Jiang, Han Wang, Matteo Manica, Sheng Shen, Zheng Xin
Yong, Harshit Pandey, Rachel Bawden, Thomas Wang, Trishala Neeraj, Jos Rozen,
Abheesht Sharma, Andrea Santilli, Thibault Févry, Jason Alan Fries, Ryan Tee-
han, Stella Biderman, Leo Gao, Tali Bers, Thomas Wolf, and Alexander M. Rush.
2021. Multitask Prompted Training Enables Zero-Shot Task Generalization. CoRR
abs/2110.08207 (2021). arXiv:2110.08207 https://arxiv.org/abs/2110.08207
Christian A. Schiller. 2024. The Human Factor in Detecting Errors of Large Lan-
guage Models: A Systematic Literature Review and Future Research Directions.
arXiv:2403.09743 [cs.CL]
Shamane Siriwardhana, Rivindu Weerasekera, Elliott Wen, Tharindu Kaluarachchi,
Rajib Rana, and Suranga Nanayakkara. 2022. Improving the Domain Adaptation
of Retrieval Augmented Generation (RAG) Models for Open Domain Question
Answering. arXiv:2210.02627 [cs.CL]
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux,
Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar,
Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023.
LLaMA: Open and Efficient Foundation Language Models. arXiv:2302.13971 [cs.CL]
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia,
Ed Chi, Quoc Le, and Denny Zhou. 2023. Chain-of-Thought Prompting Elicits
Reasoning in Large Language Models. arXiv:2201.11903 [cs.CL]
Lingling Xu, Haoran Xie, Si-Zhao Joe Qin, Xiaohui Tao, and Fu Lee Wang. 2023.
Parameter-Efficient Fine-Tuning Methods for Pretrained Language Models: A Criti-
cal Review and Assessment. arXiv:2312.12148 [cs.CL]

KDD2025LLM4ECommerce, August 2025, Toronto, Canada Praveen and Anand, et al.
Xiaoyu Zhang, Jicheng Sun, Zhiqiang Li, and Zhenyu Wang. 2022. Dialogue generation
model for conversational product search. Knowledge-Based Systems 238 (2022),
107835. https://doi.org/10.1016/j.knosys.2021.107835
Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou,
Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang,Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang,
Zikang Liu, Peiyu Liu, Jian-Yun Nie, and Ji-Rong Wen. 2023. A Survey of Large
Language Models. arXiv:2303.18223 [cs.CL]
Shen Zheng, Jie Huang, and Kevin Chen-Chuan Chang. 2023. Why Does ChatGPT
Fall Short in Providing Truthful Answers? arXiv:2304.10513 [cs.CL]