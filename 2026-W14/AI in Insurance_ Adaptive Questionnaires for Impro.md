# AI in Insurance: Adaptive Questionnaires for Improved Risk Profiling

**Authors**: Diogo Silva, João Teixeira, Bruno Lima

**Published**: 2026-04-02 13:39:53

**PDF URL**: [https://arxiv.org/pdf/2604.02034v1](https://arxiv.org/pdf/2604.02034v1)

## Abstract
Insurance application processes often rely on lengthy and standardized questionnaires that struggle to capture individual differences. Moreover, insurers must blindly trust users' responses, increasing the chances of fraud. The ARQuest framework introduces a new approach to underwriting by using Large Language Models (LLMs) and alternative data sources to create personalized and adaptive questionnaires. Techniques such as social media image analysis, geographic data categorization, and Retrieval Augmented Generation (RAG) are used to extract meaningful user insights and guide targeted follow-up questions.
  A life insurance system integrated into an industry partner mobile app was tested in two experiments. While traditional questionnaires yielded slightly higher accuracy in risk assessment, adaptive versions powered by GPT models required fewer questions and were preferred by users for their more fluid and engaging experience.
  ARQuest shows great potential to improve user satisfaction and streamline insurance processes. With further development, this approach may exceed traditional methods regarding risk accuracy and help drive innovation in the insurance industry.

## Full Text


<!-- PDF content starts -->

AI in Insurance: Adaptive Questionnaires for Improved Risk
Profiling
Diogo Silva
Deloitte and
Faculty of Engineering,
University of Porto
Porto, Portugal
diogofilipe2002@live.comJoão Teixeira
Deloitte
Porto, Portugal
joaoteixeira@deloitte.ptBruno Lima
LIACC,
Faculty of Engineering,
University of Porto
Porto, Portugal
brunolima@fe.up.pt
Abstract
Insurance application processes often rely on lengthy and standard-
ized questionnaires that struggle to capture individual differences.
Moreover, insurers must blindly trust users’ responses, increasing
the chances of fraud. The ARQuest framework introduces a new
approach to underwriting by using Large Language Models (LLMs)
and alternative data sources to create personalized and adaptive
questionnaires. Techniques such as social media image analysis,
geographic data categorization, and Retrieval Augmented Genera-
tion (RAG) are used to extract meaningful user insights and guide
targeted follow-up questions.
A life insurance system integrated into an industry partner mo-
bile app was tested in two experiments. While traditional question-
naires yielded slightly higher accuracy in risk assessment, adaptive
versions powered by GPT models required fewer questions and
were preferred by users for their more fluid and engaging experi-
ence.
ARQuest shows great potential to improve user satisfaction and
streamline insurance processes. With further development, this
approach may exceed traditional methods regarding risk accuracy
and help drive innovation in the insurance industry.
CCS Concepts
•Software and its engineering →Software verification and
validation;•Information systems →Decision support sys-
tems.
Keywords
AI-Driven Underwriting, Personalized Questionnaires, Insurance
Risk Assessment, Large Language Models, User Data Integration
ACM Reference Format:
Diogo Silva, João Teixeira, and Bruno Lima. 2026. AI in Insurance: Adaptive
Questionnaires for Improved Risk Profiling. InInternational Workshop on
Agentic Engineering (AGENT ’26), April 12–18, 2026, Rio de Janeiro, Brazil.
ACM, New York, NY, USA, 8 pages. https://doi.org/10.1145/3786167.3788425
This work is licensed under a Creative Commons Attribution 4.0 International License.
AGENT ’26, Rio de Janeiro, Brazil
©2026 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-2399-5/2026/04
https://doi.org/10.1145/3786167.37884251 Introduction
Insurance underwriting — the process of evaluating client risk — is
a cornerstone of profitability and long-term business sustainabil-
ity. Traditional methods, often relying on lengthy, manually filled
questionnaires, are prone to human error, fraud, and fail to capture
individual nuances [ 1], leading to unfair premiums for both low
and high risk clients.
The exponential growth of Artificial Intelligence (AI), coupled
with Big Data from digital platforms and social networks, offers a
transformative opportunity [ 2]. Innovations like Natural Language
Processing (NLP) and Large Language Models (LLMs) can enhance
communication with customers, while advanced algorithms (neural
networks, decision trees) can boost risk prediction [ 3]. However,
most insurance questionnaires remain static, ignoring valuable
contextual information from external sources.
This work introduces a novel approach that uses AI to dynami-
cally generate personalized insurance questionnaires by incorpo-
rating external user insights, enabling tailored risk assessment that
adapts to each individual, contrasting with conventional one-size-
fits-all forms. However, bringing AI-driven adaptive questionnaires
into real-world insurance contexts introduces several key chal-
lenges:
•Multiline scalability: Most current models are optimized for
a single insurance domain, making it difficult to adapt to
diverse product lines such as life, health, and property insur-
ance.
•Data integration: External user insights are often unstruc-
tured, context-dependent, and noisy, requiring advanced pre-
processing for AI interpretability, while also meeting strict
privacy and compliance requirements [4, 5].
•Fairness: Imperfect underwriting models risk producing bi-
ased offers, disproportionately affecting certain customer
segments and potentially undermining trust [6].
This paper addresses these challenges through the design, imple-
mentation, and evaluation of ARQuest (Adaptive Risk Questioning),
an adaptive questionnaire framework that leverages LLMs and Re-
trieval Augmented Generation (RAG) to personalize underwriting
flows, integrate heterogeneous sources, and ensure fairness and
transparency in risk profiling. This work results from a collab-
oration with a higher education institution and a multinational
professional services company[ 7], ensuring scientific relevance and
industry applicability.
To guide our study, we formulated the following research ques-
tions:arXiv:2604.02034v1  [cs.AI]  2 Apr 2026

AGENT ’26, April 12–18, 2026, Rio de Janeiro, Brazil Diogo Silva, João Teixeira, and Bruno Lima
RQ1 How can AI generate personalized insurance question-
naires in a clear way?
RQ2 What concerns arise when integrating external user in-
sights for insurance?
RQ3 What limitations do current AI models face in capturing
individual risk factors?
RQ4 How do model confidence and explainability affect AI
risk assessments?
RQ5 How can personalized underwriting solutions scale to
different types of insurance?
The remainder of this paper is structured as follows. Section 2
reviews related state-of-the-art solutions. Section 3 outlines the
methodology behind the ARQuest solution. Section 4 presents the
evaluation process and results. Finally, Section 5 summarizes find-
ings and discusses future research directions.
2 State of the Art
The personalization of insurance questionnaires is largely over-
looked in current systems. Risk prediction tasks have an underlying
set of processes that are relevant for the individualization of the
subscription journey [ 3], such as the way insurers collect data about
an applicant, the AI models and techniques employed to analyze
risk, and all surrounding ethical concerns.
2.1 Data Integration
With Big Data available on a larger scale and data-driven ML mod-
els on the rise [ 2], insurance companies do not just rely on forms
and reports to understand the risk profile of an individual. Social
media, electronic health records, wearable devices, and geograph-
ical information are data sources that enhance these systems [ 8].
APIs and web scraping methods extract insights, which then go
through a transformation process that converts them to a man-
ageable format [ 9]. NLP techniques are focused on processing the
unstructured natural language found in the data [ 10], while im-
ages can be interpreted by advanced models like BLIP [ 11], which
generates textual descriptions. However, social media mining is sus-
ceptible to nuances regarding the interpretation of a user’s lifestyle
[10].
2.2 Personalized Q&A
LLMs and transformers can process specific datasets to generate
personalized questions and answers, which could substitute the
preconceived traditional insurance questionnaires. Fine-tuning and
prompt-tuning are great modern techniques that leverage these pre-
trained general language models and either adapt or contextualize
them to the target domain [ 12]. Using well-known models like GPT,
T5, or BERT variants, different types of questions can be created
automatically and then filtered via human evaluation and adequate
metrics [ 13]. Question answering can be achieved using similar
strategies, coupled with semantic similarity for answer quality
evaluation [ 14]. RAG-based architectures are adopted to mitigate
LLM hallucinations in these scenarios, filtering the relevant content
that is provided [15], as visible in Figure 1.
It is also useful to understand how confident an LLM is in its
analysis. The confidence of an answer can be extracted from the
internal probabilities of the words generated [16].
Figure 1: RAG-based question-answering system workflow
[15].
2.3 Risk Assessment
Regarding risk assessment, insurers typically implement a variety
of ML algorithms (e.g., random forest, XGBoost, etc.) [ 17] and even
transformers, which outperform traditional models at interpreting
user risk [ 18]. The models analyze variables such as age, physical
activity, chronic diseases, accident history, and weather conditions
to infer if an individual is low or high risk, with additional tasks
like fraud detection being executed as well. GPT-4V, for instance, is
a powerful model that, following an intelligent prompt, can process
the contents of different types of input to capture interesting risk
indicators [19].
2.4 Ethical Concerns
Due to the insurance implications for user livelihoods, ethical as-
pects need to be considered when integrating ML models. Com-
pliance with regulatory frameworks like the GDPR [ 20] and the
AIA [ 21] should be a priority, as insurance apps are inherently a
high-risk system [ 21]. Attempts at protecting consumer data in-
clude data encryption and anonymization [ 10]. In addition, model
discrimination is presented as another vital challenge, as customers
from minority groups may be affected by biased premium attri-
bution [ 6]. Resampling techniques aid in transforming a dataset
so that it becomes representative of the population [ 22]. Selecting
proper evaluation benchmarks and explainability methods, such as
SHAP [ 23], LIME [ 23], or prompting-based techniques [ 24], is also
essential to integrate fairness and transparency into the analytical
and decision-making process as much as possible.
3 ARQuest
The ARQuest (Adaptive Risk Questioning) framework is presented
as a solution to personalize insurance questionnaires by adjusting
the questions to the users, aiming at an improved subscription
experience and a more precise risk assessment.
The framework, detailed in Figure 2, is divided into four unique
modules for different tasks:

AI in Insurance: Adaptive Questionnaires for Improved Risk Profiling AGENT ’26, April 12–18, 2026, Rio de Janeiro, Brazil
Figure 2: ARQuest framework and its modules.
(1)User Profiling:the applicant provides some basic personal
information, such as the location, which is used to retrieve
relevant geographical risk indicators; data from external
platforms can also be shared, with all this information being
processed and aggregated.
(2)Response Forecasting:by analyzing the collected insights,
an LLM identifies risks from a pre-defined list; these risks
have questions assigned, and the LLM predicts the respective
answers beforehand; optionally, the user can review and
correct these predictions.
(3)Dynamic Questioning:some risks are not detectable from
the initial data, so the LLM chooses the next factor (from
the remaining) to question the user; this process is repeated,
and the user continuously provides answers until the model
considers enough information is acquired; briefly, the LLM
is prompted to stop the questioning process when none of
the remaining factors are both impactful or likely to yield a
risky response.
(4)Risk Assessment:based on the completed questionnaire,
a global risk score is computed using a specific model or
algorithm; cases where the LLM’s predictions are drastically
corrected by the user are also identified using a similarity
measure between guesses and answers; if this occurs fre-
quently, either the LLM performed poorly in its analysis or
the applicant intentionally misled the evaluation.
3.1 System Implementation
A solution focused onlife insuranceis implemented within a
mobile app project from an industry partner. Figure 3 showcases
the overall flow and architecture behind this system.
As depicted, the user can share health records, fitness data, or
Instagram posts. The backend communicates with a special service
dedicated to the solution’s tasks. A GPT model is employed via
Azure services [25] that deal with data protection.
Figure 3: System’s flow and architecture.
3.2 Data Preparation
Life insurance questionsgather information to assess the mor-
tality risk of an individual. Internal and external factors play a
role here, such as age, health, genetics, habits, environment, local
services, and community. Based on templates from insurance com-
panies [ 26,27], a dataset of multiple-choice questions is created,
separated by the following categories: personal details, lifestyle and
habits, family history, and health status. The goal is to provide the
LLM with a set of risk factors from which it can choose to address,
depending on the applicant.
The system resorts togeographical indicatorscollected from
the Atlas of Healthy Municipalities website [ 28]. It contains statis-
tics on Portugal’s municipalities, spread across nine health cate-
gories: mortality, morbidity, healthcare, lifestyle, education, socioe-
conomic status, environment, infrastructure, and security. There
are a total of 53 individual priority indicators. The data is extracted
via web scraping methods. Additionally, all values are labeled us-
ing the k-means clustering algorithm, mentioned by Biddleet al.
[29]. In this way, the LLM instantly recognizes that a certain in-
dicator is either "very low" or "very high", instead of having to
interpret numerical values with no context surrounding them. Fig-
ure 4 demonstrates how raw data points are categorized for a given
regional indicator.
Figure 4: Raw and labeled municipality values for alcohol
mortality.
The creation of a large set of synthetic users can be helpful
in statistically assessing the system’s performance.Electronic

AGENT ’26, April 12–18, 2026, Rio de Janeiro, Brazil Diogo Silva, João Teixeira, and Bruno Lima
health records(EHRs) are initially collected from SyntheaTM
[30], an open-source synthetic patient generation platform with
realistic health data. The records contain information on conditions,
encounters, medications, procedures, and other relevant aspects.
Social media is useful to understand user interests and activities.
Interesting insights, such as risky hobbies, can be captured from
this data source. Images from thousands ofInstagram postsare
gathered from a Kaggle dataset [ 31]. Figure 5 displays one of the
many experiments that take place with the BLIP model [ 11] to com-
prehend its image captioning capabilities. A specific combination
of hyperparameters (temperature=0.3,top-p=0.9) offers an effective
balance between determinism and creativity. This approach is a
lightweight and direct solution that semantically enriches the user
information fed to the LLM.
Figure 5: Captions with different parameters for an image
[31].
Thesynthetic applicant datasetshould strive to represent the
target population. The following components are utilized to create
a total of 85 artificial users:
•A sample of health records matching the real age and gender
distributions.
•The population across Portugal’s municipalities.
•A coherent allocation of image descriptions from posts.
•Probabilities of working in different occupations.
•Average daily steps (fitness metric) estimated using jobs.
•Probabilities of sharing external sources.
3.3 Questionnaire Generation
For each user, answers to all possible questions from the dataset are
pre-computed, resorting to the individual information stored and
probabilities. These saved answers act as "ground truths" and auto-
mate questionnaire filling upon the execution of tests. Considering
a question addressing family diabetes, for instance, if the user’shealth records mention diabetes and the regional indicators present
a high prevalence of the disease, the answer probabilities will favor
a positive answer, as it is likely that a family member suffers from
that condition, which possesses a hereditary component to it.
Two approaches are compared:traditionalanddynamic. The
traditional questionnaires are static and composed of a pre-defined
subset of factors from the question dataset, since real questionnaires
do not cover all possible issues, only the most common and influ-
ential ones. In this case, they contain 30 questions about lifestyle
and habits, family history, and health status, 10 for each category.
The described methodology is visible in Figure 6, where the user is
presented with pre-selected questions.
Figure 6: Traditional flow.
The dynamic approach is based on the ARQuest framework, re-
sorting to a GPT model. Figure 7 showcases the prompt given to the
model containing answer prediction instructions, which is crucial
in the response forecasting task. User insights are initially included
in the input. They are gathered using the RAG methodology [ 15],
which performs a similarity search by querying all risk factors from
the dataset against all available data in the user’s external sources.
The intended return format (JSON) is presented at the end. The
prediction confidence can be extracted by analyzing the token prob-
abilities within the answer field value. An explanation is demanded
to reinforce transparency and enhance reasoning.
Figure 7: Prompt with answer prediction instructions.
3.4 Mobile App Integration
Regarding the implementation of the features in the insurance
mobile app, the traditional approach begins by simply asking the
user about some basic personal details, as visible in Figure 8.

AI in Insurance: Adaptive Questionnaires for Improved Risk Profiling AGENT ’26, April 12–18, 2026, Rio de Janeiro, Brazil
Figure 8: Traditional "Insights" section.
Subsequently, the user is required to respond to three sepa-
rate sections targeting mortality risk factors in different domains:
"Lifestyle & Habits", "Family History", and "Health Status". Figure 9
shows the interface containing these components. They are com-
posed of pre-defined questions, following the typical traditional
subscription.
Figure 9: Traditional risk forms.
Unlike the traditional flow, the dynamic approach lets the appli-
cant link external sources (health records, fitness trackers, Insta-
gram posts) to the app. Upon selection of one of them, the user just
needs to read and accept the terms and conditions and authenticatewith the corresponding platform, as shown in Figure 10. In addition
to the alternative sources, the user provides personal details, such
as age and location.
Figure 10: Dynamic "Insights" section.
Following the steps described above, the user is presented with
a single dynamic risk form, displayed in Figure 11. This question-
naire can include pre-filled questions (with predicted answers and
explanations) from the risks identified by the model based on the
initial insights, so that the user reviews (and potentially corrects)
the analysis performed in a straightforward way. The user is also
required to respond to the questions eventually selected by the
model for additional feedback.
Figure 11: Dynamic risk form.
3.5 Risk Scoring
There are multiple ways to determine the overall risk score of an
applicant based on the questionnaire content. On the one hand, a
probabilistic AI model can achieve accurate results through training

AGENT ’26, April 12–18, 2026, Rio de Janeiro, Brazil Diogo Silva, João Teixeira, and Bruno Lima
with real-world trends, but there are interpretability issues due to
its black-box nature, with results potentially being influenced by
hidden patterns. On the other hand, a deterministic model allows
for total control over all variables, even if it increases the difficulty
of reaching global optimization.
The priority here is to have a solid algorithm for comparison
purposes that is easy to analyze. Therefore, the system resorts to a
monotonic additive risk model, where risk starts at zero and can
only go up, increasing with non-optimal answers. Moreover, some
combinations of responses between related factors increase risk by
an additional value.
4 Evaluation
Two major experiments were conducted to evaluate the solution.
The first used the synthetic applicant dataset to compare traditional
questionnaires with adaptive versions powered by GPT-3.5 Turbo
and GPT-4.1. The second relied on the participation of a small
sample of real users to collect user experience insights and overall
impressions.
4.1 Synthetic User Evaluation
Each of the 85 synthetic applicants performed a single traditional
subscription and two dynamic subscriptions, one using the GPT-3.5
Turbo model and another with GPT-4.1, to assess whether the older
LLM is up to the challenge compared to the newer, more capable
one.
Furthermore, these artificial users are previously assigned with
"true" risk scores, calculated from the sum of the weights of the
answers to all possible questions from the risk dataset, not just the
ones that appear in the traditional or dynamic questionnaires.
Figure 12: Comparison of approaches for the main evaluation
metrics.
Figure 12 showcases how the traditional and dynamic (GPT-3.5
Turbo and GPT-4.1) approaches performed using some relevant
metrics. On average, an applicant needs to answer just half the
questions of the traditional forms with the dynamic flow, and even
less when using GPT-3.5 Turbo. Based on empirical observation, the
older LLM seems to be more conservative and cautious, frequently
exiting the dynamic questioning process earlier because it interprets
ambiguous criteria more narrowly or hesitates to act without strong
justification.
However, the risk scores obtained with the traditional approach
have a lower mean average error (MAE) or distance to the truerisk values, followed by the dynamic method with GPT-4.1 and
finally with GPT-3.5 Turbo. The same is observed when analyzing
the Pearson correlation with the true scores. Concerning task times,
the GPT-4.1 model is significantly faster in both answer prediction
and dynamic factor selection.
In general, the dynamic approach using the best model (GPT-
4.1) appears to be worse at capturing relevant risk factors than
the traditional one, typically by 10-30% regarding the risk scores
obtained. This is mainly due to the significant lack of family history
questions within dynamic questionnaires, which is a domain where
many users have some risk-related issues.
4.2 Real User Evaluation
Ten real participants tested both approaches using the insurance
mobile app. This sample represents the characteristics of the popu-
lation to some extent, such as the age and gender distributions. To
allow for valid comparisons, half of the users tested the traditional
method first, while the other half began with the dynamic flow.
Each validation session comprises the following stages:
(1)Experiment context:the participant is initially contextual-
ized, with some general instructions.
(2)Prior data uploading:external insights are uploaded be-
forehand in case the user intends to share them during the
experiment.
(3)Initial navigation:the user navigates to the life insurance
subscription section inside the mobile app.
(4)Testing steps:the participant tests both the traditional and
dynamic approaches, with minimal interference from the
orchestrator to avoid influencing perceptions.
(5)Final feedback form:the participant fills a final form that
collects user experience feedback.
Figure 13: Risk scores per participant using both approaches.
The risk scores for the participants using both approaches are
detailed in Figure 13. Once again, traditional questionnaires gather
a higher number of risk factors, leading to higher individual scores.
Even so, the values attained using the dynamic forms do not fall far
behind, and the relative risk distribution is similar as well. Addition-
ally, the increased trustworthiness that comes with the ARQuest-
based methodology enhances the reliability of these results.

AI in Insurance: Adaptive Questionnaires for Improved Risk Profiling AGENT ’26, April 12–18, 2026, Rio de Janeiro, Brazil
Overall, the feedback results indicate a clear preference for the
dynamic approach over the traditional static questionnaire. The
participants found it more engaging and better suited to their in-
dividual context, reinforcing the value of adaptive interactions in
insurance risk assessment.
Most of the participants consider the dynamic approach to be
more straightforward and less tedious, with 70% preferring this
solution entirely, visible by the orange bars in Figure 14. There is
also a general impression that both methods should be similar in
determining individual mortality risk, which holds true to some
extent.
Figure 14: Participant responses for Question 2 of the feed-
back form.
Focusing on the dynamic approach, participants recognize the
effective personalization of the questions presented and the coher-
ent exploration of relevant topics, as elucidated in Figure 15. The
answer predictions and explanations helped to improve the clarity
of the process, with a few exceptions from not-so-satisfied users.
Figure 15: Participant responses for Question 9 of the feed-
back form.
4.3 Discussion
After the presented methodologies and analysis, the research ques-
tions can be properly answered.
RQ1 How can AI generate personalized insurance ques-
tionnaires in a clear way?
By following the ARQuest framework. Briefly, an AI-powered
system can simply follow these steps:(1) Analyze insights about the applicant
(2) Identify risks from a pre-made list
(3)Predict those answers beforehand, automating question-
naire completion.
(4) Address additional factors, questioning them one by one.
This is a straightforward solution that is easily understand-
able by both customers and insurers.
RQ2 What concerns arise when integrating external user
insights for insurance?
Mainly data processing and ethical concerns. Alternative
sources are usually composed of unstructured data, which
can be substantially contextual and ambiguous. The deci-
sions behind the transformation processes must be thought
of carefully to allow easier LLM interpretability. The system
should also ensure transparency to build a trusting relation-
ship with the customer and approach each applicant in a
way that guarantees fair decision-making.
RQ3 What limitations do current AI models face in cap-
turing individual risk factors?
Considering the involvement of LLMs in this context, there
is a major trade-off between prompt length and model inter-
pretability. Providing the model with the entire content of
all questions from the dataset to choose from slows down
responses and increases the likelihood of hallucinations oc-
curring. Alternatively, simply listing the names of all risk
factors for selection causes a significant loss of nuance within
the prompt.
RQ4 How do model confidence and explainability affect
AI risk assessments?
By improving comprehension of the risk analysis performed
by the AI agent for both applicants and insurers, thereby
boosting trustworthiness. The confidence level of the pre-
dictions is important to identify questions where a high-
confidence prediction is drastically altered by the user. The
explanations increase transparency and are crucial in proof-
checking risk detection, following a case by case analysis.
RQ5 How can personalized underwriting solutions scale
to different types of insurance?
The ARQuest framework demonstrates how this can be
achieved. For a given insurance line, the system needs a
knowledge base with the following components: contextu-
ally relevant geographical indicators, a quality set of risks
and associated questions, a model that calculates a risk score
based on the provided user answers, and word embeddings
that help detect discrepancies between model guesses and
user corrections.
5 Conclusion
This study advances personalized questionnaires for insurance un-
derwriting, aiming to improve risk profiling across multiple do-
mains. It addresses a gap in existing research, where the question-
ing process is often overlooked and few solutions adapt seamlessly
across insurance sectors. While alternative data sources are in-
creasingly leveraged, risk assessment is frequently delegated to

AGENT ’26, April 12–18, 2026, Rio de Janeiro, Brazil Diogo Silva, João Teixeira, and Bruno Lima
pre-trained language models capable of capturing complex pat-
terns. Given the sensitivity of insurance, such solutions must also
ensure transparency, fairness, and regulatory compliance.
The proposed ARQuest framework personalizes risk assessment
by integrating external user insights to identify relevant factors,
reducing manual input. It dynamically selects the most informative
questions to complete the risk profile and incorporates mechanisms
to detect inconsistencies between model reasoning and user re-
sponses, strengthening automation reliability.
A life insurance prototype was implemented and evaluated through
two experiments: synthetic profiles combining geographic, health,
fitness, and social media inputs, and real-user testing via a mobile
application. While traditional questionnaires marginally outper-
formed the adaptive version in detecting certain risk factors, users
consistently preferred the dynamic approach for its efficiency, con-
textual relevance, and improved experience.
Adopting ARQuest offers insurers tangible operational and com-
mercial advantages. Adaptive questioning, coupled with external
data integration, shortens application and underwriting times, re-
duces manual processing costs, and enhances customer satisfaction
through a smoother, more personalized process. By capturing accu-
rate data in fewer steps, insurers can improve predictive accuracy,
streamline workflows, and refine pricing strategies, strengthening
their competitive position.
The main contributions of this work are:
•The design of ARQuest, an adaptive questionnaire frame-
work tailored to individual applicant characteristics.
•The validation of this framework with both synthetic and
real users, demonstrating technical effectiveness, user accep-
tance, and commercial viability.
5.1 Future Work
Future enhancements to the ARQuest framework will focus on four
main directions. First, the adoption of next-generation LLMs with
larger context windows and improved reasoning capabilities is ex-
pected to enable more accurate detection of subtle and complex risk
patterns. Second, the framework should be tested across multiple
insurance lines — such as health, property, and auto — to evaluate
its adaptability to diverse risk domains and varying regulatory re-
quirements. Third, fine-tuning the underlying LLM on high-quality,
domain-specific datasets could improve both consistency and pre-
dictive accuracy, reducing reliance on prompt engineering alone.
Finally, integrating agentic AI capabilities capable of autonomously
interacting with external tools and data sources would support
full end-to-end automation of the underwriting process, further
reducing operational overhead and increasing scalability.
References
[1]L. Mushunje, “Fraud detection and fraudulent risks management in the insurance
sector using selected data mining tools,” in American Journal of Data Mining
and Knowledge Discovery, 2019.
[2]M. Eling, D. Nuessle, and J. Staubli, “The impact of artificial intelligence along
the insurance value chain and on the insurability of risks, ” in The Geneva Papers
on Risk and Insurance - Issues and Practice, 2022.
[3]M. Riikkinen, H. Saarijärvi, P. Sarlin, and I. Lähteenmäki, “Using artificial intelli-
gence to create value in insurance,” in International Journal of Bank Marketing,
2018.
[4]I. Velkova, “Unstructured social media data processing with artificial intelligence,”
in Industry 4.0, 2023.[5]Z. Bednarz and K. Manwaring, “Hidden depths: The effects of extrinsic data
collection on consumer insurance contracts, ” in Computer Law & Security Review,
2022.
[6]J. Perumalsamy, B. Konidena, and B. Krothapalli, “AI-driven risk modeling in
life insurance: advanced techniques for mortality and longevity prediction,” in
Journal of Artificial Intelligence Research and Applications, 2023.
[7]Deloitte. (2025, Jun.) Deloitte. [Online]. Available:
https://www.deloitte.com/global/en.html
[8]M. Sahu, “Machine learning algorithms for automated underwriting in insurance:
techniques, tools, and real-world applications,” in Distributed Learning and Broad
Applications in Scientific Research, pp. 286–326, 2019.
[9]J. Sreerama, V. Rambabu, and C. Murth, “Machine learning-driven data integra-
tion: revolutionizing customer insights in retail and insurance,” in Journal of
Artificial Intelligence Research and Applications, pp. 485–533, 2023.
[10] D. Kanchett, “Social media data in insurance: exploring new frontiers for cus-
tomer insights and risk analysis,” in ESP Journal of Engineering Technology
Advancements, pp. 168–180, 2023.
[11] J. Li, D. Li, C. Xiong, and S. Hoi, “BLIP: bootstrapping language-image pre-training
for unified vision-language understanding and generation,” in International con-
ference on machine learning, pp. 12888–12900, 2022.
[12] S. Maity and A. Deroy, “The future of learning in the age of generative AI:
automated question generation and assessment with large language models,”
unpublished.
[13] K. Nassiri and M. Akhloufi, “Transformer models used for text-based question
answering systems,” in Applied Intelligence, pp. 10602–10635, 2023.
[14] J. Risch, T. Möller, J. Gutsch, and M. Pietsch, “Semantic answer similarity for
evaluating question answering Models,” in Proceedings of the 3rd Workshop on
Machine Reading for Question Answering, pp. 149–157, 2021.
[15] H. Patel, A. Surti, P. Goel, and B. Patel, “A comparative analysis of large language
models with retrieval-augmented generation based question answering system,”
in 8th International Conference on I-SMAC, pp. 792–798, 2024.
[16] A. Kumar, R. Morabito, S. Umbet, J. Kabbara, A. Emami, “Confidence under
the hood: an investigation into the confidence-probability alignment in large
language models,” unpublished.
[17] S. Devaguptam, S. Gorti, T. Akshaya, and S. Kamath, “Automated health insurance
processing framework with intelligent fraud detection, risk classification and
premium prediction,” in SN Computer Science, pp. 1–14, 2024.
[18] V. Preez, S. Bennet, M. Byrne, and A. Couloumy, “From bias to black boxes:
understanding and managing the risks of AI – an actuarial perspective, ” in British
Actuarial Journal, 2024.
[19] C. Lin, H. Lyu, J. Luo, and X. Xu, “Harnessing GPT-4V(ision) for insurance: a
preliminary exploration,” unpublished.
[20] V. Rambabu, J. Sreerama, and J. Singh, “AI-driven data integration: enhancing risk
assessment in the insurance industry,” in Australian Journal of Machine Learning
Research & Applications, pp. 130—179, 2022.
[21] G. Gezici, C. Mannari, and L. Orlandi. “The ethical impact assessment of selling
life insurance to titanic passengers,” in CEUR Workshop Proceedings, pp. 35–50,
2023.
[22] M. Hanafy and R. Ming, “Using machine learning models to compare various
resampling methods in predicting insurance fraud, ” in Journal of Theoretical and
Applied Information Technology, pp. 2819–2833, 2021.
[23] T. Sun, “Enhancing auto insurance risk evaluation with transformer and SHAP,”
in IEEE Access, pp. 116546–116557, 2024.
[24] H. Cai, “Explainability for large language models: a survey,” in ACM Transactions
on Intelligent Systems and Technology, pp. 1–38, 2023.
[25] Microsoft. (2025, Feb.) Azure OpenAI. [Online]. Available:
https://azure.microsoft.com/en-us/products/ai-services/openai-service
[26] Ageas SA/NV. (2025, Feb.) Ageas. [Online]. Available: https://www.ageas.com/en
[27] Allianz Life. (2025, Feb.) Allianz Life: Annuities, Life Insurance, and Asset Man-
agement. [Online]. Available: https://www.allianzlife.com/
[28] Atlas dos Municípios Saudáveis. (2025, Mar.) Atlas dos Municípios Saudáveis.
[Online]. Available: https://www.atlasmunicipiossaudaveis.pt/
[29] R. Biddle, S. Liu, and G. Xu, “Semi-supervised soft k-means clustering of life in-
surance questionnaire responses, ” in 5th International Conference on Behavioral,
Economic, and Socio-Cultural Computing (BESC), 2018.
[30] SyntheaTM. (2025, Mar.) Synthea. [Online]. Available:
https://synthetichealth.github.io/synthea/
[31] H. Kapadia. (2025, Mar.) Instagram posts dataset. [Online]. Available:
https://www.kaggle.com/datasets/thecoderenroute/instagram-posts-dataset