# Explainable AI in Usable Privacy and Security: Challenges and Opportunities

**Authors**: Vincent Freiberger, Arthur Fleig, Erik Buchmann

**Published**: 2025-04-17 13:28:01

**PDF URL**: [http://arxiv.org/pdf/2504.12931v1](http://arxiv.org/pdf/2504.12931v1)

## Abstract
Large Language Models (LLMs) are increasingly being used for automated
evaluations and explaining them. However, concerns about explanation quality,
consistency, and hallucinations remain open research challenges, particularly
in high-stakes contexts like privacy and security, where user trust and
decision-making are at stake. In this paper, we investigate these issues in the
context of PRISMe, an interactive privacy policy assessment tool that leverages
LLMs to evaluate and explain website privacy policies. Based on a prior user
study with 22 participants, we identify key concerns regarding LLM judgment
transparency, consistency, and faithfulness, as well as variations in user
preferences for explanation detail and engagement. We discuss potential
strategies to mitigate these concerns, including structured evaluation
criteria, uncertainty estimation, and retrieval-augmented generation (RAG). We
identify a need for adaptive explanation strategies tailored to different user
profiles for LLM-as-a-judge. Our goal is to showcase the application area of
usable privacy and security to be promising for Human-Centered Explainable AI
(HCXAI) to make an impact.

## Full Text


<!-- PDF content starts -->

Explainable AI in Usable Privacy and Security: Challenges and Opportunities
VINCENT FREIBERGER, Center for Scalable Data Analytics and Artificial Intelligence (ScaDS.AI) Dresden/Leipzig,
Germany and Leipzig University, Germany
ARTHUR FLEIG, Center for Scalable Data Analytics and Artificial Intelligence (ScaDS.AI) Dresden/Leipzig, Germany
and Leipzig University, Germany
ERIK BUCHMANN, Center for Scalable Data Analytics and Artificial Intelligence (ScaDS.AI) Dresden/Leipzig,
Germany and Leipzig University, Germany
Large Language Models (LLMs) are increasingly being used for automated evaluations and explaining them. However, concerns about
explanation quality, consistency, and hallucinations remain open research challenges, particularly in high-stakes contexts like privacy
and security, where user trust and decision-making are at stake. In this paper, we investigate these issues in the context of PRISMe, an
interactive privacy policy assessment tool that leverages LLMs to evaluate and explain website privacy policies. Based on a prior user
study with 22 participants, we identify key concerns regarding LLM judgment transparency, consistency, and faithfulness, as well
as variations in user preferences for explanation detail and engagement. We discuss potential strategies to mitigate these concerns,
including structured evaluation criteria, uncertainty estimation, and retrieval-augmented generation (RAG). We identify a need for
adaptive explanation strategies tailored to different user profiles for LLM-as-a-judge. Our goal is to showcase the application area of
usable privacy and security to be promising for Human-Centered Explainable AI (HCXAI) to make an impact.
Additional Key Words and Phrases: Usable Privacy, LLMs, Explainability, Presented at the Human-centered Explainable AI Workshop
(HCXAI) @ CHI 2025, DOI: 10.5281/zenodo.15170451
1 Introduction
With increasing cybersecurity threats – e.g., through the growing number of devices introducing more vulnerabilities
– and privacy threats – such as user data being used to train AI models – empowering and informing users about
these risks has become increasingly important. This is underscored by the fact that most cybersecurity incidents are
facilitated by human error [ 43]. In the field of usable privacy and security, model judgements and explanations and the
effective handling of hallucination scenarios are particularly important, as users’ sensitive data is at stake and misguided
judgments can have severe consequences. A promising approach, which has already gained traction in other research
fields and holds potential for usable privacy and security, is utilizing LLM-as-a-judge [14, 49]. This approach involves
instructing an LLM to evaluate a given document and assign scores. However, despite its potential, key challenges
remain unresolved. The explainability of LLM-generated scores, beyond instructing the LLM to self-report its reasoning,
remains underexplored. Additionally, tailoring these explanations to different user profiles to enhance their effectiveness
in the context of usable security and privacy has yet to be investigated. Furthermore, hallucinations in LLM-as-a-judge
scenarios remain largely unexamined, and mitigation strategies have yet to be explored.
Our specific use case motivating this research is PRISMe, an interactive privacy policy assessment tool we de-
signed [ 12]. It employs an LLM-as-a-judge approach to evaluate and rate privacy policies, providing users with an
Authors’ Contact Information: Vincent Freiberger, Center for Scalable Data Analytics and Artificial Intelligence (ScaDS.AI) Dresden/Leipzig, Leipzig,
Germany and Leipzig University, Leipzig, Germany, freiberger@cs.uni-leipzig.de; Arthur Fleig, Center for Scalable Data Analytics and Artificial
Intelligence (ScaDS.AI) Dresden/Leipzig, Leipzig, Germany and Leipzig University, Leipzig, Germany, arthur.fleig@uni-leipzig.de; Erik Buchmann,
Center for Scalable Data Analytics and Artificial Intelligence (ScaDS.AI) Dresden/Leipzig, Leipzig, Germany and Leipzig University, Leipzig, Germany,
buchmann@informatik.uni-leipzig.de.
1arXiv:2504.12931v1  [cs.HC]  17 Apr 2025

2 Freiberger et al.
overview while incorporating a conversational component that allows them to ask questions about the policy. Un-
derstanding how users interact with explanations provided by such a tool is crucial to ensure understanding, trust,
and good decision making [ 18], as well as determining which types of explanations are most helpful for different user
groups. Our goal is to maximize users’ privacy and security awareness without exposing them to unintended risks,
such as misinterpretation or a false sense of security.
While multiple benchmarks assess LLM hallucinations in question-answering settings [ 17] and allow to transfer
insights to PRISMe, such an understanding is still lacking for LLM-as-a-judge as in our initial policy assessment. While
our previous work [ 12] has focused on discussing issues and potential solutions for the interactive chat component, we
aim to focus on the LLM-as-a-judge component here.
In particular, we identify the following areas for future research and outline our initial thoughts on addressing them:
•Investigating how different user types interact with LLM-generated explanations in the context of privacy
and security (addressing individual factors influencing explainability needs of different users building on Li et
al. [24]).
•Customizing LLM-as-a-judge assessments for different user profiles to enhance interpretability and effectiveness.
•Continuing the trajectory laid out by Abdul et al. [ 1] and work from previous Human-Centered Explainable
AI (HCXAI) research like [ 7,8,18] to create user-tailored, interactive explanations that in our use case help
minimize exposure to security and privacy threats.
•Understanding and detecting hallucinations in an LLM-as-a-judge setting and developing mitigation strategies
tailored to privacy and security contexts (addressing the when to trust). This continues the research theme of
appropriate trust calibration in the HCXAI community [13, 16, 25].
Our goal is to highlight the promising application of usable security and privacy to the HCXAI-community, where
studying model explanations and LLM hallucinations from a human-centered perspective involving personalization can
have a significant positive impact on lay users.
2 Related Work
Challenges with Privacy Policies
Privacy policies aim to reduce the information asymmetry between service providers and users [ 27,48]. However,
they are often designed for legal compliance, with dense, complex, and lawyer-centric language [ 38]. This makes it
difficult for users to make informed decisions about their privacy online [ 30,44]. Legal regulations like the GDPR [ 10]
do not prevent persuasive language, which makes unethical practices even harder to detect [ 5,32]. Additionally, such
regulations may create a false sense of trust and security. Users rarely read and typically don’t understand privacy
policies [ 33,39], which leads to informational unfairness [ 11]. Generative AI [ 22] and Augmented Reality complicate
data management practices further, thereby exacerbating transparency issues [2, 4, 5].
LLM-based privacy policy assessment
LLM-based privacy policy assessment has shown to be as effective as NLP methods in extracting key aspects from
privacy policies, such as contact information and third parties [ 34]. ChatGPT-4 [ 31] offers performance and adaptability
in answering privacy-related questions [ 15], which motivated our use of LLMs in a interactive privacy policy assessment
tool. Privacify [ 45] is a browser extension performing information extraction and abstract summarization of policies
addressing compliance and data collection information. While it lacks interpretation, customization, and interactive
features, it further motivated us to utilize the LLM-as-a-judge paradigm in a tool that interprets and explains privacy
policies and allows users to interact with the provided information [12]. We explain our tool in Section 3.1.

Explainable AI in Usable Privacy and Security: Challenges and Opportunities 3
LLM-as-a-judge
The LLM-as-a-judge paradigm originates from the practice of using LLMs to evaluate other LLM-generated outputs [ 49].
This approach has been widely applied in text evaluation tasks, where LLMs rate and score input text [ 14,23] even
evaluated in a RAG setting [ 46]. Despite becoming a new standard in NLP evaluation, LLM-as-a-judge is subject to
biases [ 6,46,47], its judgments do not always align with human assessments [ 3,40], and the presented reasoning
sometimes is flawed [46], motivating research on LLM explanations.
LLM Explanations
In the field of HCXAI, LLMs are increasingly used to make traditional Explainable AI (XAI) methods’ outputs more
understandable for humans [ 50]. LLMs’ explanations of their own outputs cannot be seen as mechanistically reflecting
their inner workings but rather as samples of post-hoc rationalizations [ 37]. However, step-by-step self-explanations of
the model to derive an output can significantly improve output quality [ 19,26]. To ground explanations, citation-based
explanations could link to evidence in the context [ 37]. LLMs explaining themselves could help users to critically reflect,
justify, and evaluate if prompted accordingly [36].
Hallucinations
Hallucinations can be defined as the generation of nonsensical outputs or output that is unfaithful to the provided
source content [ 29]. Huang et al. divide hallucinations into ones concerning faithfulness, i.e., whether outputs are
faithful to instructions and context given, and ones concerning factuality, i.e., whether outputs are factually correct [ 17].
3 Use Case: Privacy Risk Information Scanner for Me (PRISMe)
We investigate LLM explanations and hallucinations in the application area of usable privacy and security. To this
end, we utilize the exemplary use case of PRISMe, an interactive privacy policy assessment tool. This section explains
PRISMe and a previous user study we conducted with it [12].
3.1 System
Figure 1 illustrates PRISMe, the tool we developed. When users visit any website, PRISMe fetches its privacy policy with
our scraper utilizing headless browser automation. We automatically analyze its privacy policy, highlighting concerns
using smiley icons (Figure 1, top middle). Following the LLM-as-a-judge paradigm, the tool dynamically selects and
evaluates criteria on a 5-point Likert scale for policy assessment (see prompts in Appendix A) utilizing GPT-4o [ 31].
The smiley icon (green, yellow, red) provides a quick summary of the policy’s overall rating. Clicking it opens the
Overview Panel (Figure 1, left), which summarizes key issues, with links to the dashboard and chat interfaces for further
exploration. The Dashboard Panel (Figure 1, bottom middle) displays detailed scores for each assessment criterion, rated
with corresponding smiley icons. Below the dashboard, an explanation of each score is provided. Users can further
explore specific criteria via the Criteria Chat (Figure 1, right) or use the General Chat for broader privacy-related
inquiries. Settings, accessible via a cogwheel icon, allow users to customize the length (short, medium, long) and
complexity (beginner, basic, expert) of chat responses and policy assessments, tailoring the tool to their preferences and
technical expertise.
3.2 User Study
Utilizing the prototype described above, our prior work contains a user study with 22 participants to evaluate it [ 12].
Participants used the prototype in three scenarios. In the first scenario, they were instructed to take their time exploring
the privacy policies of Paypal and focus.de (a popular German news media website) using PRISMe. The aim was

4 Freiberger et al.
Fig. 1. When the user visits a website, our prototype evaluates the privacy policy in the background and displays privacy alerts via
colored scrollbars and a point-of-entry smiley icon (top middle). Clicking the smiley opens an Overview Panel (left) summarizing key
privacy issues, with navigation to a Dynamic Dashboard and chat interface. The dashboard (bottom middle) provides detailed policy
evaluation criteria, which users can chat about (right) by clicking the respective "More" button.
to investigate understanding and awareness. The second scenario addressed participants’ efficiency using PRISMe:
participants compared four online bookshops (Amazon, Hugendubel, Buchkatalog and Kopp), considering the privacy
policy evaluations provided by our prototype. The last scenario allowed participants to use our tool on any website of
their choice (37 different websites were visited). The aim was to investigate understanding, usability, and awareness.
User studies investigating human perception of XAI often focus on related concepts like trust, understanding,
usability, or human-AI collaboration performance [ 35]. Comments made by participants during the scenarios and in
semi-structured interviews on how they perceived assessment transparency and trustworthiness of the tool motivated
us to investigate explainability of LLM-as-a-judge and to connect to the HCXAI community. Some participants wanted
clear policy evidence to justify specific ratings. They found that the LLM-generated explanations were often overly
generic, with little distinction between descriptions for good and bad ratings. Another issue was inconsistency in criteria
selection. Participants noted that the LLM interpreted similarly named criteria differently between policies or selected
entirely different criteria for similar policies. Additionally, some participants observed incoherence between chat
responses and LLM-provided ratings—the explanations justifying ratings sometimes did not align with the sentiment
expressed in chat responses. Participants still found LLM provided explanations helpful.
Our study also revealed different user profiles regarding their usage behavior. Besides usage behavior, our profiles
take into account participants’ self-reported prior knowledge and confidence in their understanding of typical privacy
policies and general data protection as collected in a questionnaire before the study. We suspect these groups differ in
the way they process LLM explanations and potential hallucinations, raising different requirements for explanation.
•Targeted Explorers: Users with prior knowledge who critically inspect evaluations. They prefer specific, extensive
explanations and clearly defined evaluation criteria.
•Novice Explorers: Users with little prior knowledge who rely on tools like PRISMe for guidance. Their learning
goals emerge as they explore. They engage actively and take their time to process information.

Explainable AI in Usable Privacy and Security: Challenges and Opportunities 5
•Information Minimalists: Users who seek quick, high-level summaries with minimal engagement. They prefer
concise overviews rather than interactive exploration.
These diverse information-seeking behaviors require tailored strategies to effectively communicate explanations and
foster awareness. This offers ground for fruitful discussions with the HCXAI community.
4 Discussion
This section discusses LLM explanation and hallucination issues, balancing transparency, trust and critical thinking,
and the trade-off between explainability and privacy. We take into account the user profiles we have identified.
4.1 Investigating the LLM Explanations and Hallucination Issues
One approach to increase the transparency in an LLM-as-a-judge approach like ours is to reduce the degrees of freedom
the LLM has. A fixed criteria catalog with precise definitions and requirements for assigning individual scores could be
created. This would likely also improve the judgment accuracy [ 46]. A drawback of such fixed criteria definitions is that
they may be too rigid and unsuitable to accurately evaluate different risk profiles. This could pose risks to Information
Minimalists , as they mostly rely on initial ratings that in such a case may miss critical information. Also, providing this
context does not ensure the LLM actually follows it upon inference. Utilizing output token uncertainty for tokens on the
scoring may help judge the LLM’s confidence in its judgement [ 41]. Multiple assessments may be sampled to see where
variance in assigned scores occurs may help quantify confidence [ 28]. Communicating such confidence levels supports
all user profiles, particularly Information Minimalists . Visualizing attention may also help as a proxy for explanations
of what tokens contributed how much to the output [ 9]. While Targeted Explorers are likely happy about additional
model explainability to calibrate their level of trust, not overwhelming other user groups with too much additional
information is important. Investigating this trade-off could continue the debate on personalizing explanations.
Hallucination scenarios in LLM-as-a-judge use cases have yet to be investigated. The majority of issues may be
faithfulness-related. The LLM may ignore the actual policy at hand for evaluation and refer to generic themes. This is
supported by the sometimes rather generic explanations the LLM provides for its evaluation. Providing clearly defined
criteria definitions in this context may help to translate the evaluation of each individual criterion into a RAG problem.
As a benefit, explanations for evaluations are probably more specific and can refer to evidence from the policy, however
at large computational costs with multiple RAG-queries being performed for the different criteria. Sampling multiple
assessments and checking them for internal consistency is also a common hallucination detection strategy [ 17,28] and
could help identify unwanted hallucinated evaluation criteria. Communicating specifics and citing evidence effectively
supports Targeted Explorers to build trust in provided explanations.
Regarding factuality, one problem is that the LLM fabricates actually irrelevant evaluation criteria. Factuality of the
Likert-scale ratings is rather difficult to judge as it is not entirely clear what a policy rated with a 4 instead of a 3 on the
criterion data minimization should do better. Assigning Likert-scale ratings is always prone to subjectivity and there is
no ground truth. The often rather generic self-explanations provided by the LLM for its rating might not suffice. Here,
we arrive at a similar conclusion to Sarkar [ 37] and call for future research, particularly in the LLM-as-a-judge setting.
4.2 Balancing Transparency, Trust, and Critical Thinking
LLM-generated explanations should help users make informed decisions by enabling an appropriate level of reliance
instead of blindly trusting the system’s judgments [ 16]. While explanations can mitigate informational unfairness [ 42],
they should also encourage users to reflect critically on privacy policies rather than passively accepting automated

6 Freiberger et al.
evaluations. Different user profiles require varying levels of customization in explanations [ 18]. Future research should
explore how to personalize explanations effectively while mitigating biases in LLM-as-a-judge frameworks.
A key challenge in the LLM-as-a-judge approach is aligning its outputs and explanations with human judgment. While
lay users often show high agreement with LLMs in expert domains, experts tend to disagree more frequently [ 40]. Experts
value factual accuracy, up-to-date and evidence-based content, and prefer concise, precise, and unambiguous language. In
contrast, LLMs often produce verbose outputs with generalized reasoning and limited expert-level rationale. Szymanski
et al. [ 40] found that prompting LLMs with expert personas – like we do – can partially reduce this misalignment.
However, this comes at the cost of reduced alignment with lay users, who generally prefer less technical language.
Incorporating data protection experts into the loop for screening sample assessments to provide clearer guidelines to
the LLM judge combined with evidence-based evaluations via RAG should help. To avoid user overwhelm, particularly
forInformation Minimalists , the prompt design needs to counter LLMs’ tendency to favor long and comprehensive
responses in complex language. Even though rare, LLM-generated explanations also raise concerns about subjectivity
and the risk of confidently justifying incorrect answers [ 20]. Despite this, they often align well with human explanations.
Their tendency toward illustrative examples and selectivity [ 20] can be seen more as a feature than a bug as it helps
users to build an understanding without getting overwhelmed, given the information presented by the LLM includes
the most important aspects. The alignment with human explanations is found to likely increase trust, which becomes
problematic considering convincing justifications for wrong answers. Particularly Information Minimalists andNovice
Explorers are likely to be misled. To what extent forcing an LLM to verify its answer in a separate step, or to incorporate
expert reviews, is an open question, where the HCXAI community could contribute significantly.
4.3 Balancing Privacy and Explainability
There is a tension between privacy and the quality of the explanation provided to users. Running smaller, local LLMs
offers improved privacy around user interests and preferences, at the cost of reduced output quality in both the judge
and conversational components. By increasing the transparency and quality of the explanations of the LLM-as-a-judge
component, there is an increasing risk of service providers slightly modifying their privacy policies to take advantage of
lacking adversarial robustness in the LLM and receive better scores without improving their data protection practices.
Lakshminarayanan and Gautam [ 21] already noted such adversarial robustness trade offs for Explainable AI in general.
Personalizing assessments requires users to directly or indirectly disclose aspects of their identity, which can compromise
privacy, especially if such data is processed by cloud-based models. Inspired by Lakshminarayanan and Gautam [ 21],
we suggest giving the user control by making automated personalization opt-in to protect privacy.
5 Conclusion
As the field of usable privacy and security increasingly relies on opaque LLMs, the need for human-centered explainable
AI (HCXAI) approaches to enhance transparency is more critical than ever. Intransparent LLM-generated judgments
can undermine user trust and lead to errors, posing significant risks when sensitive user data is at stake. Our work
highlights key research gaps to address, including user-adaptive explanations, improved interpretability, and robust
strategies for detecting and mitigating hallucinations in LLM-as-a-judge frameworks. Bridging these gaps is essential to
ensuring that AI-driven privacy tools are not only effective but also trustworthy and user-centric.

Explainable AI in Usable Privacy and Security: Challenges and Opportunities 7
References
[1] Ashraf Abdul, Jo Vermeulen, Danding Wang, Brian Y Lim, and Mohan Kankanhalli. 2018. Trends and trajectories for explainable, accountable and
intelligible systems: An hci research agenda. In Proceedings of the 2018 CHI conference on human factors in computing systems . 1–18.
[2] Bianca Bartelt and Erik Buchmann. 2024. Transparency in Privacy Policies. In 12th International Conference on Building and Exploring Web Based
Environments . IARIA Press, Online, 1–6.
[3] Anna Bavaresco, Raffaella Bernardi, Leonardo Bertolazzi, Desmond Elliott, Raquel Fernández, Albert Gatt, Esam Ghaleb, Mario Giulianelli, Michael
Hanna, Alexander Koller, et al .2024. Llms instead of human judges? a large scale empirical study across 20 nlp evaluation tasks. arXiv preprint
arXiv:2406.18403 (2024).
[4] Shmuel I Becher and Uri Benoliel. 2021. Law in Books and Law in Action: The Readability of Privacy Policies and the GDPR. In Consumer law and
economics . Springer International Publishing, Cham, 179–204.
[5] Veronika Belcheva, Tatiana Ermakova, and Benjamin Fabian. 2023. Understanding Website Privacy Policies—A Longitudinal Analysis Using Natural
Language Processing. Information 14, 11 (2023), 622.
[6] Guiming Hardy Chen, Shunian Chen, Ziche Liu, Feng Jiang, and Benyou Wang. 2024. Humans or llms as the judge? a study on judgement biases.
arXiv preprint arXiv:2402.10669.
[7]Teresa Datta and John P Dickerson. 2023. Who’s thinking? A push for human-centered evaluation of LLMs using the XAI playbook. CHI’23
Workshop on Generative AI and HCI (2023).
[8] Upol Ehsan and Mark O Riedl. 2020. Human-centered explainable ai: Towards a reflective sociotechnical approach. In HCI International 2020-Late
Breaking Papers: Multimodality and Intelligence: 22nd HCI International Conference, HCII 2020, Copenhagen, Denmark, July 19–24, 2020, Proceedings 22 .
Springer, 449–466.
[9] Nour El Houda Dehimi and Zakaria Tolba. 2024. Attention Mechanisms in Deep Learning : Towards Explainable Artificial Intelligence. In 2024 6th
International Conference on Pattern Analysis and Intelligent Systems (PAIS) . 1–7. doi:10.1109/PAIS62114.2024.10541203
[10] European Union. 2016. REGULATION (EU) 2016/679 of the European Parliament AND OF THE Council of 27 April 2016 on the protection of natural
persons with regard to the processing of personal data and on the free movement of such data, and repealing Directive 95/46/EC (General Data
Protection Regulation). Official Journal of the European Union L119/1 (2016), 1–88.
[11] Vincent Freiberger and Erik Buchmann. 2024. Legally Binding but Unfair? Towards Assessing Fairness of Privacy Policies. In Proceedings of the 10th
ACM International Workshop on Security and Privacy Analytics (IWSPA ’24) . Association for Computing Machinery, New York, NY, USA, 15–22.
[12] Vincent Freiberger, Arthur Fleig, and Erik Buchmann. 2025. "You don’t need a university degree to comprehend data protection this way":
LLM-Powered Interactive Privacy Policy Assessment. In Extended Abstracts of the CHI Conference on Human Factors in Computing Systems .
[13] Lisa Graichen, Matthias Graichen, and Mareike Petrosjan. 2022. How to Facilitate Mental Model Building and Mitigate Overtrust Using HCXAI. In
CHI’22 Workshop on Human-Centered Explainable AI .
[14] Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen, Shengjie Ma, Honghao Liu, et al .2024. A
Survey on LLM-as-a-Judge. arXiv preprint arXiv:2411.15594 (2024).
[15] Aamir Hamid, Hemanth Reddy Samidi, Tim Finin, Primal Pappachan, and Roberto Yus. 2023. GenAIPABench: A benchmark for generative AI-based
privacy assistants. arXiv preprint arXiv:2309.05138.
[16] Patrick Hemmer, Max Schemmer, Niklas Kühl, Michael Vössing, and Gerhard Satzger. 2022. On the effect of information asymmetry in human-AI
teams. CHI’22 Workshop on Human-Centered Explainable AI (2022).
[17] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin,
et al.2025. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. ACM Transactions on
Information Systems 43, 2 (2025), 1–55.
[18] Jenia Kim, Henry Maathuis, and Danielle Sent. 2024. Human-centered evaluation of explainable AI applications: a systematic review. Frontiers in
Artificial Intelligence 7 (2024), 1456486.
[19] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are zero-shot reasoners.
Advances in neural information processing systems 35 (2022), 22199–22213.
[20] Jenny Kunz and Marco Kuhlmann. 2024. Properties and challenges of llm-generated explanations. arXiv preprint arXiv:2402.10532 (2024).
[21] Rithika Lakshminarayanan and Sanjana Gautam. 2024. Balancing Act: Improving Privacy in AI through Explainability. CHI’24 Workshop on
Human-Centered Explainable AI (2024).
[22] Hao-Ping Lee, Yu-Ju Yang, Thomas Serban Von Davier, Jodi Forlizzi, and Sauvik Das. 2024. Deepfakes, Phrenology, Surveillance, and More! A
Taxonomy of AI Privacy Risks. In Proceedings of the CHI Conference on Human Factors in Computing Systems . Association for Computing Machinery,
New York, NY, USA, 1–19.
[23] Dawei Li, Bohan Jiang, Liangjie Huang, Alimohammad Beigi, Chengshuai Zhao, Zhen Tan, Amrita Bhattacharjee, Yuxuan Jiang, Canyu Chen,
Tianhao Wu, et al. 2024. From generation to judgment: Opportunities and challenges of llm-as-a-judge. arXiv preprint arXiv:2411.16594.
[24] Zhaoxin Li, Sophie Yang, and Shijie Wang. 2024. Exploring Personality-Driven Personalization in XAI: Enhancing User Trust in Gameplay. CHI’24
Workshop on Human-Centered Explainable AI (2024).
[25] Q Vera Liao and Kush R Varshney. 2021. Human-centered explainable ai (xai): From algorithms to user experiences. arXiv preprint arXiv:2110.10790
(2021).

8 Freiberger et al.
[26] Hui Liu, Qingyu Yin, and William Yang Wang. 2019. Towards Explainable NLP: A Generative Explanation Framework for Text Classification. In
Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics . 5570–5581.
[27] Gianclaudio Malgieri. 2020. The Concept of Fairness in the GDPR: A Linguistic and Contextual Interpretation. In Proceedings of the 2020 Conference
on fairness, accountability, and transparency . Association for Computing Machinery, New York, NY, USA, 154–166.
[28] Potsawee Manakul, Adian Liusie, and Mark JF Gales. 2023. Selfcheckgpt: Zero-resource black-box hallucination detection for generative large
language models. arXiv preprint arXiv:2303.08896 (2023).
[29] Joshua Maynez, Shashi Narayan, Bernd Bohnet, and Ryan McDonald. 2020. On Faithfulness and Factuality in Abstractive Summarization. In
Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics , Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel Tetreault
(Eds.). Association for Computational Linguistics, Online, 1906–1919. doi:10.18653/v1/2020.acl-main.173
[30] Abraham Mhaidli, Selin Fidan, An Doan, Gina Herakovic, Mukund Srinath, Lee Matheson, Shomir Wilson, and Florian Schaub. 2023. Researchers’
experiences in analyzing privacy policies: Challenges and opportunities. Proceedings on Privacy Enhancing Technologies 4 (2023), 287–305.
[31] OpenAI. 2024. GPT-4o. https://openai.com/index/hello-gpt-4o/ Accessed: Jan 2025.
[32] Irene Pollach. 2005. A Typology of Communicative Strategies in Online Privacy Policies: Ethics, Power and Informed Consent. Journal of Business
Ethics 62 (2005), 221–235.
[33] Joel R Reidenberg, Travis Breaux, Lorrie Faith Cranor, Brian French, Amanda Grannis, James T Graves, Fei Liu, Aleecia McDonald, Thomas B
Norton, Rohan Ramanath, et al .2015. Disagreeable privacy policies: Mismatches between meaning and users’ understanding. Berkeley Tech. LJ 30
(2015), 39.
[34] David Rodriguez, Ian Yang, Jose M Del Alamo, and Norman Sadeh. 2024. Large language models: a new approach for privacy policy analysis at
scale. Computing 106 (2024), 1–25.
[35] Yao Rong, Tobias Leemann, Thai-Trang Nguyen, Lisa Fiedler, Peizhu Qian, Vaibhav Unhelkar, Tina Seidel, Gjergji Kasneci, and Enkelejda Kasneci.
2023. Towards human-centered explainable ai: A survey of user studies for model explanations. IEEE transactions on pattern analysis and machine
intelligence 46, 4 (2023), 2104–2122.
[36] Advait Sarkar. 2024. AI Should Challenge, Not Obey. Commun. ACM 67, 10 (2024), 18–21.
[37] Advait Sarkar. 2024. Large Language Models Cannot Explain Themselves. CHI’24 Workshop on Human-Centered Explainable AI (2024).
[38] Florian Schaub, Rebecca Balebako, and Lorrie Faith Cranor. 2017. Designing effective privacy notices and controls. IEEE Internet Computing 21, 3
(2017), 70–77.
[39] Nili Steinfeld. 2016. “I agree to the terms and conditions”:(How) do users read privacy policies online? An eye-tracking experiment. Computers in
human behavior 55 (2016), 992–1000.
[40] Annalisa Szymanski, Noah Ziems, Heather A Eicher-Miller, Toby Jia-Jun Li, Meng Jiang, and Ronald A Metoyer. 2025. Limitations of the LLM-as-
a-Judge approach for evaluating LLM outputs in expert knowledge tasks. In Proceedings of the 30th International Conference on Intelligent User
Interfaces . 952–966.
[41] Neeraj Varshney, Wenlin Yao, Hongming Zhang, Jianshu Chen, and Dong Yu. 2023. A stitch in time saves nine: Detecting and mitigating
hallucinations of llms by validating low-confidence generation. arXiv preprint arXiv:2307.03987 (2023).
[42] Erik Buchmann Vincent Freiberger. 2024. Fair balancing? Evaluating LLM-based privacy policy ethics assessments. In Proceedings of the Third
European Workshop on Algorithmic Fairness (EWAF’24) . CEUR Workshop Proceedings, Aachen, Germany.
[43] Steven Walker-Roberts, Mohammad Hammoudeh, Omar Aldabbas, Mehmet Aydin, and Ali Dehghantanha. 2020. Threats on the horizon: Under-
standing security threats in the era of cyber-physical systems. The Journal of Supercomputing 76 (2020), 2643–2664.
[44] Maximiliane Windl, Niels Henze, Albrecht Schmidt, and Sebastian S Feger. 2022. Automating contextual privacy policies: Design and evaluation
of a production tool for digital consumer privacy awareness. In Proceedings of the 2022 CHI Conference on Human Factors in Computing Systems .
Association for Computing Machinery, New York, NY, USA, 1–18.
[45] Justin Woodring, Katherine Perez, and Aisha Ali-Gombe. 2024. Enhancing privacy policy comprehension through Privacify: A user-centric approach
using advanced language models. Computers & Security 145 (2024), 103997.
[46] Austin Xu, Srijan Bansal, Yifei Ming, Semih Yavuz, and Shafiq Joty. 2025. Does Context Matter? ContextualJudgeBench for Evaluating LLM-based
Judges in Contextual Settings. arXiv preprint arXiv:2503.15620 (2025).
[47] Jiayi Ye, Yanbo Wang, Yue Huang, Dongping Chen, Qihui Zhang, Nuno Moniz, Tian Gao, Werner Geyer, Chao Huang, Pin-Yu Chen, et al .2024.
Justice or prejudice? quantifying biases in llm-as-a-judge. arXiv preprint arXiv:2410.02736 (2024).
[48] Razieh Nokhbeh Zaeem and K. Suzanne Barber. 2020. The Effect of the GDPR on Privacy Policies: Recent Progress and Future Promise. ACM Trans.
Manage. Inf. Syst. 12, 1, Article 2 (dec 2020), 20 pages.
[49] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al .
2023. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in Neural Information Processing Systems 36 (2023), 46595–46623.
[50] Alexandra Zytek, Sara Pidò, and Kalyan Veeramachaneni. 2024. Llms for xai: Future directions for explaining explanations. CHI’24 Workshop on
Human-Centered Explainable AI (2024).

Explainable AI in Usable Privacy and Security: Challenges and Opportunities 9
A Appendix
A.1 Prompting Approach for Initial Assessment Generation
Your output must be a maximum of 600 words long! You are an expert in data protection and a member of an ethics
council. You are given a privacy policy. Your task is to uncover aspects in data protection declarations that are ethically
questionable from your perspective. Proceed step by step :
(1)Criteria: From your perspective, identify relevant ethical test criteria for this privacy policy as criteria for a
later evaluation. When naming the test criteria, stick to standardized terms and concepts that are common in
the field of ethics. Keep it short!
(2)Analysis: Based on this, check for ethical problems or ethically questionable circumstances in the privacy
policy.
(3)Evaluation: Only after you have completed step 2: Rate the privacy policy based on your analysis regarding
each of your criteria on a 5-point Likert scale. Explain what this rating means. Explain what the ideal case with
5 points and the worst case with one point would look like. The output in this step should look like this: [Insert
rating criterion here]: [insert rating here]/5 [insert line break] [insert justification here]
(4)Conclusion: Reflect on your evaluation and check whether it is complete.
Important: Check for errors in your analysis and correct them if necessary before the evaluation. You must present
your approach clearly and concisely and follow the steps mentioned. Your output must not exceed 600 words.
A.2 Prompting Approach for Chat Answer Generation
System prompt criteria chat: Keep it short! Privacy policy: <Privacy policy here> | Rating: <criteria evaluation result
here>. Users want to know more about how this rating is justified in the privacy policy. When answering the questions,
focus on the given topic of the rating. Keep it short! <Complexity and answer length according to settings>
System prompt general chat: You are an expert in data protection with many years of experience in consumer
protection. You have analyzed the following privacy policy and are aware of its risks and ethical implications for users:
<Privacy policy here>. You should advise users and explain the implications for them in a conversation. <Complexity
and answer length according to settings>