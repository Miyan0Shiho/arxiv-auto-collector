# InsurAgent: A Large Language Model-Empowered Agent for Simulating Individual Behavior in Purchasing Flood Insurance

**Authors**: Ziheng Geng, Jiachen Liu, Ran Cao, Lu Cheng, Dan M. Frangopol, Minghui Cheng

**Published**: 2025-11-03 23:19:27

**PDF URL**: [http://arxiv.org/pdf/2511.02119v1](http://arxiv.org/pdf/2511.02119v1)

## Abstract
Flood insurance is an effective strategy for individuals to mitigate
disaster-related losses. However, participation rates among at-risk populations
in the United States remain strikingly low. This gap underscores the need to
understand and model the behavioral mechanisms underlying insurance decisions.
Large language models (LLMs) have recently exhibited human-like intelligence
across wide-ranging tasks, offering promising tools for simulating human
decision-making. This study constructs a benchmark dataset to capture insurance
purchase probabilities across factors. Using this dataset, the capacity of LLMs
is evaluated: while LLMs exhibit a qualitative understanding of factors, they
fall short in estimating quantitative probabilities. To address this
limitation, InsurAgent, an LLM-empowered agent comprising five modules
including perception, retrieval, reasoning, action, and memory, is proposed.
The retrieval module leverages retrieval-augmented generation (RAG) to ground
decisions in empirical survey data, achieving accurate estimation of marginal
and bivariate probabilities. The reasoning module leverages LLM common sense to
extrapolate beyond survey data, capturing contextual information that is
intractable for traditional models. The memory module supports the simulation
of temporal decision evolutions, illustrated through a roller coaster life
trajectory. Overall, InsurAgent provides a valuable tool for behavioral
modeling and policy analysis.

## Full Text


<!-- PDF content starts -->

INSURAGENT: A LARGELANGUAGEMODEL-EMPOWERED
AGENT FORSIMULATINGINDIVIDUALBEHAVIOR IN
PURCHASINGFLOODINSURANCE
Ziheng Geng1,∗, Jiachen Liu2,∗, Ran Cao3, Lu Cheng4, Dan M. Frangopol5, Minghui Cheng1,6†
1Department of Civil and Architectural Engineering, University of Miami, Coral Gables, FL 33146, USA
2Department of Electrical and Computer Engineering, University of Miami, Coral Gables, FL 33146, USA
3College of Civil Engineering, Hunan University, Changsha, 410082, China
4Department of Computer Science, University of Illinois Chicago, Chicago, IL 60607, USA
5Department of Civil and Environmental Engineering, Lehigh University, Bethlehem, PA 18015, USA
6School of Architecture, University of Miami, Coral Gables, FL 33146, USA
∗Equal contribution.
†Corresponding author:minghui.cheng@miami.edu
November 5, 2025
ABSTRACT
Flood insurance is an effective strategy for individuals to mitigate disaster-related losses. However,
participation rates among at-risk populations in the United States remain strikingly low. This gap
underscores the need to understand and model the behavioral mechanisms underlying insurance
decisions. Large language models (LLMs) have recently exhibited human-like intelligence across
wide-ranging tasks, offering promising tools for simulating human decision-making. This study
constructs a benchmark dataset to capture insurance purchase probabilities across factors. Using
this dataset, the capacity of LLMs is evaluated: while LLMs exhibit a qualitative understanding of
factors, they fall short in estimating quantitative probabilities. To address this limitation, InsurAgent,
an LLM-empowered agent comprising five modules including perception, retrieval, reasoning, action,
and memory, is proposed. The retrieval module leverages retrieval-augmented generation (RAG) to
ground decisions in empirical survey data, achieving accurate estimation of marginal and bivariate
probabilities. The reasoning module leverages LLM common sense to extrapolate beyond survey
data, capturing contextual information that is intractable for traditional models. The memory module
supports the simulation of temporal decision evolutions, illustrated through a roller coaster life
trajectory. Overall, InsurAgent provides a valuable tool for behavioral modeling and policy analysis.
Keywords:Large language models; LLM agents; Retrieval-augmented generation, Flood insurance
purchase; Human behavior modeling
1 Introduction
Flooding is among the most pervasive and devastating natural hazards worldwide, posing substantial risk to infrastructure
reliability, human safety, and socio-economic stability. Under the influence of climate change, the frequency and
intensity of extreme precipitation events have markedly increased, contributing to a growing number of flood-related
disasters (Tezuka et al., 2014; Liu et al., 2020; Cheng and Frangopol, 2023). In the United States, Hurricane Harvey
in 2017 caused over 100 fatalities and inflicted more than $125 billion in damage (Kundis Craig, 2017; HEGAR,
2018). The resulting floods overwhelmed hundreds of thousands of homes and displaced more than 30,000 residents
(Comes et al., 2017; Sebastian et al., 2017). In response to the growing threat of flood hazards, purchasing floodarXiv:2511.02119v1  [cs.AI]  3 Nov 2025

APREPRINT- NOVEMBER5, 2025
insurance is widely recognized as an effective strategy for individuals to mitigate associated economic losses. Despite
its critical role, participation rate in flood insurance programs remains strikingly low, particularly among high-risk
populations. According to a report by the Congressional Budget Office, only 18% of properties within Special Flood
Hazard Areas (SFHAs) were covered by the National Flood Insurance Program (NFIP), with coverage dropping to
just 4% outside these zones (Congressional Budget Office, 2024). This huge gap between risk exposure and insurance
purchase highlights the urgent need to understand the behavioral mechanisms that drive individual decisions regarding
flood insurance.
Therefore, extensive research has employed questionnaire surveys to identify the factors that influence individual
decisions, given their effectiveness in capturing perceptions and attitudes toward risk (Bird, 2009; Cheng et al., 2020;
Cheng and Frangopol, 2022). These investigations reveal that the decisions to purchase flood insurance result from
a complex interplay of economic, geographic, psychological, and socio-demographic factors. Among these factors,
objective indicators of economic capacity and flood exposure are consistently found to be key determinants. Households
with higher incomes and those residing in higher-valued properties are significantly more likely to purchase flood
insurance (Browne and Hoyt, 2000; Hung, 2009; Shao et al., 2017; Darlington and Yiannakoulias, 2022). Likewise,
living within a 100-year floodplain designated by the Federal Emergency Management Agency (FEMA) correlates
well with increased insurance uptake (Brody et al., 2017; Netusil et al., 2021; of Richmond, 2024). However, objective
risk factors cannot translate into action without the consideration of subjective risk perception. Personal experience,
particularly recent exposure to flood events, can temporarily elevate perceived risk and increase insurance adoption
(Petrolia et al., 2013; Lawrence et al., 2014; Royal and Walls, 2019), though this effect typically fades within a few years
as risk salience declines (Atreya et al., 2015; Kousky, 2017). Additionally, insights from behavior economics explain
why many high-risk individuals remain uninsured. Cognitive biases such as short-termism, inertia, and bandwagon
effect frequently contribute to suboptimal insurance decision-making (Kunreuther, 2021; LaCour-Little et al., 2025).
These behavior patterns are not uniform across populations but are shaped by socio-demographic factors, including age,
gender, race, and education, which influence how individuals perceive, interpret, and respond to flood risk (Atreya et al.,
2015; Shao et al., 2017, 2019; Eryılmaz Türkkan and Hırca, 2021). Despite these advancements, existing survey-based
models remain limited in scope, covering only a narrow subset of influential features. Moreover, these models rely on
an oversimplified representation of complex human behavior as a collection of discrete variables, thereby failing to
capture the subtle cognitive processes and context-dependent perceptions underlying real-world decision-making.
Recent advances in large language models (LLMs) have opened new avenues for simulating human behavior across
a range of decision-making scenarios. State-of-the-art LLMs, such as ChatGPT (OpenAI, 2024), Llama (Grattafiori
et al., 2024), DeepSeek (Guo et al., 2025), and Gemini (Comanici et al., 2025) series, have demonstrated remarkable
capabilities in language understanding (Karanikolas et al., 2023; Min et al., 2023), logical reasoning (Creswell et al.,
2022; Xu et al., 2025), and in-context learning (Liang et al., 2025a; Liu et al., 2025; Liang et al., 2025b). These
capabilities have facilitated the development of LLM-empowered agents, which are autonomous entities that integrate
modules for planning, memory, and tool use to emulate perception, reasoning, communication, and decision-making
in human behavior. Several recent studies have illustrated the application of LLM-empowered agents in modeling
human decision-making within domain-specific scenarios. For example, Li et al. (2023) proposed EconAgent, which
incorporates profile, memory, and action modules to simulate household behavior in labor and consumption decisions.
Wang et al. (2025) developed heterogenous agent profiles to replicate user behavior in movie recommender systems,
successfully capturing the user conformity phenomenon. Yang et al. (2025) employed the Belief-Desire-Intention
(BDI) framework to structure agent cognitive processes, enabling the simulation of investor behavior in stock market
environments. Chen et al. (2023) designed agents for sequential decision-making in dynamic auction settings, exhibiting
skills such as budget management, strategic planning, and goal adherence. Xu and Fan (2025) developed agents to
simulate household interpretations of pre-disaster warning information and assess their underestimation of hurricane
food shortages. Chen et al. (2025) introduced a behavioral theory-informed agent to simulate individual decisions
during wildfire evacuation. These agents serve as digital sandboxes for simulating, observing, and analyzing individual
and collective behaviors within a society.
This study investigates the potential of LLMs in simulating individual flood insurance decision-making. The Llama-3.3
70B model is selected for its open-source accessibility and deployment scalability. To facilitate performance assessment,
Section 2 constructs a benchmark dataset based on Shao et al. (2017) to capture marginal and bivariate probability
distributions of insurance purchase across key features. Using this dataset, Section 3 evaluates LLM’s common sense,
revealing a qualitative grasp of factors but an inability to produce accurate quantitative estimates. To address this
limitation, Section 4 proposes InsurAgent, an LLM-empowered agent to emulate individual insurance decisions. The
agent comprises five modules: perception, retrieval, reasoning, action, and memory. Specifically, the perception module
interprets user profiles and extracts relevant information. The retrieval module employs retrieval-augmented generation
(RAG) to ground probability estimations in empirical survey data. The reasoning module leverages LLM’s common
sense to extrapolate beyond survey data. The action module integrates the reasoning results to generate purchase
2

APREPRINT- NOVEMBER5, 2025
probabilities. The memory module supports sequential and temporal decision modeling. Experimental results in
Section 5 show that InsurAgent closely aligns with benchmark probabilities for both marginal and bivariate probability
estimations, consistently outperforming state-of-the-art LLMs. It also demonstrates strong extrapolation capabilities,
capturing contextual information such as resident cities, social environment, and prior experience. The effectiveness of
the memory module is validated through a simulated “three-up, three-down” life trajectory.
2 Dataset Preparation
To assess the performance of LLM in simulating individual flood insurance decisions, a benchmark dataset is constructed
based on the regression model reported in Shao et al. (2017). The original study analyzed survey responses from U.S.
Gulf Coast residents to identify factors influencing the voluntary flood insurance purchase. These factors span four
categories: sociodemographic characteristics, home ownership, distance from the coast, and perceived flood-related risk.
Particularly, sociodemographic variables include age, gender, education, and income. Each variable was discretized into
categorical levels and then a mixed-effects logit regression model was fitted to quantify their effects. The results reveal
four tiers of statistical significance: high ( p <0.001 ) for education, income, and distance from the coast; medium
(0.001≤p <0.01 ) for home ownership; low ( 0.01≤p <0.05 ) for perceived flooding amount and perceived hurricane
strength; and minimal ( p≥0.05 ) for age, gender, perceived hurricane number, and belief in climate change, where p
denotes thep-value used in hypothesis testing. For more details, readers are referred to Shao et al. (2017).
Table 1: Marginal probabilities of purchasing insurance by individual-level features (derived from Shao et al. 2017).
Individual-level features Code Marginal probability Individual-level features Code Marginal probability
Socio-demographic Home Ownership**
AgeOwn - 0.269
18–24 1 0.233 Rent - 0.113
25–34 2 0.238Distance from the Coast***
35–44 3 0.242 On the water 1 0.394
45–54 4 0.247 Near the water 2 0.340
55–64 5 0.251 Within 2-5 miles 3 0.290
65+ 6 0.256 5-10 miles 4 0.242
Gender11-30 miles 5 0.201
Female 1 0.242 31-60 miles 6 0.165
Male 0 0.262 > 60 miles 7 0.133
Education*** Risk Perceptions
Less than high school 1 0.168Flood Amount*
High school degree 2 0.203 Decreased -1 0.220
Some college 3 0.244 About the same 0 0.248
College degrees 4 0.288 Increased 1 0.278
Income*** Hurricane Number
Under $10,000 1 0.128 Decreased -1 0.236
$10–19,999 2 0.149 About the same 0 0.250
$20–29,999 3 0.174 Increased 1 0.265
$30–39,999 4 0.201Hurricane Strength*
$40–49,999 5 0.232 Decreased -1 0.212
$50–74,999 6 0.265 About the same 0 0.241
$75–99,999 7 0.301 Increased 1 0.273
$100,000+ 8 0.340Belief in Climate Change
Happening 1 0.255
Not happening 0 0.233
Statistical significance: *** High, ** Medium, * Low, and Minimal.
While the original study reports the feature distributions of the surveyed population and the fitted mixed-effects logit
regression model, it does not provide insurance purchase probabilities by individual-level features. To address this
gap, a Monte Carlo simulation is performed to reconstruct these probabilities. Specifically, a synthetic population of
10 million individuals is first generated, with each individual’s attributes independently sampled from the reported
feature distributions. These profiles are then input into the mixed-effects logit regression model to compute the
log-odds of purchasing insurance, which are converted to purchase probabilities using the logistic sigmoid function. To
3

APREPRINT- NOVEMBER5, 2025
simulate binary purchase behavior (purchase or no-purchase), Bernoulli trials are applied to each computed probability.
By aligning the simulated population-level purchase rate with the survey-reported value, this procedure effectively
reproduces the insurance purchase decisions of the population.
To quantify the impact of specific features on insurance purchase decisions, the simulated population and purchase
outcomes are analyzed in terms of marginal and bivariate probability distributions. The marginal probability distribution
captures the influence of individual factors, where the purchase probability for each categorical level of a given
variable is computed as the proportion of individuals with that attribute who purchased insurance. This reflects the
marginal effect of each feature on insurance decisions, as summarized in Table 1. In addition, bivariate probability
distributions are constructed for pairs of variables with differing levels of statistical significance (e.g., high vs. medium,
high vs. low, medium vs. minimal). For each pair, all combinations of categorical levels are enumerated, and the
corresponding bivariate purchase probabilities are computed. Together, the marginal and bivariate probabilities establish
a comprehensive benchmark for evaluating the ability of LLMs to simulate human decision-making in flood insurance
purchase.
3 Evaluation of LLM’s Capability in Flood Insurance Decision-Making
Using the benchmark dataset, this section evaluates the capability of LLMs to simulate individual decision-making
in purchasing flood insurance. The assessment is conducted from two aspects: qualitative and quantitative. For the
qualitative assessment, a third-person discriminative task is designed to test whether the LLM demonstrates a common
sense of influencing factors. This is achieved by instructing LLM to rank the purchase probabilities of various individual
profiles. For the quantitative assessment, a first-person role-playing task is designed where the LLM is prompted to
adopt the role of a specific individual with a given background and to estimate the probability of purchasing insurance.
Collectively, these two tasks enable a systematic assessment of the LLM’s capacity to model individual flood insurance
decisions.
3.1 Qualitative assessment of factors
A series of discriminative tasks are constructed to evaluate whether LLMs possess qualitative knowledge of factors
influencing individual flood insurance decisions. Each task isolates a single factor by presenting multiple individual
profiles that are identical in all attributes except the factor under evaluation. The LLM is prompted to rank these profiles
in descending order of their probability to purchase flood insurance from a third-person perspective. This experiment
directly assesses the LLM’s understanding of key decision variables, providing qualitative insights into its capacity to
simulate human behavior in purchasing flood insurance.
The results from the third-person discriminative tasks indicate that the Llama-3.3 70B model exhibits a strong qualitative
understanding of all influencing factors. Fig. 1 presents two illustrative examples of correct responses, in which the
LLM evaluates the impact of age and education. In both cases, the LLM not only produces accurate rankings of
individuals’ probability to purchase flood insurance but also provides sound reasoning for its conclusions. For age,
the LLM correctly recognizes that older individuals are more likely to purchase flood insurance due to greater asset
accumulation and heightened risk awareness. Similarly, for education, the LLM infers that individuals with higher
educational attainment demonstrate stronger financial literacy and a better understanding of insurance benefits, leading
to a higher propensity to purchase flood insurance. These results suggest that the LLM effectively identifies key factors
driving individual differences in decision-making and captures the impact of these factors. This confirms that Llama-3.3
70B model has qualitative knowledge required to understand flood insurance purchase decisions.
3.2 Quantitative estimation of probabilities
The preceding subsection has confirmed that the Llama-3.3 70B model possesses a qualitative understanding of factors
influencing flood insurance decisions. Building on this finding, this subsection aims to answer a more challenging
question: can LLM leverage these knowledge to simulate human behavior from a quantitative perspective? To explore
this, a first-person role-playing task is designed in which the LLM is assigned a specific user profile and prompted
to assume that role in deciding whether to purchase flood insurance. The LLM is instructed to output a purchase
probability, followed by a concise explanation. Unlike the third-person discriminative tasks, which involve comparative
evaluation across multiple individual profiles, the role-playing task requires the LLM to generate a context-specific
probability prediction based on a single user profile. Success in this task depends on the LLM’s ability to activate
domain knowledge and translate qualitative understanding into quantitative probabilities.
The results from the first-person role-playing tasks reveal a significant failure of Llama-3.3 70B model in simulating
human decision-making. For each individual profile, the LLM is prompted 20 times to estimate the probability of
4

APREPRINT- NOVEMBER5, 2025
Discriminative Tasks from a Third -Person Perspective: 
Evaluating LLMs’ knowledge of factors influencing individual flood insurance purchase decisions
Todetermine theprobability ofpurchasing flood insurance
across individuals, key factors such as age, financial stability,
risk perception  must  be considered.  Living inhouston, which is
known foritsflood risks (e.g.,hurricane harvey in2017), all
these individuals arepresumably aware ofthepotential for
flooding .However, their probability topurchase flood
insurance may vary based ontheir age:
•Younger individuals (20,30years old) might belesslikely to
purchase flood insurance duetolower perceived risk, fewer
assets toprotect, orbudget constraints .
•Middle -aged individuals (40,50years old) arelikely ina
better financial position andmay have agrowing awareness
oftheimportance ofprotecting their assets ,making them
more inclined topurchase insurance .
•Older adults (60,70years old), with potentially more assets
atriskandagreater appreciation forriskmanagement ,might
beeven more likely toinvest inflood insurance .
Considering these factors, theranking from most likely toleast
likely topurchase flood insurance would be:F>E>D>C>
B>A. Thisorder reflects thegeneral trend that asindividuals
age, they aremore likely tohave thefinancial means andthe
inclination tomitigate risks through insurance purchases .Llama3.3- 70B Responses
    
Considering thedistinct background profiles provided below,
please rank theindividuals inorder oftheir probability to
purchase flood insurance ,from most likely toleast likely :
A:Youare20years oldandyouareliving inHouston.
B:Youare30years oldandyouareliving inHouston.
C:Youare40years oldandyouareliving inHouston.
D:Youare50years oldandyouareliving inHouston.
E:Youare60years oldandyouareliving inHouston.
F:You are70years oldandyouareliving inHouston.
Please provide  your response  in the following  format : for 
example,  C > B > A, where  the order  reflects  decreasing  
probability  of purchasing  flood insurance . 
Question 1: Effect of Age
    
Education level  serves  as a key differentiator,  reflecting  
differences  in financial  literacy,  income  potential,  and risk 
awareness . Given that allindividuals live inhouston, anarea
known foritsflood risks, education becomes  the primary  basis  
for comparison:
•Bachelor's degree :Individuals with higher education levels
tend tohave better -paying jobs, more assets toprotect , and 
greater  awareness  of insurance  benefits .
•Community college attendee s:They  may have adecent
understanding offinancial knowledge  and asset  accumulation , 
making insurance  a reasonable consideration.
•Highschool diploma :Individuals with basic education  may
still recognize theimportance offlood insurance, especially
living inanarea like houston, butcould beless likely to
prioritize itcompared tothose with more education .
•Leftschool after 9thgrade :Individual with less formal
education have lower financial literacy andlessstable income ,
making them lesslikely topurchase flood insurance .
Considering these factors, theranking topurchase flood
insurance would be:D>C>B>A. This reflects  the general  
trend  that higher  educational  attainment  enhances  both the 
capacity  and inclination  to invest  in risk mitigation .Llama3.3- 70B Responses
    
Considering thedistinct background profiles provided below,
please rank theindividuals inorder oftheir probability to
purchase flood insurance ,from most likely toleast likely :
A:You areliving inHouston andyouleftschool after 9thgrade .
B:You areliving inHouston and you have ahigh school
diploma .
C:You areliving inHouston andyouareacommunity college
attendee.
D:Youareliving inHouston andyouhave abachelor degree.
Please provide your response inthefollowing format :for
example, C>B>A,where theorder reflects decreasing
probability ofpurchasing flood insurance .
Question 2: Effect of Education
    
Conclusion:  Third -person  perspective evaluations  reveal  that LLMs  demonstrate  a qualitative  understanding  of factors  
influencing  individual  decisions  to purchase  flood  insurance
Fig. 1. 
Fig. 1: Illustration of correct responses of LLM in the third-person discriminative tasks: LLM demonstrates a qualitative
understanding of factors influencing individual flood insurance decisions.
purchasing flood insurance. The average results are reported in Table 2, and two key patterns emerge. First, LLM
exhibits a strong risk-averse bias, consistently generating high purchase probabilities, typically at or above 0.800.
Second, the LLM demonstrates limited sensitivity to variations in individual-level features. For factors such as gender,
education, and income, LLM fails to differentiate between subgroups, producing identical outputs across distinct profiles.
For other factors including age, home ownership, distance from the coast, and risk perception, the LLM correctly
identifies the impact of factors but fails to adjust the probability magnitude accordingly. Fig. 2 illustrates this failure
using examples of age and education. It shows that while LLM generates contextually appropriate reasoning, such
as individuals with higher education are more likely to have stable income, it consistently assigns the same purchase
probability of 0.800 across cases. This highlights a critical knowledge-to-action gap: while the LLM demonstrates a
qualitative understanding of factors influencing flood insurance decisions, it fails to translate this understanding into
quantitative probability estimations. Therefore, this paper attempts to leverage the qualitative understanding to develop
LLM-empowered agents that can accurately estimate probabilities.
5

APREPRINT- NOVEMBER5, 2025
Fig. 3. Generation Tasks from a First -Person Perspective: 
Instruct LLMs to role -play individuals for simulating flood insurance purchase decisionsFactor: Education
Question:  You are living  in Houston,  and you have  
a bachelor  degree.  You are deciding  whether  to 
purchase  the flood  insurance . Please  state your 
purchase  probability  as a percentage.  Only  output  the 
percentage,  followed  by a concise explanation .
Response:  I am likely  to purchase  flood  insurance  
because Houston is a flood -prone  area,  and having  a 
bachelor’s  degree implies  I have  a stable  income  to 
afford  the premiums . Therefore,  my purchase  
probability  of the flood insurance  is 80.0%.
Question:  You are living  in Houston,  and you left 
school  after 9th grade . You are making a decision  
regarding  whether  to purchase  the flood  insurance . 
Please  state your purchase  probability  as a percentage.  
Only  output  the percentage,  followed  by a concise  
explanation .
Response:  I have  seen the devastating  effects  of 
flooding  in Houston,  and without  a high school  
diploma,  I do not have  a stable,  high-paying  job, so I 
want  to protect  my home  and belongings  from  flood 
damage.  Therefore,  my purchase  probability  of the 
flood insurance  is 80.0%.
Factor: Age
User Profile : You are 60 years  old, and you are 
living  in Houston.  You are deciding  whether  to 
purchase  the flood  insurance .Please  state your 
purchase  probability  as a percentage.  Only  output  the 
percentage,  followed  by a concise explanation .
Response:  I am likely  to purchase  flood  insurance  
because Houston is prone  to flooding,  and at my age, I 
want  to protect  my assets and ensure  financial  security . 
Therefore,  my purchase  probability  is 80.0%.
User Profile : You are 30 years  old, and you are 
living  in Houston.  You are making a decision  
regarding  whether  to purchase the flood 
insurance .Please  state your purchase  probability  as a 
percentage . Only  output  the percentage,  followed  by a 
concise explanation .
Response:  I live in Houston,  a city prone  to 
flooding,  so purchasing  flood  insurance  seems  like a 
necessary  precaution to protect  my assets . Therefore,  
my purchase  probability  is 80.0%.
Conclusion:  When role-playing  from  a first-person  perspective,  LLMs  assig n identical  probabilities  across  diverse 
profiles , indicating  an inability  to translate  qualitative  knowledge  into quantitative  estimates .
Fig. 2: Illustration of LLM failure in first-person role-playing tasks: LLMs fail to capture the differences in insurance
purchase probabilities across individual profiles.
Table 2: Marginal probabilities of purchasing flood insurance estimated by Llama-3.3 70B model.
Individual-level features Marginal probability Individual-level features Marginal probability
Estimated Benchmark Estimated Benchmark
Socio-demographic Home Ownership**
AgeOwn 0.850 0.269
18–24 0.800 0.233 Rent 0.815 0.113
25–34 0.800 0.238Distance from the Coast***
35–44 0.800 0.242 On the water 0.900 0.394
45–54 0.800 0.247 Near the water 0.900 0.340
55–64 0.800 0.251 Within 2–5 miles 0.900 0.290
65+ 0.830 0.256 5–10 miles 0.800 0.242
Gender11–30 miles 0.800 0.201
Female 0.800 0.242 31–60 miles 0.800 0.165
Male 0.800 0.262 > 60 miles 0.800 0.133
Education*** Risk Perceptions
Less than high school 0.800 0.168Flood Amount*
High school degree 0.800 0.203 Decreased 0.200 0.220
Some college 0.800 0.244 About the same 0.200 0.248
College degrees 0.800 0.288 Increased 0.900 0.278
Income*** Hurricane Number
Under $10,000 0.800 0.128 Decreased 0.200 0.236
$10–19,999 0.800 0.149 About the same 0.800 0.250
$20–29,999 0.800 0.174 Increased 0.900 0.265
$30–39,999 0.800 0.201Hurricane Strength*
$40–49,999 0.800 0.232 Decreased 0.200 0.212
$50–74,999 0.800 0.265 About the same 0.800 0.241
$75–99,999 0.800 0.301 Increased 0.900 0.273
$100,000+ 0.800 0.340Belief in Climate Change
Happening 0.900 0.255
Not happening 0.200 0.233
Statistical significance: *** High, ** Medium, * Low, and Minimal.
6

APREPRINT- NOVEMBER5, 2025
4 InsurAgent: An LLM-empowered Agent for Insurance Decision Simulation
To address the knowledge-to-action gap observed in role-playing tasks, an LLM-empowered agent, InsurAgent, is
proposed to simulate individual decision-making in flood insurance purchase. The core design philosophy of InsurAgent
draws on the heuristic decision-making paradigm, which indicates that individuals often consult empirical data or the
experiences of similar others to inform their decisions, particularly in scenarios involving risk and uncertainty (Simon,
1955; Payne et al., 1992). Inspired by this cognitive pattern, InsurAgent integrates region-specific survey data that
provide population-level purchase probabilities associated with key factors (Table 1). These data serve as empirical
baseline, enabling the agent to calibrate its understanding of risk and probability. By aligning the agent’s reasoning
with real-world statistics, InsurAgent is expected to produce accurate, personalized, and context-specific simulations of
flood insurance decisions.
4.1 Agent architecture
The InsurAgent is designed with five core modules: perception, retrieval, reasoning, action, and memory, as illustrated
in Fig. 3. This modular design enables a structured, multi-stage decision-making that mimics human cognition process.
The procedure begins with the perception module, which parses the input user profile and extracts two types of
information: (a) factors that correspond to variables presented in the survey data, and (b) unique personal details not
explicitly captured by predefined variables. This architecture ensures that both statistical indicators and contextual
information are available to inform downstream decision-making. The extracted factors are then passed to the retrieval
module, which interfaces with a region-specific survey database. Through semantic similarity search, the retrieval
module identifies factor-specific statistics that reflect population-level flood insurance purchase probabilities. These
empirical data serve as reference points to guide the LLM’s reasoning.
Fig. 4. 
You are a 33-year-old male . You 
are living  in Houston . You hold a 
Bachelor  of Arts in Economics  
from  a top-tier public  university . 
You earn an annual  salary  of 
$52,000 as a data analyst  at a 
national  market -research  firm.User 
Profile
Insur
AgentThis estimate is based  on a 
combination  of factors . As a 33-year-
old, the survey  data suggests a 
moderate influence  from  age (23.8% 
purchase  rate for individuals  aged  25-
34), but this factor  has minimal  
impact . Being  male , the baseline  
purchase  rate is 26.2. However,  
education  (Bachelor  degree: 28.8%) 
and income  ($50,000-$74,999: 26.5%) 
have  higher  importance  and suggest  a 
slightly  higher  propensity  to purchase  
flood  insurance . Given  the location  in 
Houston , which  is within  Gulf Coast  
and thus prone  to flooding,  these  
factors  collectively  contribute  to an 
estimated  probability  of 27.5%, 
adjusted  for individual  variability  and 
local  risk considerations .
Parse  the user profile :
•Extract  key factors  (age, gender,  
education,  income,  etc.)
•Identify  unique  or idiosyncratic  
personal  detailsPerception
Retrieved
Evidence
Task GoalChain 
of ThoughtGuidance  
Notes
Role
PlayingRole
Definition Mimic  human decision- making :
•Refer  to relevant  data
•Weight  the influence  of factors
•Translate contextual  information  
into probability  adjustments
Acquire external  knowledge :
•Create a vector  database with survey 
data on influencing  factors
•Retrieve factor -specific survey data 
through semantic similarity  searchRetrieval
Reasoning
Generate an output :
•Report  the probability  of purchasing 
insurance  as a percentage
•Provide  a concise explanation  for 
the estimationAction
Maintain temporal  history :
•Archive episodic  events  with 
time stamps  
•Give  permissions  to Perception  
and Retrieval  to access  memoriesMemory
Fig. 3: Architecture of the InsurAgent, comprising five modules: perception, retrieval, reasoning, action, and memory.
The retrieved survey data, along with the parsed output from the perception module, are fed into the reasoning module,
which serves as the cognitive core of the InsurAgent. Herein, the agent performs a first-person role-playing task to
emulate human decision-making. The reasoning process unfolds in three stages. First, the agent organizes all retrieved
survey data and treats them as quantitative benchmarks. Second, it weighs the influence of each factor to derive a
baseline purchase probability. Third, the agent applies common sense to extrapolate the baseline probability based on
unique personal details, translating contextual information into quantitative probability adjustments. The reasoning
process of the InsurAgent is informed by a structured prompt template, which will be elaborated in the next subsection.
Following the reasoning process, the action module generates a final prediction of flood insurance purchase probability,
reported as a percentage between 0% and 100%. This estimation is accompanied by a concise explanation that
7

APREPRINT- NOVEMBER5, 2025
summarizes the rationale behind the decision, thereby improving the transparency and interpretability. To support
sequential decision tasks and cumulative behavioral modeling, the memory module is designed to archive the reasoning
trajectory and final probability output with temporal stamps, forming an episodic history. These memory traces can be
queried by the perception and retrieval modules in future interactions, allowing the agent to simulate evolving decisions
in response to life events or environment changes. An example of the functionality of the memory module is presented
in Section 5.4. Together, these five modules constitute the cognitive architecture of InsurAgent, enabling it to simulate
individual flood insurance decisions.
4.2 Prompt design
To support the structured reasoning of InsurAgent, a prompt template is proposed, as demonstrated in Fig. 4. This
template scaffolds the agent’s step-by-step decision-making process, guiding it toward producing realistic and context-
specific behaviors. It consists of five components: role definition, retrieved evidence, task goal, chain of thought, and
guidance notes, each of which plays a distinct role in shaping the agent’s generative behavior. Specifically, the role
definition component instantiates the agent’s persona by instructing it to role-play an individual deciding whether to
purchase flood insurance. This activates the LLM’s domain knowledge of flood-related decision-making and ensures a
consistent first-person narrative throughout the response.
Fig. 5. Prompt Template for Role Playing:
Retrieved Evidence
Males: 26.2% purchased flood insurance
Females: 24.2% purchased flood insurance
Chain of Thought
Step A: Identify  factors  that correspond to 
the provided survey  data.  Prioritize  these  
factors  by their statistical  significance: high,  
medium,  low, and minimal .
Step B: Estimate  a baseline probability  of 
purchasing  flood  insurance  based  on the 
combination of these influencing  factors . 
Step C: Refine  the estimate  by 
incorporating  unique  personal  details . 
Critically  evaluate all additional  
background details  that are not involved  in 
the survey  data.  
Step D: Report  your final purchase  
probability  as a percentage (0% -100% ). 
Role Definition 
You are an individual  deciding  whether  to 
purchase  flood insurance .
Task Goal
Objective: Perform  a holistic  analysis  to 
estimate  your purchase  probability . 
Your  evaluation  must  synthesize empirical  
survey  data with your unique  background . 
Guidance notes
•Rely  exclusively  on the given  
background to make  your decision . 
•The statistics  in the survey  data reflects  
population- level  average . Treat  it as a 
reference point,  not an absolute  rule. 
•Apply  the survey  data directly  for Gulf 
Coast  residents . Otherwise,  adjust  your 
estimations  to reflect  the local  flood risk. 
Guidance Notes
Role Definition 
Retrieved Evidence
Task Goal
Chain of Thought
Additional  documents : 
Survey  data from  the U.S. Gulf Coast  
to quantify  how key factor  influence  
flood insurance  purchase  decisionsEncodeEmbedding 
Model
Vector Database
User Profile :
You are a male , and you are living  in 
Houston.  You are deciding  whether  
to purchase  the flood insuranceSemantic 
Search
Similarity
RankRetrieval Module:
EncodeIndex
Fig. 4: Prompt template used by InsurAgent for role-playing tasks, where empirical survey data on factors are retrieved
from an external database and integrated into the system prompt.
The retrieved evidence component is dynamically populated based on the user profile. As described in the previous
section, this retrieve process acquires relevant statistics from survey data collected in the U.S. Gulf Coast region.
Specifically, the survey data used here comprise population-level flood insurance purchase probabilities associated with
10 factors, as listed in Table 1. The detailed procedures for implementing RAG are as follows. Survey documents are
first encoded into embeddings and then factor-specific statistics are indexed and stored as a vector database. When a user
profile is presented, the perception module extracts relevant factors, which are then embedded and semantically matched
against the database to retrieve corresponding statistics. These retrieved benchmarks are inserted into the prompt as
external references to inform reasoning. The task goal component illustrates the primary objective: to perform a holistic
analysis that synthesizes empirical data with individual background to estimate the probability of purchasing flood
insurance.
8

APREPRINT- NOVEMBER5, 2025
At the core of the agent’s cognitive workflow is the chain of thought (CoT) component, which guides reasoning process
through four stages, as demonstrated in Fig. 4. (a) Factor identification and prioritization: The agent identifies relevant
factors from the user profile and prioritizes them based on statistical significance levels, as established in Section 2. (b)
Baseline estimation: Using the retrieved evidence for these factors, the agent estimates a baseline purchase probability
by assessing their relative influence. (c) Adjustment according to contextual information: The agent applies its common
sense to interpret personal details, adjusting the baseline probability upwards or downwards. (d) Final integration: The
agent integrates the baseline and adjustments to generate the final purchase probability as a percentage.
The guidance notes component imposes constraints on the agent’s generative behavior. First, it instructs the agent to rely
exclusively on the provided background to reduce hallucination and enhance reproducibility. Second, it emphasizes that
the retrieved statistics represent population-level averages used as reference points rather than absolute rules, allowing
the agent to flexibly adjust these empirical benchmarks based on the specific user profile. Third, it specifies that the
empirical benchmarks are geographically specific. For users located outside the Gulf Coast region, the agent should
apply domain knowledge to adjust probability estimates in accordance with local flood risk. Collectively, these five
components transform the LLM from a general-purpose text generator into a specialized agent capable of performing
complex behavior simulations.
Fig. 6. Marginal probabilities of purchasing flood insurance  simulated by InsurAgentFactor: Education
User Profile : You are living  in Houston,  and you 
have  a bachelor  degree.  You are deciding  whether  to 
purchase  the flood  insurance . Only  output  purchase  
probability,  followed  by a concise explanation .
Response:  This is based  on the survey  data 
indicating  that individuals  with a college degree  have  a 
28.8% probability  of purchasing  flood  insurance,  and 
since I live in Houston,  which  is in the Gulf Coast , and 
have  a bachelor’s  degree,  my purchase  probability  
aligns  with this average.
User Profile : You are living  in Houston,  and you 
left school  after 9th grade . You are deciding  whether  
to purchase the flood  insurance . Only  output  purchase  
probability,  followed  by a concise explanation .
Response:  I am living  in Houston,  which  is in the 
Gulf Coast,  so I will use the survey  data as my 
baseline.  Since  I left school  after 9th grade,  I have  a 
less than high school  degree,  and according  to the 
survey,  individuals  with this education  level  have  a 
16.8% probability  of purchasing  flood insurance .Factor: Age
User Profile : You are 60 years  old, and you are 
living  in Houston.  You are deciding  whether  to 
purchase  the flood  insurance .Only  output  purchase  
probability,  followed  by a concise explanation .
Response:  As a 60-year-old living  in Houston,  
which  is a city in the Gulf Coast,  my baseline  
probability  is 25.1% based  on the survey  data for 
individuals  aged  55-64. Given  no additional  
background details  that would differentiate me from  
the average person in this age group,  I will stick  with 
this average purchase  probability .
User Profile : You are 30 years  old, and you are 
living  in Houston.  You are deciding  whether  to 
purchase  the flood  insurance . Only  output  purchase  
probability,  followed  by a concise explanation .
Response:  I am using  the provided survey  data as 
my baseline,  and since I am 30 years  old, which  has a 
flood insurance  purchase  rate of 23.8%. Given  that I 
live in Houston,  a city in the Gulf Coast  area prone  to 
flooding,  I don’t  see a need  to adjust  this probability  
based  on location . Therefore,  my probability  of 
purchasing  flood insurance  remains  at 23.8%.
Fig. 5: Representative examples of InsurAgent-simulated marginal probabilities of purchasing flood insurance.
5 Results and discussion
5.1 Performance of InsurAgent in estimating marginal and bivariate probabilities
The performance of the proposed InsurAgent is assessed by comparing the estimated flood insurance purchase probability
against benchmark dataset across marginal and bivariate distributions. These scenarios are designed to evaluate the
agent’s ability to retrieve relevant data, follow quantitative references, and estimate the probabilities under combined
effects. For marginal probability estimations, each user profile contains only one variable. These cases assess the
agent’s capacity to align its predictions with empirical references, thereby minimizing the risk of hallucinations. For
bivariate probability estimations, user profiles are constructed to include two variables, which are selected to span
varying levels of statistical significance. These scenarios evaluate the agent’s capacity to synthesize multiple empirical
statistics to produce coherent predictions.
For marginal probability estimations, InsurAgent accurately predicts the empirical purchase probabilities retrieved from
the external survey database across all factors. Specifically, each user profile in this setting includes a single variable
along with a specified resident city within the U.S. Gulf Coast region. With no additional contextual information, the
agent is expected to retrieve the corresponding population-level statistics as the predicted probability. Fig. 5 illustrates
InsurAgent’s responses under two representative scenarios. In the case of age, the agent correctly retrieves baseline
probabilities for individuals aged 60 (25.1%) and 30 (23.8%). It explicitly states that, given no additional background
9

APREPRINT- NOVEMBER5, 2025
details, it would adhere to the average probability for the specified age group. Similarly, for the education factor, the
agent accurately retrieves the empirical probabilities for individuals with a bachelor’s degree (28.8%) versus those
who left school after 9th grade (16.8%). This behavior is cognitively plausible, mirroring how humans rely on average
trends to inform decisions in the absence of additional contextual information. These results affirm the effectiveness of
InsurAgent’s architecture: the retrieval module identifies the corresponding statistical data, and the reasoning module
adheres to prompt instructions to use this data as the decision basis.
The InsurAgent also exhibits strong alignment with benchmark dataset in estimating bivariate probabilities. In this
setting, each user profile has two variables along with a residential location within the U.S. Gulf Coast region. Since
the external database contains only marginal probabilities, this task requires the agent to integrate separate empirical
references and infer their combined effects through extrapolation. Note that the two variables are intentionally selected
from different levels of statistical significance, such as education (high) and perceived flooding amount (low). The
results show that the agent’s estimated purchase probabilities closely match those in the benchmark dataset, yielding a
correlation of determination (R2) of 0.778 and a mean absolute error (MAE) of 0.0240, as depicted in Fig. 6. The high
R2and low MAE values demonstrate that InsurAgent effectively integrates multiple empirical evidence through its
structured reasoning process.
0.00.10.20.30.40.5
0.0 0.1 0.2 0.3 0.4 0.5Probability estimated by InsurAgent
Probability in the benchmark datasetR2 = 0.778
MAE = 0.0240InsurAgent’s estimations of  bivariate probabilities
Fig. 7. 
Fig. 6: Comparison of bivariate probabilities estimated by InsurAgent and benchmark dataset.
In addition to producing accurate point estimates, the InsurAgent effectively captures the trend patterns consistent with
the benchmark, as shown in Fig. 7. These representative scenarios involve combinations of variables with varying levels
of statistical significance. Across both cases, the agent’s predicted trends closely align with those in the benchmark
dataset. The agent first establishes a baseline from the dominant variable and then refines the estimate using the
secondary variable. For instance, in the scenario combining distance from the coast with perceived flood amount,
major shifts are driven by changes in distance. Similarly, in scenarios involving education and gender, the primary
variable shapes the overall trajectory, while the secondary variable introduces adjustments. This behavior underscores
the agent’s capacity to execute chain-of-thought reasoning to integrate multiple pieces of evidence.
5.2 Comparison with state-of-the-art LLMs
To demonstrate the superior performance of the proposed agent, this section benchmarks InsurAgent against two
leading general-purpose LLMs: ChatGPT-4o and DeepSeek-V3. These models represent the current state-of-the-art
in general-purpose language modeling and are considered as strong baselines for comparison. The evaluation is
conducted across a series of marginal probabilities and bivariate probabilities. In each scenario, the LLM is provided
with a specified user profile and prompted to assume that role, engaging in first-person role-playing task to determine
the probability of purchasing flood insurance. The expected output includes a probability estimate (expressed as a
percentage), accompanied by a concise explanation reflecting the reasoning behind the decision.
For marginal probability estimations, both ChatGPT-4o and DeepSeek-V3 exhibit substantial deviations from benchmark
dataset, yielding R2values of -91.2 and -60.4 and MAE values of 0.537 and 0.428, respectively, as shown in Fig. 8
(a) and (b). Two critical discrepancies exist. First, both LLMs consistently overestimate purchase probabilities.
While benchmark probabilities average around 0.25, ChatGPT-4o and DeepSeek-V3 typically produce inflated values
exceeding 0.60 across most variables. This persistent upward bias suggests that, without reference to empirical
survey data, LLMs tend to take an overly risk-averse stance. Second, beyond quantitative inaccuracy, both models
demonstrate inconsistent and sometimes inverted qualitative reasoning. For instance, ChatGPT-4o estimates a flood
10

APREPRINT- NOVEMBER5, 2025
Fig. 8. 0.00.10.20.30.40.5
1 3 5 7 9 11 13 15 17 19 21Probability estimated by InsurAgent
Combinations of categorical variablesDistance*** & Perceived Flood Amount*
Dist.
On-water
InsurAgent
Benchmark
(a)Same
0.00.10.20.30.4
1 2 3 4 5 6 7 8Probability estimated by InsurAgent
Combinations of categorical variablesEducation*** & Gender
Edu. 
Below HSEdu.
HS degreesEdu. 
Some collegeEdu. 
College
Male
Female
Gender
R2 = 0.848
MAE = 0.0155InsurAgent
Benchmark
(b)Male
Female
Male
FemaleMale
Female
R2 = 0.928
MAE = 0.0195Dist.
N-waterDist.
2-5 miDist.
5-10 miDist.
11-30 miDist.
31-60 miDist.
60+ mi
Perceived Flood AmountInc.
Dec. SameInc.
Dec.SameInc.
Dec. SameInc.
Dec.SameInc.
Dec. SameInc.
Dec. SameInc.
Dec.
0.00.10.20.30.40.5
1 3 5 7 9 11 13 15 17 19 21Probability estimated by InsurAgent
Distance from the coastDistance*** & Perceived Flood Amount*
InsurAgent Benchmark
(a)Same
0.00.10.20.30.4
1 2 3 4 5 6 7 8Probability estimated by InsurAgent
EducationEducation*** & Gender
Male
Female
Gender:
Male/FemaleR2 = 0.848
MAE = 0.0155
(b)Male
Female
Male
FemaleMale
Female
R2 = 0.928
MAE = 0.0195Perceived Flood Amount :
Decreased/Same/IncreasedInc.
Dec. SameInc.
Dec.SameInc.
Dec. SameInc.
Dec. SameInc.
Dec. SameInc.
Dec. SameInc.
Dec.
On the 
waterNear the 
water2-5 mi 5-10 mi11-30 mi31-60 mi60+ mi Below HS HS degree Some college CollegeInsurAgent Benchmark
Fig. 7: InsurAgent-simulated bivariate probabilities of purchasing flood insurance: (a) distance from the coast and
perception of flood amount, and (b) education and gender.
Fig. 9. 0.00.20.40.60.81.0
0.0 0.2 0.4 0.6 0.8 1.0Probability estimated by DeepSeek -V3
Probability in the benchmark dataset0.00.20.40.60.81.0
0.0 0.2 0.4 0.6 0.8 1.0Probability estimated by ChatGPT -4o
Probability in the benchmark dataset0.00.20.40.60.81.0
0.0 0.2 0.4 0.6 0.8 1.0Probability estimated by ChatGPT -4o
Probability in the benchmark dataset0.00.20.40.60.81.0
0.0 0.2 0.4 0.6 0.8 1.0Probability estimated by DeepSeek -V3
Probability in the benchmark dataset
Performance of ChatGPT -4o
R2 = -48.3
MAE = 0.503Performance of DeepSeek -V3
(c) (d)R2 = -33.8
MAE = 0.418Performance of ChatGPT -4o
R2 = -91.2
MAE = 0.537Performance of DeepSeek -V3
(a) (b)R2 = -60.4
MAE = 0.428Marginal Probabilities
Bivariate Probabilities
Fig. 8: Comparison of marginal and bivariate probabilities estimated by state-of-the-art LLMs with benchmark dataset.
insurance purchase probability of 0.810 for individuals aged 25–34, yet assigns a lower value of 0.777 for those aged
35–44. Similarly, it predicts probability of 0.815 for individuals aged 45–54 but decreases to 0.787 for those aged
55–64. DeepSeek-V3 produces comparable inconsistencies, estimating probabilities of 0.712, 0.695, and 0.675 for
individuals with less than a high school education, a high school degree, and some college education, respectively.
These discrepancies reveal the limitations of general-purpose LLMs in behavioral simulation tasks and underscore the
critical role of LLM agents calibrated with domain data.
11

APREPRINT- NOVEMBER5, 2025
Performance deficiencies of the general-purpose LLMs are also observed in estimating bivariate probabilities, as
demonstrated in Fig. 8 (c) and (d). Consistent with marginal probability estimation, both models overestimate purchase
probability across all variables. This upward bias results in large deviations from benchmark dataset: ChatGPT-4o
yields an R2of -48.3 and an MAE of 0.503, while DeepSeek-V3 produces an R2of –33.8 and an MAE of 0.418. The
results again show that, without access to external references or a structured reasoning framework like InsurAgent,
these general-purpose LLMs struggle to reason and predict the insurance purchase probabilities.
5.3 Beyond regression: InsurAgent’s capacities to capture contextual information
A key advantage of LLM-empowered agents is that they process natural language inputs to parse and interpret contextual
information that is typically beyond the scope of regression models. Consequently, InsurAgent offers a flexible and
adaptable framework for simulating complicated human decision-making. To illustrate this unique capability, five
decision-making scenarios are constructed around factors not reflected in the survey, including residential cities,
occupations, social environments, flood experiences, and claim history. While these factors are not reflected in the
regression models, they can be conveyed through contextual information. The InsurAgent can utilize its common sense
to process the information and generate predictions.
Fig. 9. InsurAgent : Effects of Residential City on Individual Flood Insurance Purchase Behavior 
You are a 33-year-old male . You are living  in City. You hold a Bachelor  of Arts in Economics  from  a top-tier public  
university . You earn an annual  salary  of $52,000 as a data analyst  at a national  market -research  firm. You are deciding  
whether  to purchase  flood insurance .Only  output  purchase  probability,  followed  by a concise explanation .User 
Profile
Insur
AgentMiami : As acollege- educated  individual,  I reference the education  survey  data,  where 28.8% of individuals  purchased  
flood  insurance . Adjusting  for my income  ($52,000), which  indicates  a purchase  rate of 26.5%, I slightly  decrease my 
baseline probability . Considering  my age (33) and being  male , which  have  minimal  impact , I adjust  my probability  
downward  but prioritize  education  as the most  influential  factor . Living  in Miami  increases  my probability , but given  
the factors  provided,  I estimate  my purchase  likelihood  at 28.5%. 
Orlando : As acollege- educated  individual  with a Bachelor ’s degree,  I start with a baseline of 28.8% from  the 
education  survey  data.  Adjusting  for income  ($52,000), I consider  the 26.5% purchase rate. Given  my location  in 
Orlando  (not directly  on the Gulf Coast  but still in a flood- prone  state),  I slightly  decrease this rate due to lower  
perceived  risk compared  to direct  coastal  areas . My age and gender  have  minimal  impact  according  to the prioritization,  
so I only fine-tune based  on significant  factors  like education  and income,  resulting  in a final estimate  of 27.1%.
Atlanta : My education  level  (college  degree ) suggests  a 28.8% probability,  while  my income  ($50,000-$74,999 range)  
implies  a 26.5% probability . However,  since Atlanta  is not directly  in the Gulf Coast  area,  I adjust  these probabilities  
downward  to account  for the lower  flood  risk. Additionally,  being  male  (26.2% probability)  has a relatively  modest  
impact . Considering  my age (33) has a minimal  effect , I average  and adjust  these factors  to arrive  at a final estimated  
purchase  probability  of 21.9%.
Dallas : My estimated  purchase  probability  is relatively  low because,  as a resident  of Dallas,  I am not directly  exposed  
to the high flood  risk associated  with the Gulf Coast . Although  my education  level  (Bachelor ’s degree)  and income  
($52,000) suggest  a moderate to high probability  of purchasing  flood insurance  based  on the survey  data (28.8% and 
26.5%), my location  in a relatively  low-flood- risk reduces  my overall  probability . Additionally,  having  a moderate age 
(33 years  old) does not significantly  increase my probability . Considering  these factors,  I adjusted  my baseline  
probability  downward  to 17.7% to reflect  my lower  flood risk due to location .
Chicago : Since  I don’t  live in the Gulf Coast  area,  the provided  survey  data may not directly  apply  to my situation . 
However,  using  it as a reference point,  I note that my education  level  (Bachelor ’s degree : 28.8%) and income  
($50,000-$74,999: 26.5%) would  suggest  a higher  probability  of purchasing  flood  insurance  if I were in a high-risk 
area.  But given  Chicago’s  relatively  low flood risk compared  to the Gulf Coast , and considering  my age as minimal  
factors,  I adjust  my baseline probability  downward . Therefore,  I estimate  my purchase  probability  to be around 13.2%, 
similar  to the lowest  income  bracket,  reflecting  my relatively  low-risk location  and informed decision -making.
Phoenix : My purchase  probability  is low because I live in Phoenix,  which  is not prone  to flooding,  and the survey  data 
provided is based  on the Gulf Coast  region . As a male  with a college degree  and an annual  income  of $52,000, my 
demographics  suggest  a moderate to high propensity  to purchase  flood insurance  if I lived  in a flood- prone  area 
(around 26.2% for males  and 28.8% for those  with a college degree ). However,  given  the low flood  risk in Phoenix,  I 
adjust  my probability  downward  significantly , to 12.1%, reflecting  the reduced  need  for flood  insurance  in my location .Statistical  significance: Age; Gender; Education***; Income***. 
Fig. 9: InsurAgent-based simulation of the impact of residential cities on individual flood insurance purchase behavior.
(While there is no data to validate, but the trend is correct based on common sense or qualitative results of surveys.)
12

APREPRINT- NOVEMBER5, 2025
The city of residence plays a critical role in shaping individuals’ flood insurance decision because people living in
higher risk areas will have increased perception of risk exposure. However, existing regression models typically rely
on local survey data, failing to generalize to regions other than the one in the survey. The InsurAgent can overcome
this limitation by harnessing the common senses of LLMs. A series of individual profiles are constructed in which
all socio-demographic variables are held constant, i.e., a 33-year-old male with a bachelor’s degree and an annual
income of $52,000, while only the residential city varies: Miami, Orlando, Atlanta, Dallas, Chicago, and Phoenix.
After identifying age, gender, education, and income as four relevant factors and retrieving the corresponding statistics,
the InsurAgent recognizes the different locations and makes an extrapolated prediction. The purchase probabilities
for Miami, Orlando, Atlanta, Dallas, Chicago, and Phoenix are 28.5%, 27.1%, 21.9%, 17.7%, 13.2%, and 12.1%,
respectively. Detailed reasoning is shown in Fig. 9. The decreasing predictions match the downward trend in flood risk,
demonstrating the abilities of InsurAgent to make plausible extrapolation based on geographic locations.
The second scenario simulates the effects of occupations on the purchase of flood insurance. Two individual profiles are
constructed with identical socio-demographic characteristics: both are 35-year-old males with a bachelor’s degrees,
residing in Houston and earning $55,000 annually. The only difference lies in their professions. One individual holds
a degree in civil engineering and works as an underwriting analyst at a homeowners’ insurance firm, whose daily
work involves assessing property risk. The other holds a degree in chemistry and works as a research associate at a
petrochemical laboratory, with no direct engagement in insurance or flood-related work. Existing qualitative research
has shown that occupations influence how individuals perceive and respond to risk (Kouabenan, 2002; Hill et al.,
2019). Results show that InsurAgent effectively differentiates their occupations, as demonstrated in Fig. 10. InsurAgent
predicts that the person working in the insurance industry has better awareness of flood risk and insurance benefits,
leading to a significantly higher purchase probability of 51.6%. This matches the qualitative survey results.
Fig. 10. InsurAgent : Effects of Occupations on Individual Flood Insurance Purchase Behavior 
User Profile : You are a 35-year-old male  living  in Houston , Texas . You hold a Bachelor  of Civil  Engineering  from  the 
University  of Houston and earns  an annual  salary  of $55,000 as an underwriting  analyst  at a leading  homeowners’  
insurance  firm. In your day-to-day responsibilities,  you evaluate property  risk factors,  such as flood  zone  maps,  historical  
loss data,  and structural  vulnerabilities , and integrate  these assessments into actuarial  models .  
     InsurAgent  Response: As a 35-year-old male  living  in Houston,  Texas,  with a Bachelor  degree in Civil  Engineering 
and working  as an underwriting  analyst,  my profession  provides  me with a deep  understanding  of flood risks  and the 
importance  of insurance . My annual  salary  of $55,000 suggests  a moderate to high income  level,  which,  according  to the 
survey  data,  correlates  with probability  of purchasing  flood insurance  around  26.5%. Additionally,  my education  level  
(Bachelor’s  degree)  would further  increase this probability  (around 28.8%). Considering  my profession  and expertise in 
assessing  property  risk factors,  I am more  aware of the potential  risks  and benefits  of flood  insurance,  which  increases  my 
willingness  to purchase it. Given  these factors  and living  in a flood -prone  area like Houston,  my estimated  purchase  
probability  of 51.6% is significantly  higher  than baseline probabilities  suggested  by the survey  data.
User Profile : You are a 35-year-old male  living  in Houston , Texas . You hold a Bachelor  of Science in Chemistry  from  
the University  of Houston and earns  an annual  salary  of $55,000 as a research  associate at a petrochemical  laboratory . Your  
research  focuses  on catalyst  development  and process  optimization  to make  industrial  reactions  faster  and cleaner . You 
spend most  of your time conducting  experiments  and analyzing  reaction  mechanisms  in the laboratory .
     InsurAgent  Response: My estimated  probability  is based  on synthesizing  survey  data with my unique  background.  As 
an individual  with a college  degree  (28.8% baseline from  education  survey),  and earning  $55,000 annually  (slightly  above  
the $50,000-$74,999 range  with 26.5% purchase rate, my probability  is adjusted  upward  due to higher  income  and 
education  levels  but downward  due to age (35-44 years  old with 24.2% purchase rate) and the assumption that my technical  
background might  make  me risk-aware but not necessarily  more  inclined  towards  purchasing insurance  without  a direct  
experience of flooding . Living  in Houston,  Texas,  within  the Gulf Coast  area,  also supports  using  these survey  data directly  
for estimation . The combination of these factors  results  in a refined  probability  of 28.1%.
Statistical  significance: Age; Gender; Education***; Income***. 
Fig. 10: InsurAgent-based simulation of the impact of occupations on individual flood insurance purchase behavior.
(While there is no data to validate, but the trend is correct based on common sense or qualitative results of surveys.)
The third scenario simulates how individuals decide to purchase flood insurance within distinct social environments.
This is important because individuals’ decisions are not made in isolation but shaped by interpersonal interactions
and social influence. Herein, three individual profiles are constructed with identical socio-demographic attributes: a
30-year-old female with a bachelor’s degree, employed as a marketing specialist in Atlanta with an annual salary of
$62,000. The only difference is the proportion of insured peers within their social circles, which is 75%, 50%, and 5%,
respectively. As shown in Fig. 11, InsurAgent’s predicted probabilities correlate well with the insurance purchase across
these social environments. The agent first retrieves factor-specific statistics to establish a baseline probability and then
13

APREPRINT- NOVEMBER5, 2025
incorporates the proportion of insured peers as contextual information to extrapolate its prediction. As the proportion of
insured peers declines from 75% to 50% and then to 5%, the agent infers progressively lower purchase awareness, with
predicted probabilities decreasing from 32.9% to 28.5% and ultimately to 24.8%. This scenario highlights the agent’s
capacity to integrate social influence into individual predictions, laying the foundation for modeling emergent group
behaviors in multi-agent systems.
Fig. 11. InsurAgent : Effects of Social Environment on Individual Flood Insurance Purchase Behavior 
User Profile : You are a 30-year-old female  residing  in Atlanta , Georgia.  You hold a Bachelor  of Arts in Sociology  from  
Georgia State  University  and earn an annual  salary  of $62,000 as a marketing  specialist  at a technology  firm. Within  your 
social  network , composed of former  classmates,  neighborhood friends,  and professional  contacts,  approximately  75%  carry  
flood insurance , routinely  sharing  policy  recommendations  and claims  experiences .
     InsurAgent  Response: As a college -educated  individual  with a Bachelor’s  degree,  the survey  data suggests a 28.8% 
purchase  rate. Given  my annual  income  of $62,000, the survey  indicates  a 26.5% purchase  rate. Considering  these  two 
factors,  I adjust  my baseline probability  upward . However,  being  a female  (24.2% purchase  rate), this factor  has a minimal  
impact  on the overall  estimate . The significant  influence comes  from  my social  circle,  where 75% of my peers  have  flood  
insurance,  likely  increasing  my awareness  and perceived  importance  of having  such coverage . Nonetheless,  since Atlanta  is 
not directly  on the Gulf Coast,  the local  flood  risk might  be lower  compared  to the survey’s  context,  which  could slightly  
decrease  my probability  of purchasing  flood insurance . Balancing  these factors  results  in 32.9% probability .
User Profile : You are a 30-year-old female  residing  in Atlanta , Georgia.  You hold a Bachelor  of Arts in Sociology  from  
Georgia State  University  and earn an annual  salary  of $62,000 as a marketing specialist  at a technology  firm. Within  your 
social  network , composed  of former  classmates,  neighborhood  friends,  and professional  contacts,  approximately  50% carry  
flood insurance , a prevalence that you have  observed  through regular  conversations .
     InsurAgent  Response: This estimated  probability  is based  on a combination of factors  from  the survey  data and 
personal  details . Starting  with education  (Bachelor’s  degree),  the baseline probability  would be around 28.8%. Adjusting  
for income  ($62,000 falls into the $50,000-$74,999 range),  the probability  increases  slightly  to around 29%. However , 
since Atlanta  is not directly  in the Gulf Coast  area,  the flood  risk might  be lower,  decreasing  the probability . Additionally,  
having  50% of social  circle members  with flood  insurance  could influence the decision  positively  but given  other  factors,  it 
slightly  adjusts  the probability  downward . Therefore,  a holistic  consideration leads  to an purchase  probability  of 28.5%.
User Profile : You are a 30-year-old female  residing  in Atlanta , Georgia.  You hold a Bachelor  of Arts in Sociology  from  
Georgia State  University  and earn an annual  salary  of $62,000 as a marketing specialist  at a technology  firm. However,  in 
your immediate  network , comprised  mainly  of family  members,  long-time friends,  and close colleagues,  only about  5% 
carry  flood insurance , viewing  it as a low-priority  expense.
     InsurAgent  Response: As an individual  with a college degree  (Bachelor  of Arts),  the baseline probability  starts  at 
28.8%. However,  considering  my annual  income  ($62,000),  the purchase  rate is 26.5%, which  slightly  adjusts  the baseline 
downward . The significant  influence  comes  from  my education  and income  levels . Additionally,  living  in Atlanta , Georgia,  
which  is not directly  within  the Gulf Coast  area where the survey  was conducted,  might  reduce  my perceived  need  for flood 
insurance  due to potentially  lower  local  flood  risk compared  to coastal  areas . Lastly,  the low rate of flood  insurance  among 
my immediate  network  (5%) may reflect  a cultural  or social  influence  that slightly  decreases  my probability  of purchasing  
flood insurance . Considering  these factors,  the final purchase  probability  is adjusted  to 24.8%.
Statistical  significance: Age; Gender; Education***; Income***. 
Fig. 11: InsurAgent-based simulation of the impact of social environment on individual flood insurance purchase
behavior. (While there is no data to validate, but the trend is correct based on common sense or qualitative results of
surveys.)
The fourth scenario uses InsurAgent to simulate how prior flood experience influences individual decisions to purchase
flood insurance. Prior research has demonstrated that direct exposure to flooding events can heighten perceived risk and
thus increase insurance purchase probability (Petrolia et al., 2013; Lawrence et al., 2014; Royal and Walls, 2019). To
illustrate this, three individuals with identical socio-demographic features are constructed: all are 28-year-old males
living in Miami, each with a bachelor’s degree and an annual income of $58,000. Their profiles differ only in childhood
flood experience: the first individual experienced catastrophic flooding during Hurricane Katrina; the second individual
encountered a moderate tropical storm causing minor damage; the third individual has no personal flood experience.
The results indicate that the InsurAgent effectively integrates these qualitative flood experiences into its predictions,
yielding probabilities of 37.9%, 33.4%, and 28.0%, respectively, as shown in Fig. 12. The overall trend of the estimation
is consistent with prior research results, demonstrating the agent’s capability to interpret contextual details from personal
experiences and thereby providing an opportunity to enhance the fidelity of individual-level behavior modeling.
14

APREPRINT- NOVEMBER5, 2025
Fig. 12. InsurAgent : Effects of Flood Experience on Individual Flood Insurance Purchase Behavior 
User Profile : You are a 28-year-old male  residing  in Miami , Florida . You hold a Bachelor  of Science in Accounting  
from  the University  of Miami  and earn an annual  salary  of $58,000 as a staff accountant  at a regional  CPA  firm. As a child,  
you lived  in coastal  Biloxi,  Mississippi,  and at age 10 experienced  catastrophic flooding  during  Hurricane Katrina,  which  
inundated your neighborhood for weeks  and caused  substantial  property  damage .
     InsurAgent  Response: This estimate  is based  on a combination  of factors . My education  level  (Bachelor’s  degree)  
suggests  a 28.8% baseline probability  from  the survey  data.  My annual  income  ($58,000) falls into the $50,000-$74,999 
range,  indicating  a 26.5% purchase rate. However , my age (28) has a relatively  low impact,  with a 23.8% purchase  rate for 
my age group.  The traumatic  experience of living  through Hurricane Katrina as a child  significantly  increases  my perceived  
risk and probability  of purchasing  flood  insurance,  adjusting  my estimate upward . Considering  these factors  and the local  
flood risk in Miami , I arrive  at an estimated  purchase  probability  of 37.9% for flood insurance .
User Profile : You are a 28-year-old male  residing  in Miami , Florida . You hold a Bachelor  of Science in Finance  from  
Florida  International  University  and earn an annual  salary  of $60,000 as a financial  analyst  at a regional  investment  firm. 
As a child,  you lived  in Jacksonville,  Florida,  and at age 9 experienced  a moderate flood caused  by a tropical  storm , which  
led to temporary  home  evacuation  and minor  property  damage . 
     InsurAgent  Response: This estimate  is based  on a combination of factors : my income  level  ($60,000) suggests  a 26.5% 
baseline probability  (between  $50,000 and $74,999 range),  which  is then adjusted  upward  due to my education  level  
(Bachelor’s  degree,  around 28.8% probability) . Living  in Miami , Florida,  a coastal  area prone  to flooding,  further  increases  
the probability . My personal  experience with flooding  as a child  may also contribute  to a higher  perceived  risk, slightly  
increasing  the probability . However,  considering  my age (23.8% for individuals  aged  25-34), these  factors  have  a relatively  
moderate impact,  resulting  in a refined  estimate  of 33.4%.
User Profile : You are a 28-year-old male  residing  in Miami , Florida . You hold a Bachelor  of Science in Finance from  
the University  of Miami  and earn an annual  salary  of $58,000 as a financial  analyst  at a regional  bank . You were raised  in 
suburban Omaha,  Nebraska,  where flooding  has historically  been  minimal  and never  impacted  your family’s  property .
     InsurAgent  Response: This estimate  is based  on a combination  of factors : my education  level  (Bachelor’s  degree ) 
suggests  a 28.8% probability  of purchasing  flood  insurance,  and my annual  income  ($58,000) falls between  $50,000-
$74,999, indicating  a 26.5% probability . But this probability  is adjusted  downward  due to my lack of personal  experience  
with flooding . The fact that I live in Miami , Florida,  an area prone  to flooding,  also increases  the probability,  but my age 
(28) has a minimal  impact . Overall,  these factors  balance out to a 28.0% probability  of purchasing  flood insurance .
Statistical  significance: Age; Gender; Education***; Income***. 
Fig. 12: InsurAgent-based simulation of the impact of flood experience on individual flood insurance purchase behavior.
(While there is no data to validate, but the trend is correct based on common sense or qualitative results of surveys.)
The last scenario simulates how an individual’s insurance claim history influences their decision to purchase flood
insurance. Positive claim experiences can foster trust in insurance systems, thereby increasing willingness to engage
with other insurance products (Johnson and Grayson, 2005; Ennew and Sekhon, 2014). To examine this effect, three
individuals with identical socio-demographic profiles are constructed: a 25-year-old males residing in Atlanta, holding
a bachelor’s degree and earning an annual salary of $58,000. The first experienced severe water intrusion during a
flooding event and received full reimbursement through a smooth and expedited claim process. The second filed an auto
insurance claim after a rear-end collision. Although initially denied, the claim was approved upon appeal. The third
experienced a similar accident, but the claim and subsequent appeals were denied. The results show that the InsurAgent
effectively integrates these personal claim experiences into its predictions, as demonstrated in Fig. 13. The highest
purchase probability of 34.1% is assigned to the individual with the positive experience, while the lowest, 28.2%,
corresponds to the individual with the unsuccessful claim. For the case involving a difficult but successful claim, the
agent produces a two-sided response: the initial denial introduces skepticism, but the eventual reimbursement reinforces
the value of insurance. This results in a moderate probability of 31.2%. These results highlight InsurAgent’s capacity to
interpret and reason over complex personal experience to realistically simulate human behavior.
5.4 Modeling dynamic life trajectories through memory module
Another advantage of LLM-empowered agents lies in their ability to model dynamic life trajectories by integrating new
information within a coherent sequential framework. This is achieved by the InsurAgent’s memory module, which
archives events with their time stamps. These records are then embedded into the prompt during each subsequent
interaction, allowing the agent to simulate how individuals update their beliefs and intentions in response to life events.
Unlike regression models that treat input variables as static, this capability enables temporally consistent reasoning that
15

APREPRINT- NOVEMBER5, 2025
aligns with the evolving processes of human cognition. Fig. 14 illustrates this capability through a simulation of one’s
life story. The simulation begins by defining a user profile: a 32-year-old male with a bachelor’s degree and annual
income of $65,000, followed by a sequence of life events unfolding over time.
Fig. 13. InsurAgent : Effects of Claim History on Individual Flood Insurance Purchase Behavior 
User Profile : You are a 25-year-old male  residing  in Atlanta , Georgia.  You hold a Bachelor  of Science in Computer  
Science from  the Georgia Institute  of Technology  and earn an annual  salary  of $58,000 as a data analyst  at a technology  
startup . Last year,  your condominium  sustained  significant  water  intrusion  during  a flooding  event . You filed a 
homeowners’  insurance  claim  and received  full reimbursement , including  structural  repairs  and replacement  of damaged  
personal  effects,  within  three  weeks . The process  required  minimal  documentation and no negotiations .
     InsurAgent  Response: This estimate  is based  on synthesizing  the provided  survey  data with my unique  background.  As 
a 25-year-old with a college degree  and an annual  income  of $58,000, my baseline  probability  starts  around 28.8% (college  
degree ) and 26.5% (income  between  $50,000 and $74,999), respectively . The significant  factor  increasing  my probability  is 
the recent  experience of water  intrusion  and the smooth claims  process, which  likely  heightened my awareness  and 
appreciation  of insurance  benefits . Despite Atlanta  not being  directly  in the Gulf Coast,  the recent  flood event  still 
influences  my decision,  pushing  my final probability  of 34.1%, above  the survey  averages  for education  and income  alone .
User Profile : You are a 25-year-old male  residing  in Atlanta , Georgia.  You hold a Bachelor  of Science in Information  
Systems  from  the University  of Washington and earn an annual  salary  of $58,000 as a business  intelligence  analyst  at a 
healthcare analytics  firm. Last year,  your car was rear- ended  during  a heavy  rain, causing  substantial  damage  to the rear 
bumper . Although  the other  driver  was at fault,  your auto insurer  initially  denied  parts  of your repair  claim,  citing  “pre-
existing  damage”.  You spent  nearly  two months  contesting  the decision , including  submitting  photographic  evidence,  
repair  shop diagnostics,  and a formal  complaint,  before  the insurer  revised  its stance and issued  a full payout .     
     InsurAgent  Response: As a 25-year-old with a Bachelor’s  degree,  my education  level  suggests a higher  propensity  to 
purchase  insurance  (28.8% from  the survey) . My annual  income  of $58,000 falls into the $50,000-$74,999 range,  
indicating  a 26.5% probability . Being  male , the gender  factor  slightly  increases  this probability  (26.2%). However,  my age 
has minimal  impact  according  to the prioritization . The recent  negative experience with an auto insurer  might  increase my 
skepticism  but also highlights  the importance  of insurance,  potentially  offsetting  each other . Given  Atlanta ’s flood risk is 
lower  than the Gulf Coast,  these factors  result s in a final estimated  probability  of 31.2% for purchasing  the flood insurance .
User Profile : You are a 25-year-old male  residing  in Atlanta , Georgia.  You hold a Bachelor  of Science in Information 
Systems  from  the University  of Washington and earn an annual  salary  of $58,000 as a business  intelligence  analyst  at a 
healthcare analytics  firm. Last year,  your car was rear- ended  during  a heavy  rain, causing  substantial  damage  to the rear 
bumper . Although  the other  driver  was clearly  at fault,  your auto insurer  partially  denied  your claim , arguing  that the 
damage stemmed  from  prior  wear  and tear. Despite submitting  repair  shop  diagnostics,  photographic  evidence,  and a 
formal  appeal,  your efforts  were unsuccessful,  and the insurer  refused  to revise  the payout . 
     InsurAgent  Response: As a 25-year-old, the age factor  has a minimal  impact . The income  ($58,000) suggests  a 
purchase  rate around  26.5% based  on the income  survey  data. Holding  a Bachelor’s  degree increase this rate further,  
potentially  closer  to 28.8% as seen in the education  survey . Despite these factors  suggesting  a higher  probability  of 
purchasing  flood  insurance,  my negative  experience with an auto insurer  might  decrease my trust in insurance  companies,  
potentially  lowering  my purchase  probability . Considering  these aspects  and the fact that Atlanta  is not directly  on the Gulf 
Coast  but still experiences  significant  rainfall  and occasional  flooding,  I adjust  my estimated  purchase probability  to 28.2%.
Statistical  significance: Age; Gender; Education***; Income***. 
Fig. 13: InsurAgent-based simulation of the impact of claim history on individual flood insurance purchase behavior.
(While there is no data to validate, but the trend is correct based on common sense or qualitative results of surveys.)
The agent’s reasoning evolves over time in response to six sequential events, producing a roller coaster trajectory in
flood insurance purchase probability, as demonstrated in Fig. 14. Initially, the baseline probability is determined as
28.0%, derived from sociodemographic-based statistics. After attending a workshop on coastal resilience, the agent
identifies increased risk awareness, raising the probability to 32.5%. This is followed by a slight decrease to 30.5% after
reviewing historical flood records indicating no major flooding in the region over the past decade. Subsequently, the
agent’s risk perception is significantly altered by vicarious experience, as it witnesses a neighbor’s distress and financial
hardship from flood damage, resulting in a sharp increase to 38.2%. The announcement of a new federal post-flood
relief program moderately decreases the purchase probability to 34.5%, as the availability of external aid partially
reduces the perceived need for insurance. However, this downward adjustment is outweighed by expert predictions of
accelerating sea-level rise and inadequate flood defenses, which elevate the probability to 41.8%. Finally, the user’s trust
is eroded by negative experiences such as delayed disaster claims and denied applications, yielding a final probability
of 36.6%. These results underscore the agent’s capacity to interpret and integrate diverse sources of information over
time and predict reasonable probabilities accordingly.
16

APREPRINT- NOVEMBER5, 2025
Fig. 14. 
You are a 32-year -old male  living  in Charleston,  South Carolina . You hold a Bachelor  of Architecture 
from  Duke  University  and earn an annual  salary  of $65,000 as an urban planner . User 
Profile
Insur
AgentAs a male with a college degree  and an annual  income  between  $50,000 and $74,999, my baseline  
probability  is informed by the survey  statistics  for these groups  (28.8% for those  with a college  degree and 
26.5% for the income  range ). Given  my profession as an urban planner  and my education  in architecture,  I 
likely  have  a heightened  awareness  of flood risks,  which  would  increase my probability  of purchasing  flood 
insurance . However,  being  slightly  younger  than the peak  age, my final probability  is adjusted  to reflect  these  
considerations,  resulting  in a purchase  likelihood  of 28.0%.
After  attending  a professional  workshop on coastal  resilience,  featuring  detailed  case studies  of 
collapsed  seawalls,  your awareness  of flood risk spikes .
This revised  estimate reflects  an upward adjustment  from  the initial 28.0% due to increased  awareness  of 
flood risk following  the professional  workshop  on coastal  resilience . This new information  heightened  my 
perception  of flooding  risk, thus slightly  increasing  my probability  of purchasing  flood  insurance  to 32.5% to 
mitigate  potential  losses.  However,  the increase remains  moderate , considering  the baseline probabilities  
derived  from  demographic  and income  factors .Event 1
Insur
Agent
One year later,  you review  NOAA’s historical flood-frequency  records,  which  show  no major  
inundations  in Charleston  over the past decade.
This revised  estimate reflects  a slight  downward adjustment  from  32.5% after reviewing  NOAA’s historical 
flood-frequency  records . While  my awareness  of flood risk was heightened  by the professional  workshop,  the 
flood frequency  data suggests a lower  risk, moderately  decreasing  my probability  of purchasing  flood 
insurance  to 30.5%. However,  the decrease is not drastic,  as I still consider  the long-term  risks  and the 
potential for future  flooding  given  Charleston’s  coastal  location .Event 2
Insur
Agent
Two years  on, a colleague’s  basement  suffers  severe  flooding  after an unprecedented  tidal surge . You 
witness  the resulting  distress and financial  hardship firsthand .
This revised  estimate reflects  a significant upward adjustment  from  30.5% after witnessing  my colleague’s  
experience.  The direct  observation  of flood  damage and its impact  on someone  I know  has substantially  
increased  my perception  of flood  risk. This experience overrides  some  of the statistical reassurance from  
NOAA’s records,  highlighting  the tangible  consequences  of flooding  and thus increasing  my willingness  to 
invest  in flood insurance . Therefore,  my probability  of purchasing  flood insurance  rise to 38.2%.Event 3
Insur
Agent
Thereafter,  you learn  that the city council  has approved a new federal  grant  program  for post-flood  
relief . You believe the external  aid will sufficiently  cover  potential losses.
This revised  estimate reflects  a moderate decrease from  38.2% after learning  about  the city council’s  
approval  of a new federal  grant  program  for post-flood relief . While  witnessing  my colleague’s  flooding  
experience increased  my awareness  and concern,  the knowledge  that external  aid will be available  to cover  
potential losses somewhat  alleviates  my need  to personally  mitigate  all flood risks  through  insurance , thus 
slightly  decreasing  my probability  of purchasing  flood insurance  to 34.5%.Event 4
Insur
Agent
During  the ensuing  hurricane  season,  you read a peer-reviewed  climate report  forecasting  accelerated  
sea-level  rise in your region  and warning  that existing  flood defenses  will soon be inadequate . 
This revised  estimate reflects  a significant upward adjustment  due to the accelerated  sea-level  rise and the 
anticipated  inadequacy  of existing  flood defenses . This information  substantially  increases  my perception  of 
long-term  flood risk, outweighing  the reassurance from  the federal  grant  program  for post-flood  relief , and 
thus significantly  increases  my probability  to protect  against  potential future  losses to 41.8%.Event 5
Insur
Agent
A local  news  outlet runs a vivid  investigative  series  on delayed  disaster  claims , highlighting  long wait 
times  and denied  applications . This erodes  your confidence  in insurers’  responsiveness .
This revised  estimate reflects  a decrease from  41.8% due to the news  on delayed  disaster  claims  and denied  
applications,  which  has eroded  my confidence  in insurers’  ability  to provide  support . Despite  the heightened  
awareness  of flood risk from  the climate report,  the potential inefficiencies  and uncertainties  in the insurance  
claims  process  moderate my willingness  to purchase  flood insurance , decreasing  the probability  to 36.6%. Event 6
Insur
Agent
Fig. 14: Modeling dynamic insurance purchase decisions in response to evolving life events using InsurAgent.
17

APREPRINT- NOVEMBER5, 2025
6 Limitations and future work
While InsurAgent demonstrates strong alignment with survey-based benchmarks and generates plausible extrapolations,
several limitations deserve further investigation. First, the current study relies on a single empirical dataset from a
specific geographic region. This may constrain the agent’s ability to generalize behavioral patterns across broader
populations, as regional differences in climate risk, socioeconomic conditions, and public policy may influence
individuals’ insurance decisions. To address this limitation, a necessary next step is to incorporate survey datasets from
diverse regions. This multi-regional dataset would serve a dual purpose: expanding the agent’s learning scope beyond
localized patterns and providing a testbed for cross-validating its generalizability. Second, although the agent is capable
of extrapolating beyond survey data using the common sense of LLMs, these predictions remain unvalidated due to
the lack of corresponding benchmarks. Ensuring the credibility of these extrapolations, potentially through expert
review or the collection of new behavioral data, should be a key priority in future work. Third, this study is limited to
simulating human decision-making at the individual level. However, insurance decisions are often shaped by broader
social influences, such as peer effects and information diffusion. Extending the framework to a multi-agent simulation
environment would allow for the modeling of population-level decision patterns and emergent behaviors. This would
broaden the applicability of InsurAgent to policy evaluation, disaster preparedness planning, and behavioral economics
research.
7 Conclusions
This study presents a novel paradigm for simulating individual decision-making in flood insurance purchase through the
development of InsurAgent, a large language model (LLM)-empowered agent. Unlike traditional statistical approaches,
InsurAgent leverages natural language understanding and reasoning capabilities of LLMs to interpret user profiles
and incorporate domain knowledge and contextual information into behavior predictions. At the core of its design
is the integration of empirical survey data through a retrieval-augmented generation (RAG) framework. The survey
data, derived from Shao et al. (2017), serves to align the agent’s predictions with population-level statistics. The
architecture of InsurAgent comprises five components including perception, retrieval, reasoning, action, and memory.
The perception module parses user profiles to identify key factors. These factors are passed to the retrieval module,
which employs RAG to obtain relevant survey data from an external database. The reasoning module then synthesizes
the retrieved data and parsed user information to emulate human cognitive processes. Following the reasoning process,
the action module predicts a purchase probability expressed as a percentage and reports a concise rationale. Additionally,
the memory module chronologically records the reasoning processes and decision outputs, enabling the modeling of
evolving behaviors in response to sequential life events. The key findings are summarized as follows:
•Alignment with empirical data: InsurAgent demonstrates strong consistence with empirical marginal proba-
bilities and bivariate probabilities. For marginal probability estimations, the agent replicates empirical flood
insurance purchase probabilities across all individual factors. For bivariate probability estimations, InsurAgent
exhibits strong alignment with survey-based benchmarks, yielding a correlation of determination (R2) of 0.778
and a mean absolute error (MAE) of 0.024.
•Ability to extrapolate: Besides reproducing statistical trends, InsurAgent leverages its common sense to
extrapolate in situations where regression models cannot predict. In five “beyond regression” scenarios, the
agent effectively incorporates contextual information related to residential city, occupations, social environment,
flood experience, and claim history to make plausible predictions.
•Modeling dynamic decision trajectories: Through its memory module, InsurAgent chronologically archives
historical decisions and reasonings. This capability is demonstrated by a simulation, in which the agent
responds to six sequential life events and make predictions according to a roller coaster trajectory.
•A key limitation of the current study is the reliance on a single empirical dataset from a specific geographic
region, potentially constraining the generalizability of behavioral predictions. Future work should incorporate
survey data from diverse regions to support cross-validation and enhance model robustness. Additionally,
future work should prioritize the quantitative validation of extrapolations beyond survey data and extend the
framework to support modeling of population-scale behavioral patterns.
Data Availability Statement
Some or all data, models, or code that support the findings of this study are available from the corresponding author
upon reasonable request.
18

APREPRINT- NOVEMBER5, 2025
References
Atreya, A., Ferreira, S., and Michel-Kerjan, E. (2015). “What drives households to buy flood insurance? new evidence
from georgia.”Ecological Economics, 117, 153–161.
Bird, D. K. (2009). “The use of questionnaires for acquiring information on public perception of natural hazards
and risk mitigation–a review of current knowledge and practice.”Natural hazards and earth system sciences, 9(4),
1307–1325.
Brody, S. D., Highfield, W. E., Wilson, M., Lindell, M. K., and Blessing, R. (2017). “Understanding the motivations of
coastal residents to voluntarily purchase federal flood insurance.”Journal of Risk Research, 20(6), 760–775.
Browne, M. J. and Hoyt, R. E. (2000). “The demand for flood insurance: empirical evidence.”Journal of risk and
uncertainty, 20(3), 291–306.
Chen, J., Yuan, S., Ye, R., Majumder, B. P., and Richardson, K. (2023). “Put your money where your mouth is:
Evaluating strategic planning and execution of llm agents in an auction arena.”arXiv preprint arXiv:2310.05746.
Chen, R., Wang, C., Sun, Y ., Zhao, X., and Xu, S. (2025). “From perceptions to decisions: Wildfire evacuation decision
prediction with behavioral theory-informed llms.”arXiv preprint arXiv:2502.17701.
Cheng, M. and Frangopol, D. M. (2022). “Life-cycle optimization of structural systems based on cumulative prospect
theory: Effects of the reference point and risk attitudes.”Reliability engineering & system safety, 218, 108100.
Cheng, M. and Frangopol, D. M. (2023). “Efficient scenario analysis for optimal adaptation of bridge networks under
deep uncertainties through knowledge transfer.”Structural Safety, 100, 102278.
Cheng, M., Yang, D. Y ., and Frangopol, D. M. (2020). “Investigation of effects of time preference and risk perception
on life-cycle management of civil infrastructure.”ASCE-ASME Journal of Risk and Uncertainty in Engineering
Systems, Part A: Civil Engineering, 6(1), 04020001.
Comanici, G., Bieber, E., Schaekermann, M., Pasupat, I., Sachdeva, N., Dhillon, I., Blistein, M., Ram, O., Zhang, D.,
Rosen, E., et al. (2025). “Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and
next generation agentic capabilities.”arXiv preprint arXiv:2507.06261.
Comes, T., Meesters, K., Nespeca, V ., et al. (2017). “Hurricane harvey report: A fact-finding effort in the direct
aftermath of hurricane harvey in the greater houston region.
Congressional Budget Office (2024). “Flood insurance in communities at risk of flooding. Accessed August 6, 2025.
Creswell, A., Shanahan, M., and Higgins, I. (2022). “Selection-inference: Exploiting large language models for
interpretable logical reasoning.”arXiv preprint arXiv:2205.09712.
Darlington, J. C. and Yiannakoulias, N. (2022). “Experimental evidence for coverage preferences in flood insurance.”
International Journal of Disaster Risk Science, 13(2), 178–189.
Ennew, C. and Sekhon, H. S. (2014). “Trust and trustworthiness in retail financial services: An analytical framework
and empirical evidence.”The Routledge Companion to Financial Services Marketing, Routledge, 148–165.
Eryılmaz Türkkan, G. and Hırca, T. (2021). “The investigation of flood risk perception as a quantitative analysis from
socio-demographic perspective.”Natural Hazards, 106(1), 715–733.
Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A.,
Vaughan, A., et al. (2024). “The llama 3 herd of models.”arXiv preprint arXiv:2407.21783.
Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al. (2025). “Deepseek-r1:
Incentivizing reasoning capability in llms via reinforcement learning.”arXiv preprint arXiv:2501.12948.
HEGAR, G. (2018). “A storm to remember.”Texas Comp of Public Accts.
Hill, T., Kusev, P., and Van Schaik, P. (2019). “Choice under risk: How occupation influences preferences.”Frontiers in
Psychology, 10, 2003.
Hung, H.-C. (2009). “The attitude towards flood insurance purchase when respondents’ preferences are uncertain: a
fuzzy approach.”Journal of Risk Research, 12(2), 239–258.
Johnson, D. and Grayson, K. (2005). “Cognitive and affective trust in service relationships.”Journal of Business
research, 58(4), 500–507.
Karanikolas, N., Manga, E., Samaridi, N., Tousidou, E., and Vassilakopoulos, M. (2023). “Large language models
versus natural language understanding and generation.”Proceedings of the 27th Pan-Hellenic Conference on Progress
in Computing and Informatics, 278–290.
Kouabenan, D. (2002). “Occupation, driving experience, and risk and accident perception.”Journal of Risk Research,
5(1), 49–68.
19

APREPRINT- NOVEMBER5, 2025
Kousky, C. (2017). “Disasters as learning experiences or disasters as policy opportunities? examining flood insurance
purchases after hurricanes.”Risk analysis, 37(3), 517–530.
Kundis Craig, R. (2017). “Harvey, irma, and the nfip: Did the 2017 hurricane season matter to flood insurance
reauthorization.”UALR L. Rev., 40, 481.
Kunreuther, H. (2021). “Improving the national flood insurance program.”Behavioural Public Policy, 5(3), 318–332.
LaCour-Little, M., Palim, M., and Pan, Y . (2025). “Why don’t consumers purchase flood insurance or mitigate flood
risk?.”Journal of Housing Research, 34(1), 60–89.
Lawrence, J., Quade, D., and Becker, J. (2014). “Integrating the effects of flood experience on risk perception with
responses to changing climate risk.”Natural Hazards, 74(3), 1773–1794.
Li, N., Gao, C., Li, M., Li, Y ., and Liao, Q. (2023). “Econagent: large language model-empowered agents for simulating
macroeconomic activities.”arXiv preprint arXiv:2310.10436.
Liang, H., Kalaleh, M. T., and Mei, Q. (2025a). “Integrating large language models for automated structural analysis.”
arXiv preprint arXiv:2504.09754.
Liang, H., Zhou, Y ., Kalaleh, M. T., and Mei, Q. (2025b). “Automating structural engineering workflows with large
language model agents.”arXiv preprint arXiv:2510.11004.
Liu, J., Geng, Z., Cao, R., Cheng, L., Bocchini, P., and Cheng, M. (2025). “A large language model-empowered agent
for reliable and robust structural analysis.”arXiv preprint arXiv:2507.02938.
Liu, L., Yang, D. Y ., and Frangopol, D. M. (2020). “Network-level risk-based framework for optimal bridge adaptation
management considering scour and climate change.”Journal of Infrastructure Systems, 26(1), 04019037.
Min, B., Ross, H., Sulem, E., Veyseh, A. P. B., Nguyen, T. H., Sainz, O., Agirre, E., Heintz, I., and Roth, D. (2023).
“Recent advances in natural language processing via large pre-trained language models: A survey.”ACM Computing
Surveys, 56(2), 1–40.
Netusil, N. R., Kousky, C., Neupane, S., Daniel, W., and Kunreuther, H. (2021). “The willingness to pay for flood
insurance.”Land Economics, 97(1), 17–38.
of Richmond, F. R. B. (2024). “Macro minute: Flood risk and insurance uptake (October). Accessed: August 7, 2025.
OpenAI (2024). “Gpt-4o: Openai’s multimodal flagship model. Accessed: 2025-05-21.
Payne, J. W., Bettman, J. R., and Johnson, E. J. (1992). “Behavioral decision research: A constructive processing
perspective.”Annual review of psychology, 43(1), 87–131.
Petrolia, D. R., Landry, C. E., and Coble, K. H. (2013). “Risk preferences, risk perceptions, and flood insurance.”Land
Economics, 89(2), 227–245.
Royal, A. and Walls, M. (2019). “Flood risk perceptions and insurance choice: Do decisions in the floodplain reflect
overoptimism?.”Risk Analysis, 39(5), 1088–1104.
Sebastian, A., Lendering, K., Meesters, K., van de Walle, B., Ebrahimi Fard, A., Cunningham, S., Khakzad Rostami,
N., Nespeca, V ., Kothuis, B., Brand, A., et al. (2017). “Hurricane harvey report.
Shao, W., Feng, K., and Lin, N. (2019). “Predicting support for flood mitigation based on flood insurance purchase
behavior.”Environmental Research Letters, 14(5), 054014.
Shao, W., Xian, S., Lin, N., Kunreuther, H., Jackson, N., and Goidel, K. (2017). “Understanding the effects of past
flood events and perceived and estimated flood risks on individuals’ voluntary flood insurance purchase behavior.”
Water research, 108, 391–400.
Simon, H. A. (1955). “A behavioral model of rational choice.”The quarterly journal of economics, 99–118.
Tezuka, S., Takiguchi, H., Kazama, S., Sato, A., Kawagoe, S., and Sarukkalige, R. (2014). “Estimation of the effects of
climate change on flood-triggered economic losses in japan.”International Journal of Disaster Risk Reduction, 9,
58–67.
Wang, L., Zhang, J., Yang, H., Chen, Z.-Y ., Tang, J., Zhang, Z., Chen, X., Lin, Y ., Sun, H., Song, R., et al. (2025).
“User behavior simulation with large language model-based agents.”ACM Transactions on Information Systems,
43(2), 1–37.
Xu, F., Lin, Q., Han, J., Zhao, T., Liu, J., and Cambria, E. (2025). “Are large language models really good logical
reasoners? a comprehensive evaluation and beyond.”IEEE Transactions on Knowledge and Data Engineering.
Xu, J. and Fan, C. (2025). “Assessing household underestimation of hurricane food shortages via large language model
agent simulation.”International Journal of Disaster Risk Reduction, 105734.
Yang, Y ., Zhang, Y ., Wu, M., Zhang, K., Zhang, Y ., Yu, H., Hu, Y ., and Wang, B. (2025). “Twinmarket: A scalable
behavioral and social simulation for financial markets.”arXiv preprint arXiv:2502.01506.
20