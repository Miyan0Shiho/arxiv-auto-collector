# Agentic Multimodal AI for Hyperpersonalized B2B and B2C Advertising in Competitive Markets: An AI-Driven Competitive Advertising Framework

**Authors**: Sakhinana Sagar Srinivas, Akash Das, Shivam Gupta, Venkataramana Runkana

**Published**: 2025-04-01 01:37:02

**PDF URL**: [http://arxiv.org/pdf/2504.00338v1](http://arxiv.org/pdf/2504.00338v1)

## Abstract
The growing use of foundation models (FMs) in real-world applications demands
adaptive, reliable, and efficient strategies for dynamic markets. In the
chemical industry, AI-discovered materials drive innovation, but commercial
success hinges on market adoption, requiring FM-driven advertising frameworks
that operate in-the-wild. We present a multilingual, multimodal AI framework
for autonomous, hyper-personalized advertising in B2B and B2C markets. By
integrating retrieval-augmented generation (RAG), multimodal reasoning, and
adaptive persona-based targeting, our system generates culturally relevant,
market-aware ads tailored to shifting consumer behaviors and competition.
Validation combines real-world product experiments with a Simulated Humanistic
Colony of Agents to model consumer personas, optimize strategies at scale, and
ensure privacy compliance. Synthetic experiments mirror real-world scenarios,
enabling cost-effective testing of ad strategies without risky A/B tests.
Combining structured retrieval-augmented reasoning with in-context learning
(ICL), the framework boosts engagement, prevents market cannibalization, and
maximizes ROAS. This work bridges AI-driven innovation and market adoption,
advancing multimodal FM deployment for high-stakes decision-making in
commercial marketing.

## Full Text


<!-- PDF content starts -->

Published as a conference paper at ICLR 2025
AGENTIC MULTIMODAL AI FOR HYPER -
PERSONALIZED B2B AND B2C A DVERTISING IN
COMPETITIVE MARKETS : A NAI-D RIVEN COMPETI -
TIVE ADVERTISING FRAMEWORK
Sakhinana Sagar Srinivas∗ †
Tata Research
Bangalore, India
sagar.sakhinana@tcs.comAkash Das∗
Tata Research
Bangalore, India
akash@tcs.comShivam Gupta
Tata Research
New Delhi, India
shivam.gupta@tcs.com
Venkataramana Runkana
Tata Research
Pune, India
venkat.runkana@tcs.com
ABSTRACT
The increasing deployment of foundation models (FMs) in real-world applica-
tions necessitates strategies to enhance their adaptivity, reliability, and efficiency
in dynamic market environments. In the chemical industry, AI-discovered materi-
als are driving innovation in new chemical products, but their commercial success
depends on effective market adoption. This requires FM-driven advertising frame-
works capable of operating in-the-wild, adapting to diverse consumer segments
and competitive market conditions. We introduce an AI-driven, multilingual,
multimodal framework that leverages foundation models for autonomous, hyper-
personalized, and competitive advertising in both Business-to-Business (B2B) and
Business-to-Consumer (B2C) markets. By integrating retrieval-augmented gen-
eration (RAG), multimodal reasoning, and adaptive persona-based targeting, our
framework generates culturally relevant and market-aware advertisements tailored
to dynamic consumer behaviors and competitive landscapes. Our approach is val-
idated through a combination of real-world experiments using actual product data
and a Simulated Humanistic Colony of Agents to model consumer personas and
optimize ad strategies at scale while maintaining privacy compliance. This ensures
market-grounded and regulatory-compliant advertising. Synthetic experiments are
designed to mirror real-world scenarios, enabling the testing and optimization of
advertising strategies by simulating market conditions, consumer behaviors, and
product scenarios. This approach helps companies avoid costly real-world A/B
tests while ensuring privacy compliance and scalability, allowing them to refine
strategies through simulations before actual deployment. By combining struc-
tured retrieval-augmented reasoning with in-context learning (ICL) for adaptive
ad generation, the framework enhances engagement, prevents market cannibaliza-
tion, and optimizes Return on Ad Spend (ROAS). This work presents a scalable
FM-driven solution that bridges AI-driven novel product innovation and market
adoption, advancing the deployment of multimodal, in-the-wild AI systems for
high-stakes decision-making environments such as commercial marketing.
1 I NTRODUCTION
In recent years, generative AI has revolutionized materials discovery. For instance, MatterGen has
enabled the design of novel materials, while MatterSim validates their performance under real-world
conditions Zeni et al. (2023); Yang et al. (2024). Complementing these efforts, autonomous labs
like A-Lab integrate AI with robotics to propose and execute synthesis recipes, accelerating the
∗Equal contribution.
†Corresponding author.
1arXiv:2504.00338v1  [cs.LG]  1 Apr 2025

Published as a conference paper at ICLR 2025
discovery and development of innovative materials for applications such as energy storage and sus-
tainability Burger et al. (2024). These materials form the foundation of a wide range of chemi-
cal products, with applications spanning fast-moving consumer goods (FMCG), the semiconductor
industry, the electronics sector, and beyond. Recent advancements in AI-driven materials discov-
ery, autonomous synthesis, and process automation Zeni et al. (2023); Yang et al. (2024); Burger
et al. (2024), coupled with progress in scaling the production of chemical products from simu-
lations and laboratory experiments to large-scale industrial manufacturing Srinivas et al. (2024);
Gowaikar et al. (2024), have paved the way for commercialization. However, a significant bot-
tleneck remains in translating these innovations into market adoption. Success depends not only
on technical performance but also on stakeholders’ ability to effectively communicate their value
to diverse audiences—ranging from consumers and investors to regulatory bodies. The rapid evo-
lution of digital platforms has transformed advertising, compelling businesses to adopt AI-driven
programmatic advertising to stay competitive. Chemical product manufacturers increasingly lever-
age Demand-Side Platforms (DSPs) to participate in Real-Time Bidding (RTB) auctions across both
business-to-business (B2B) and business-to-consumer (B2C) channels. The primary objective is to
maximize Return on Ad Spend (ROAS) by optimizing ad visibility across various platforms. This
optimization spans search engines (Google Ads, Bing Ads), where advertisers engage in Open Auc-
tions or negotiate Programmatic Guaranteed deals; social media platforms such as LinkedIn for
B2B engagement and Instagram/TikTok for consumer outreach; and e-commerce marketplaces like
Amazon, Alibaba, and Knowde. At the core of these campaigns is RTB, a process where publish-
ers (sellers) offer individual ad impressions for sale through Supply-Side Platforms (SSPs), while
advertisers (buyers) compete to purchase these impressions via DSPs in real-time auctions. These
auctions often follow a second-price bidding model or header bidding, ensuring that advertisers op-
timize ROAS while publishers maximize their ad revenue. SSPs facilitate auctions by passing user
and page-level data—including geolocation, device type, search intent, and potentially Mobile Ad
IDs (MAIDs)—to DSPs, ensuring compliance with privacy regulations such as GDPR and CCPA.
To refine targeting, advertisers integrate first-party data, such as website visits, purchase history, and
CRM records; third-party audience intelligence sourced via Data Management Platforms (DMPs),
which provide insights into demographics, interests, and behaviors; and real-time contextual data,
such as webpage content, device type, and live user intent, ensuring ads appear at the most relevant
moments. These elements help DSPs optimize bids, enhancing targeting precision, while Ad Ex-
changes determine the winning bid. A critical factor in RTB success is Click-Through Rate (CTR)
prediction, which selects the most relevant ad creative (often HTML5-based rich media) for display.
This personalized approach enhances user engagement and mitigates ad fatigue, ensuring ads re-
main relevant, engaging, and likely to drive conversions. RTB platforms reward high-CTR ads with
lower cost-per-click (CPC), reducing acquisition costs and improving conversion efficiency for ad-
vertisers. Given this dynamic, AI-driven Dynamic Creative Optimization (DCO) plays a crucial role
in enabling hyper-personalized advertising, a key driver of ROAS improvement. DCO dynamically
adjusts visuals, messaging, and calls-to-action based on user behavior, engagement metrics, and real-
time contextual data, ensuring that ads resonate deeply with each audience segment. By leveraging
DCO techniques, advertisers enhance hyper-personalization, drive higher engagement and conver-
sions, and ultimately maximize ROAS. Despite advances in AI, existing advertising systems often
fail to leverage contextual multimodal user behavior insights, limiting their effectiveness in competi-
tive product marketing, where multimodal, multilingual, and persona-specific ad targeting is critical.
Advertising multiple products within the same category—such as consumer goods—from different
manufacturers presents additional challenges, including avoiding cannibalization (where ads for one
product negatively impact another), tailoring messages to diverse user preferences, and emphasizing
each product’s unique selling points (USPs), such as superior features, pricing, or brand percep-
tion. Moreover, creating culturally relevant ads for global audiences while addressing competition
within similar product categories from different manufacturers remains a significant hurdle. To ad-
dress these gaps, we propose an automated, data-driven, multilingual, and multimodal framework
capable of delivering hyper-personalized, competitive advertisements tailored to diverse consumer
personas and dynamic market conditions. This framework integrates multimodal data analysis by
leveraging advanced large foundational models and persona-specific targeting to ensure ads resonate
across cultural contexts and stakeholder groups. To validate this approach, we conduct experiments
in both real-world and synthetic settings. Real-world experiments involve using data from actual
companies, including their product information, advertising strategies, and consumer insights. To
model target consumer behavior based on this real-world data, we leverage a Simulated Human-
istic Colony of Agents instead of conducting expensive or complex human trials, allowing us to
2

Published as a conference paper at ICLR 2025
efficiently evaluate and refine advertising strategies before market launch. In contrast, synthetic ex-
periments involve creating controlled environments to simulate various market conditions and using
a Simulated Humanistic Colony of Agents to model personas (consumer segments) for personalized
ad optimization of competing products from different manufacturers. This approach enables the
systematic evaluation of ad strategies at scale while ensuring regulatory compliance (e.g., GDPR)
and avoiding the costs and complexities associated with real-world data collection. By combining
the robustness of real-world validation with the flexibility and scalability of synthetic testing, these
experiments serve as both a proof of concept and a demonstration of the framework’s ability to
generate competitive, impactful advertisements. In real-world settings, the comprehensive frame-
work integrates three interconnected systems: (1) The Multimodal Agentic Advertisement Market
Survey (MAAMS) system (see Figure 2), led by a Meta-Agent, combines insights from publicly
available sources—such as social media, financial databases, and market research tools—to eval-
uate brand sentiment, visual identity, emotional engagement, financial performance, and market
trends. This system generates a comprehensive overview of a product’s market position, provid-
ing a detailed, multimodal analysis of how different companies present and position their chemical
products. (2) The Personalized Market-Aware Targeted Advertisement Generation (PAG) system
(see Figures 3-4) builds on MAAMS to create tailored, multilingual ads for diverse consumer per-
sonas. Using a Simulated Humanistic Colony of Agents, it aligns ads with specific user preferences.
The Adv Curator Agent designs personalized ads, while the Social Media Agent optimizes them
for platforms like Twitter and Instagram. (3) The Competitive Hyper-Personalized Advertisement
System (CHPAS) (see Figure 5) takes the personalized ads generated by PAG and enhances their
competitiveness by differentiating them for competing products from different manufacturers. It
strategically emphasizes unique selling points—such as affordability, functionality, and quality—to
ensure relevance and engagement for target audiences. Together, these systems leverage advanced
agentic architectures powered by LLMs to deliver data-driven, adaptive, and impactful advertising
strategies. For ad evaluations, we utilize a multi-faceted approach that assesses clickability rates
and ad quality through automated reward models, LLM evaluators, and human evaluators. For ad
copy optimization, we evaluate various hyper-personalized versions using simulated human per-
sonas. This iterative, persona-driven testing and evaluation process helps maximize engagement
and conversion rates. Additionally, we utilize Open-Domain Question Answering (ODQA) with
Retrieval-Augmented Generation (RAG) to answer specific questions by searching and analyzing
the market intelligence gathered by the MAAMS system on the product’s market position. Fur-
thermore, we conduct synthetic experiments, offering a powerful tool for advertisers to develop,
test, and optimize AI-driven advertisement generation frameworks in a cost-effective and scalable
manner. These experiments provide a controlled environment for precise testing, ensure inherent
privacy compliance, enhance framework robustness by exposing it to a wide range of scenarios (in-
cluding rare edge cases), and accelerate innovation—all while minimizing the risks associated with
real-world testing. The synthetic, data-driven, hypothetical framework for optimizing multi-product,
persona-specific advertising integrates seven key components: market research data, persona profil-
ing, product analysis, competitive analysis, ad generation using foundational models, ad platform
optimization, and ad evaluation. We discuss this framework in detail in the technical appendix. The
experimental results demonstrate the potential of AI-driven, multimodal frameworks to transform
advertising, providing businesses with a powerful tool to navigate modern marketing challenges.
This work establishes a foundation for future research in AI-driven advertising, offering a scalable
solution for industries seeking to bridge product innovation and market communication. The paper
is organized as follows: Section 2 details our proposed method, Section 3 describes the experimental
setup, Section 5 presents results and analysis, and Section 6 concludes with key findings and future
directions.
2 P ROPOSED METHOD
This section presents a comprehensive framework for multimodal analysis and personalized adver-
tisement generation in the chemical industry (see Figure 1). The framework consists of three inter-
connected systems: (a) The Multimodal Agentic Advertisement Market Survey (MAAMS) system,
which integrates insights across various modalities to assess product sentiment, branding, engage-
ment, profitability, and competitive positioning. (b) The Personalized Market-Aware Targeted Ad-
vertisement Generation (PAG) system, designed to create tailored, multilingual advertisements for
diverse consumer personas. (c) The Competitive Hyper-Personalized Advertisement System (CH-
PAS), which generates differentiated advertisements for competing products, ensuring that unique
selling points are highlighted for specific user preferences. These systems employ advanced agen-
3

Published as a conference paper at ICLR 2025
MAAMS
AI-powered
market intelligencePAG
Personalized ads with
consumer personas and
multilingual optimizationCHPAS
Competitive ad
optimization for
multiple products
Market Intel
Sentiment & visual analysis
financial insightsPersonalization
Content tailored to
user personasOptimization
Strategic product
positioning
Figure 1: AI framework architecture showing the main systems (MAAMS, PAG, CHPAS) and sup-
porting features for chemical product advertising optimization.
tic architectures powered by Large Language Models (LLMs) to deliver data-driven, adaptive, and
impactful advertising strategies. The MAAMS system, orchestrated by a meta-agent, systemati-
cally gathers insights into a brand’s market presence, performance, and perception across multiple
modalities. It employs specialized agents—Text, Image, Video, Finance, and Market—each pow-
ered by LLMs to retrieve and analyze multimodal data, synthesizing a comprehensive understanding
of product positioning and consumer engagement. As illustrated in Figure 2, these agents extract
insights to evaluate brand sentiment, visual identity, emotional engagement, financial performance,
and market trends. The Text Agent, powered by LLMs, utilizes SerpAPI to retrieve data from
search engines such as Google, Bing, and Yahoo, analyzing brand sentiment, customer perception,
and messaging strategies. By leveraging advanced natural language processing, LLMs extract in-
sights from unstructured text data, generating structured summaries of key findings. Additionally,
the system assesses critical aspects such as chemical compositions, safety regulations (e.g., REACH,
GHS labeling), and compliance messaging, ensuring that advertisements align with regulatory stan-
dards and industry best practices. The Image Agent, enhanced by LLMs, extracts visual data from
platforms like Instagram and Pinterest to evaluate visual identity, audience appeal, and lifestyle as-
sociations. LLMs interpret visual content such as logos, packaging, hazard symbols, and barcodes,
ensuring both aesthetic effectiveness and regulatory compliance. They generate descriptive sum-
maries that connect visual elements to textual context and brand messaging. The Video Agent,
supported by LLMs, aggregates content from platforms like YouTube and TikTok to assess emo-
tional appeal, storytelling themes, product highlights, and cultural relevance. By processing video
transcripts, subtitles, and metadata, LLMs provide nuanced insights into how safety protocols, en-
vironmental impact, and sustainability practices are communicated, evaluating the effectiveness of
video advertisements. The Finance Agent, utilizing LLMs, gathers financial data from platforms like
Bloomberg and Yahoo Finance, analyzing revenue trends, expense breakdowns, profitability indica-
tors, and risk factors. Through financial reports, earnings calls, and market research, LLMs evaluate
financial performance and the cost-effectiveness of advertising strategies, generating actionable in-
sights. The Market Agent, powered by LLMs, collects market data from sources such as Statista
and Google Trends, focusing on customer satisfaction, market trends, competitor benchmarking,
and consumer concerns. By processing survey results, customer feedback, and competitor data,
LLMs identify emerging trends, measure customer loyalty, and provide deep insights into consumer
behavior. The Meta-Agent, leveraging LLMs as its core intelligence layer, integrates insights from
all specialized agents into a unified knowledge base. It synthesizes textual, visual, financial, and
market data to generate a comprehensive report that aligns with consumer expectations, regulatory
requirements, and industry standards. This unified knowledge directly supports two key systems: (a)
The PAG system for creating personalized multilingual advertisements. (b) The CHPAS system for
generating differentiated ads for competing products. The unified knowledge is particularly valu-
able as it combines traditional market data with specialized chemical industry requirements, such as
safety regulations and compliance standards (e.g., REACH, GHS labeling).
4

Published as a conference paper at ICLR 2025
Meta Adv Agent
Text Agent
Brand Sentiment: Tide is …..
Key Message: Tide’s mar…
Tone and Sentiment: Tide.. 
Customer Perception: Con…
Text Abstracted Knowledge
Image Agent
Brand Visual Identity: Ora..
Appeal to Audience: Tid…… 
Product Presentation:  Tide..
Lifestyle Association: Consu ..
Image Abstracted Knowledge
Comprehensive Analysis of Tide (P&G Product) and Procter & Gamble (P&G):
1.Introduction: This report provides an integrated analysis of Tide a flagship product under Procter & Gamble (P&G), and 
an…
2. Tide: Brand Insights: 2.1 Brand Sentiment: Tide enjoys overwhelmingly positive sentiment, frequently praised for its …..
3. P&G: Financial and Market Overview: 3.1 Revenue and Profitability Trends : Procter & Gamble reported steady growth...
Multimodal Agglomerated KnowledgeMultimodal Agentic Advertisement Market Survey
Finance Agent
Revenue Trend: P&G repo…
Expense Breakdown: P&G’s.
Profitability Indicators: … 
Risk Analysis: Inflationary ..Market Agent
Customer Satisfaction: Ti..
Market Trend: Surveys ind..
Competitor Benchmarking.. 
Customer Concerns: ……
Video Agent
Emotion Elicitation: Ads evo.
Storytelling Theme: Themes..
Product Feature Highlight:.. 
Cultural/Regional Eng: ……
Video Abstracted Knowledge
 Finance Abstracted Knowledge Market Abstracted Knowledge
Figure 2: The MAAMS system, led by a Meta-Agent, collects data from Text, Image, Video, Fi-
nance, and Market agents to analyze brand sentiment, visual identity, emotional engagement, finan-
cial performance, and market trends. It compiles these insights into a comprehensive report on a
product’s overall performance, marketing effectiveness, financial health, and market standing.
Simulated Humanistic 
Colony of Agents
….
“N” Individual Agentic Personas
Adv Curator Agent
Social Media AgentPersonalized Market -Aware Targeted Advertisement Generation
Comprehensive Analysis of Tide (P&G Product) and Procter & Gamble (P&G):
1.Introduction: This report provides an integrated analysis of Tide a flagship product under 
Procter & Gamble (P&G), and an…
2. Tide: Brand Insights: 2.1 Brand Sentiment: Tide enjoys overwhelmingly positive 
sentiment, frequently praised for its…..
3. P&G: Financial and Market Overview: 3.1 Revenue and Profitability Trends : Procter….
Multimodal Agglomerated Knowledge
1. Logical Strategist: Sees the world as a system of logical patterns, prioritizing clarity, …
2. Visionary Trailblazer:: Views opportunities in terms of transformative potential, focus…
3. Harmonious Connector: Sees relationships and teamwork as the foundation of success...
N-1. Resilient Optimist: Perceives obstacles as temporary and solvable through persisten …
N. Organized Architect:  Believes in the power of structured approaches and clear road…
Individual Agentic Perceptions
“N” Personalized Multilingual AdvertisementsTarget Persona 2: 
Generated Adv: 
With Tide ’s 
innovative pods 
and eco -friendly …Target Persona N -1:
Generated Adv: 
Tide isn’t just a 
detergent —it’s a 
way to show love….Target Persona N: 
Generated Adv: 
Les désordres de la 
vie n'ont aucune 
chance contre Tide.Target Persona 1 : 
Generated Adv: 
Tide delivers 
unbeatable stain 
removal …..
<email>
Subject: Unbeatable 
Cleaning Backed by 
Science
Body: Did you know<twitter>
When it comes to 
stain removal, 
science speaks for 
itself. Tide’s...
<instagram >
Logic meets 
laundry. 
  With 
Tide’s advanced …..<facebook >
Why settle for less 
when science 
guarantees more? 
Tide’s advanced ….….
Generated Social Media Post for Persona 1
Figure 3: The figure illustrates the workflow of the PAG system, designed to generate personal-
ized, multilingual advertisements tailored to diverse consumer personas. Leveraging multimodal
agglomerated knowledge, a simulated humanistic colony of agents mimics consumer personas (e.g.,
Logical Strategist, Visionary Trailblazer) to align ads with specific preferences, such as innovation,
sustainability, or emotional connections. The Adv Curator Agent creates personalized multilingual
advertisements, ensuring cultural and linguistic appropriateness for global relevance and engage-
ment. The Social Media Agent further optimizes these ads for platforms like Twitter, Instagram, and
Facebook, maximizing their impact across diverse audiences.
5

Published as a conference paper at ICLR 2025
This integrated approach ensures that the resulting advertisements are not only engaging and cultur-
ally relevant but also technically accurate and compliant with industry standards, ultimately support-
ing the creation of better-targeted and more effective advertising strategies for chemical products. In
summary, the MAAMS system provides an automated, multimodal approach to analyzing chemical
product advertising and market positioning, delivering deep insights into brand performance, con-
sumer engagement, and regulatory compliance. We present the PAG system, designed to achieve
the goal of personalized advertising by delivering highly relevant and engaging messages to indi-
vidual consumers or specific segments. As illustrated in Figure 3, the system leverages aggregated
knowledge from the MAAMS system at its core. It integrates data on product information—such
as unique features, usage instructions, and regulatory compliance—alongside brand sentiment, con-
sumer behavior, and financial performance to create tailored advertising strategies. A simulated
humanistic colony of agents enables the emulation of individuals with specific personalities, in-
terests, and goals through diverse individual agentic personas (e.g., Logical Strategist, Visionary
Trailblazer, Harmonious Connector; see Figure 3). These personas represent distinct consumer
characteristics and preferences, enabling the creation of highly tailored advertisements. For exam-
ple: The Logical Strategist values clarity, logic, and evidence-based decision-making, responding
well to ads that emphasize scientific innovation and data-driven benefits. The Visionary Trailblazer
focuses on transformative potential and innovation, preferring ads that highlight sustainability and
eco-friendly practices. The Harmonious Connector values relationships and emotional connections,
engaging with ads that evoke family values and community impact. The Resilient Optimist views
challenges as solvable and responds to ads that inspire confidence and problem-solving. The Or-
ganized Architect prioritizes structure and efficiency, preferring ads that emphasize ease of use and
dependability. By leveraging these personas, the system ensures that advertisements are highly rel-
evant, emotionally engaging, and action-oriented, aligning with the unique preferences and values
of diverse consumer segments. The Adv Curator Agent generates personalized multilingual ad-
vertisements by optimizing ads for the characteristics and preferences of the personas. It ensures
that the advertisements are culturally and linguistically appropriate while aligning with each per-
sona’s distinct traits. Simultaneously, the Social Media Agent tailors these ads for platforms like
Twitter, Instagram, and Facebook, creating platform-specific posts that resonate with the target per-
sonas’ audiences. Together, these agents ensure that the advertisements are not only personalized
but also optimized for diverse cultural contexts and platforms. By combining data-driven insights
from the MAAMS system, human-like creativity via multi-agent personas, and platform-specific op-
timization the PAG system creates persuasive advertising campaigns to drive consumer purchasing
decisions. The system leverages multimodal agglomerated knowledge to optimize ads for specific
segments, emphasizing benefits while building brand image, emotional connections, quality, and
reliability. Figure 4 demonstrates the PAG system’s capability to generate tailored, multilingual
advertisements that align with consumer preferences, behaviors, and emotional triggers across di-
verse cultural contexts while maintaining regulatory compliance. Unlike the PAG system, which
focuses on product-centric messaging and tailoring ads for specific user personas, the CHPAS sys-
tem (refer to Figure 5) is designed for competitive personalized ad optimization across multiple
products from different manufacturers within the same category. For example, it can effectively pro-
mote competing chemical products such as laundry detergents from brands like Tide, Persil, Arm &
Hammer, and Gain by tailoring advertisements to individual users’ preferences, demographics, and
behaviors. Leveraging personalized advertisements from the PAG system (see Figures 3-4), the CH-
PAS system creates competitive personalized ads that are tailored for specific user personas while
strategically highlighting each product’s competitive advantages. The system ranks products for
each persona according to affinity and competitive strengths, ensuring that the most suitable prod-
ucts are highlighted to each user. For example, for a college student interested in affordability and
trendy aesthetics, Tide might emphasize convenience and influencer-driven trust, while Gain could
focus on long-lasting freshness and social media appeal. By tailoring ads to individual preferences
and strategically positioning each product’s unique selling points, the system avoids cannibalization
and maximizes ad effectiveness. This results in a data-driven, persona-centric strategy that delivers
highly personalized, competitive, and high-performing advertisements. By aligning ads with user
preferences and optimizing for engagement, CHPAS enhances consumer action and brand loyalty
across diverse product categories.
6

Published as a conference paper at ICLR 2025
3 E XPERIMENTS
To validate the effectiveness of our proposed framework, we conducted a series of experiments in
both real-world and synthetic settings. These experiments were designed to evaluate the perfor-
mance of our multimodal, AI-driven systems in generating impactful, personalized advertisements.
We discuss synthetic data experiments in the technical appendix. Below, we describe the datasets,
evaluation metrics, experimental setup, and key findings.
3.1 D ATASETS
Developing hyper-personalized and competitive advertisements for chemical products requires ad-
dressing two key challenges: (a) Modeling diverse consumer behaviors at scale while ensuring com-
pliance with privacy regulations such as GDPR. (b) Accurately representing chemical products’
technical specifications, safety requirements, and value propositions within regulatory constraints.
To overcome these challenges, our framework integrates two complementary data sources: (a) A
Simulated Humanistic Colony of Agents, which serves as a privacy-compliant, AI-driven model
of real-world consumer behavior. (b) Real-world company and product category data, ensuring
AI-generated advertisements remain market-grounded, commercially competitive, and regulatory-
compliant. The simulated colony enables the PAG and CHPAS systems to test, optimize, and refine
advertisements across diverse consumer segments, product categories, and competitive market con-
ditions while maintaining strict privacy compliance. It does so by simulating consumers along four
key behavioral dimensions: (i) Occupational diversity: Covers office support, management, sales,
healthcare, education, and engineering (Figure 7), ensuring advertisements align with profession-
specific purchasing decisions, industry needs, and professional priorities. (ii) Emotional state diver-
sity: Agents exhibit a spectrum of psychological states, ranging from neutral to complex emotions
(Figure 8), allowing AI to evaluate advertisement resonance, engagement levels, and psychological
impact. (iii) Multilingual representation: Supports English, Spanish, Asian, European, and Mid-
dle Eastern languages (Figure 9), enabling global cultural adaptability, linguistic optimization, and
localized messaging strategies. (iv) Socioeconomic stratification: Categorizes agents into lower,
middle, and upper income classes (Figure 10), facilitating targeted ad messaging based on spend-
ing behavior, price sensitivity, and purchasing power. By integrating these demographic, emotional,
linguistic, and financial variables, the simulated colony provides privacy-compliant, synthetic be-
havioral insights, ensuring AI-generated advertisements are precisely targeted, engagement-driven,
and adaptable to dynamic market conditions. Complementing this behavioral simulation, our frame-
work incorporates real-world company and product category data, capturing competitive position-
ing, market dynamics, and regulatory constraints. The dataset spans 50+ globally recognized FMCG
companies and 2,000+ products, encompassing household and personal care items, specialty clean-
ing products, cosmetics, and many others. This integration enables advertisements to be aligned
with real-world industry benchmarks, ensuring technically accurate product representation, com-
petitive differentiation, and optimized pricing strategies. The MAAMS system serves as the central
intelligence layer, aggregating market intelligence—such as brand sentiment, financial performance,
competitive analysis, and regulatory compliance insights—to provide a data-driven foundation for
PAG and CHPAS. By leveraging retrieval-augmented multimodal analysis, the framework synthe-
sizes market intelligence with AI-driven behavioral modeling, ensuring AI-generated advertisements
are hyper-personalized, commercially competitive, linguistically and culturally adaptable, and tech-
nically robust. This dual-source approach, combining synthetic consumer behavior modeling and
real-world market intelligence, allows businesses to refine advertising strategies at scale, delivering
personalized, engagement-driven, and compliance-aware advertisements while eliminating privacy
risks and reducing the high costs associated with traditional consumer trials.
4 E VALUATION METRICS
We evaluated our framework using two key metrics: Clickability Rates and Ad Quality Scores.
Clickability Rates measure engagement levels by calculating the ratio of ad clicks to impressions for
initial, personalized, and hyper-personalized advertisements (Figure 6). Ad Quality Scores assess
advertisement quality across five dimensions: Helpfulness (usefulness and actionable information),
Correctness (accuracy and factual consistency), Coherence (logical flow and clarity), Complexity
(level of detail and sophistication), and Verbosity (conciseness and comprehensiveness). These
scores are derived from three evaluation methods: Reward Model Scoring, using models such as
NVIDIA Nemotron-4-340b-reward; LLM-as-Judge, where large language models such as GPT-4o
7

Published as a conference paper at ICLR 2025
Lucien Moreau
 Madison Taylor
 Carmen Ruiz
 Miguel Ramírez Hi, I’m Madison Taylor.
•Age: 20
•Nationality : USA
•Occupation : College student 
(Psychology major)
•Interests : Social media (Instagram, 
TikTok), fashion, skincare
•Shopping Preferences : Affordable, 
stylish, online convenience
•Ad Preferences : Short video ads, 
personalized recommendations, 
influencer collaborationsHi, I’m Lucien Moreau.
•Age: 34
•Nationality : France
•Occupation : Bakery owner 
specializing in French pastries
•Interests : Baking, foodie blogs, 
premium ingredients
•Shopping Preferences : Artisan 
tools, eco -friendly packaging, 
professional -grade supplies
•Ad Preferences : Sophisticated 
visuals, storytelling, discounts for 
professionalsHi, I’m Carmen Ruiz.
•Age: 27
•Nationality : Spain
•Occupation : Flamenco and 
contemporary dance instructor
•Interests : Music, fitness gear, 
cultural diversity
•Shopping Preferences : Activewear, 
performance costumes, music tech
•Ad Preferences : Dynamic visuals, 
catchy music, creative and vibrant 
contentHi, I’m Miguel Ramirez.
•Age: 40
•Nationality : Mexico
•Occupation : Construction project 
manager
•Interests : Soccer, grilling, DIY 
projects
•Shopping Preferences : Tools, 
outdoor gear, time -saving gadgets
•Ad Preferences : Straightforward, 
durability -focused, family -oriented 
branding
Perception
Brain
Faites briller vos 
vêtements comme vos 
créations pâtissières 
avec Tide. 
 Fabriqué 
avec soin, tout comme 
vos desserts. Propre, 
doux et respectueux de 
l'environnement. Un 
produit digne des 
artisans. #TideArtisan 
#NettoyagePremium 
#SoinDeQualitéKeep your clothes fresh 
with Tide – the 
detergent that matches 
your style and budget. 
 Whether you’re 
crushing it in class or 
taking the perfect 
selfie, Tide helps you 
look great with less 
effort! 
  #FreshFits 
#TideThatWorks 
#CollegeEssentialsCon Tide, tu ropa está 
lista para bailar. 
Un detergente que 
mantiene tu ropa de 
baile fresca y lista para 
el siguiente show. 
¿Eres como nosotros? 
¡Un movimiento 
enérgico y con estilo! 
#TideMovimiento 
#FreshDance 
#RopaImpecableTide hace frente a la 
suciedad más difícil. 
Para ropa de 
trabajo que resiste todo, 
confía en la fuerza de 
Tide. Rápido, efectivo 
y económico. ¿Por qué 
conformarse con 
menos? #TideFuerza 
#RopaResistente 
#LimpiezaImparablePlanning
Action<Planning Process of Madision >
Ad Clickability : 60%
Reason: I enjoy convenience..
Appeals: Affordability of Tide...
Improvement: The ad could 
include a trendy, relatable 
influencer using the product, …<Planning Process of Lucien>
Ad Clickability : 45%
Reason: I’m more focused on 
quality ..
Appeals: The simplicity and.. ..
Improvement: The ad should 
showcase Tide’s eco -friendly …<Planning Process of Carmen>
Ad Clickability : 55%
Reason: While I appreciate.. ..
Appeals: The simplicity and ...
Improvement The ad could 
benefit from a more dynamic, 
vibrant aesthetic with upbeat… <Planning Process of Miguel>
Ad Clickability : 50%
Reason While I understand the ..
Appeals: The practicality and ...
Improvement: The ad should 
focus more on Tide’s toughness 
and ability to clean heavy -duty …Subset from Simulated 
Humanistic Colony
of Agents
Personalized Multilingual Advertisements
Multimodal Agglomerated 
Advertisement Knowledge
Ad Curator AgentBased  on your perception,  analyze  the ad for Tide Laundry  Detergent :
1. How  likely  are you to click  on this ad (on a scale  of 0-100%), and why?
2. Which  features  of the ad appeal  to your financial  behavior  as a value -
driven  individual?
3. Suggest  improvements  to the ad to better  align  with your interests  and 
spending  power .
Now, design an ad that:
1. Aligns with their feedback and addresses their suggestions.
2. Is written in their preferred language to make it more personalized.
3. Resonates with their emotional state and cultural context.
4. Focuses on their interests and goals.
Figure 4: The figure demonstrates the PAG system for creating customized ads tailored to diverse
consumer personas. Using multimodal agglomerated knowledge, the Adv Curator Agent evalu-
ates individual preferences, cultural contexts, and feedback to produce personalized multilingual
advertisements for specific consumer segments. This approach ensures that the ads are engaging,
emotionally resonant, and effective in driving consumer action, highlighting their value for modern
personalized advertising.
Madison Taylor
(Individual Persona)
“N” 
Agentic Personas
.
.
.
.
Hi, I’m Madison Taylor.
•Age: 20
•Nationality : USA
•Occupation : College student 
(Psychology major)
•Interests : Social media (Instagram, 
TikTok), fashion, skincare
•Shopping Preferences : Affordable, 
stylish, online convenience
•Ad Preferences : Short video ads, 
personalized recommendations, 
influencer collaborationsMake laundry day effortless with 
Tide! 
  Whether you’re prepping 
for class or catching up with friends, 
Tide gives you the fresh, clean clothes 
you love with minimal effort. 
  
Affordable, reliable, and always fresh 
– just like your college life. #FreshFits 
#TideThatWorks 
#LaundryMadeEasy
Say goodbye to tough stains and hello 
to fresh! 
  Persil ProClean  gets 
your clothes spotless, so you can focus 
on what matters – studying, social 
media, and that perfect Insta shot. 
 Discover the power of clean 
with Persil, the detergent you can 
trust. #FreshAndClean #PersilPower 
#CollegeReady
Keep it clean and affordable with 
Arm & Hammer! 
  Get fresh 
clothes without breaking the bank. 
Arm & Hammer Clean Burst cleans 
your clothes while keeping your 
wallet happy, so you can save more 
for that weekend shopping spree! 
#AffordableFreshness 
#ArmAndHammerClean#Shopping
Let your clothes smell as fresh as 
your style with Gain! 
  From 
class to your next TikTok, Gain’s 
long-lasting scent keeps you feeling 
fresh all day long. Enjoy a detergent 
that works as hard as you do – with 
that extra pop of freshness. 
#GainFresh #TikTokReady 
#LaundryWithStyle
Hyper -personalized Advertisements
 Smell Fresh, Stay Stylish! Gain’s 
irresistible scent keeps you feeling 
and smelling amazing all day long. 
 Perfect for that OOTD snap or a 
night out with friends. Who says 
laundry can’t be fun? 
  
#GainFresh #LaundryGlowUp 
#ScentThatLasts
 Clean Clothes, Smart Choices! 
Why spend more when Arm & 
Hammer gives you spotless laundry 
on a student -friendly budget? 
  
Your outfits stay fresh, and your 
wallet stays happy. Ready to shop 
smarter? 
  #AffordableFresh 
#SmartLaundry #BudgetQueen
 Tough on Stains, Gentle on Style! 
Whether it ’s coffee spills or late -night 
pizza stains, Persil ProClean  is here 
to keep your outfits Insta -ready. 
  
Because life ’s too short for bad 
laundry days! 
  #ProCleanFresh 
#NoStainsAllowed #LaundryGoals
 Fresh Fits, Fresh Start! 
  
College life gets messy, but with Tide, 
laundry day is easy and affordable. 
Your favorite  influencer -approved 
detergent keeps you looking great for 
every TikTok -worthy moment! 
  
#TideThatWorks #FreshFits 
#EffortlessClean
N-Competing Company Personalized Advertisements
Focus  on affordability,  convenience,  
and influencer -driven  trust. 
Leverage  vibrant,  youthful  visuals  
to resonate  with Madison's  lifestyleHighlight  premium  stain  removal  
power  in a fun, trendy  way that 
aligns  with Madison’s  social  media -
driven  lifestyleFocus  on affordability  and 
practicality  with a playful  tone and 
visual  appeal  that resonates  with 
Madison ’s lifestyle  as a studentEmphasize  the luxurious,  fresh  scent  
with trendy  visuals  and a focus  on 
how it complements  Madison’s  
lifestyle  and aesthetic  preferencesCompetitors  focus  on premium  stain  
removal  and scent,  but Tide can 
stand  out by emphasizing  its 
everyday  practicality  and trendy  
appeal  for younger  audiencesCompetitors  focus  on affordability  
(Tide,  Arm & Hammer)  or scent  
(Gain) . Persil  can highlight  its stain -
removing  strength  while  framing  it 
in a lifestyle,  visually  appealing  wayArm & Hammer  is typically  
perceived  as practical  but less 
trendy . It can stand  out by 
emphasizing  its affordability  while  
adding  a modern,  aesthetic  twistCompetitors  emphasize  stain  
removal  (Persil)  or affordability  
(Arm  & Hammer,  Tide) . Gain  can 
stand  out by highlighting  its fresh,  
vibrant  scent  and trendy  appealMadison  values  trendy  aesthetics,  
affordability,  and influencer -driven  
campaigns . Tide’s  core strength  lies 
in affordability  and its trusted  
reputationMadison  appreciates  functionality  
but prefers  relatable,  trendy  content . 
Persil  is known  for its stain  removal  
power,  making  it an excellent  
choice  for tough  laundry  needsMadison  is budget -conscious  and 
values  practicality,  but she’s also 
drawn  to stylish  and relatable  
contentMadison  loves  freshness,  trendy  
aesthetics,  and social  media -worthy  
products . Gain’s  long-lasting  scent  
is a strong  selling  pointPersona
Analysis
Competing
Ads Gap
Marketing
Edge
Agent’s Thought Process
Figure 5: The figure demonstrates the generation of hyper-personalized advertisements for compet-
ing products within the same category from different manufacturers, tailored to a specific consumer
persona. Each advertisement emphasizes unique selling points—such as affordability, functionality,
freshness, and trendy appeal—while aligning with the persona’s preferences, lifestyle, and shopping
behaviors. The system strategically highlights each product’s competitive advantages, ensuring rel-
evance and engagement for the target audience. This approach showcases the system’s ability to
create differentiated advertisements for competing products, effectively positioning each manufac-
turer’s unique strengths while maintaining personalization and platform optimization.
8

Published as a conference paper at ICLR 2025
evaluate ads based on predefined criteria; and Human Evaluation, providing a benchmark assessment
(Figure 13). Together, these metrics ensure that advertisements are engaging, accurate, clear, well-
structured, and appropriately detailed.
5 R ESULTS AND ANALYSIS
The experimental results highlight the potential of AI-driven, multimodal frameworks to revolu-
tionize the advertising landscape. Figures 6 and 11 illustrate clickability rates, which measure the
percentage of times users click on an advertisement when it is displayed. This metric is computed as
the ratio of clicks to ad impressions. The figures compare clickability rates for three advertisement
types—initial, personalized, and hyper-personalized—across different product categories. Specifi-
cally, Figure 6 evaluates clickability rates for household cleaning and detergent brands such as Tide,
Surf Excel, Lysol, Mrs. Meyer’s Clean Day, Clorox, and OdoBan, while Figure 11 presents results
for Ariel, Gain, Fabuloso, Pine-Sol, Ajax, and Microban. Across all products, hyper-personalized
advertisements consistently achieved the highest clickability rates, followed by personalized ads,
while initial ads recorded the lowest engagement. For instance, in Figure 6, hyper-personalized
ads for Tide Detergent reached a clickability rate of approximately 92.5%, compared to 83.0%
for personalized ads and 66.0% for initial ads. A similar pattern is observed in Figure 11, where
hyper-personalized ads for Ariel Detergent achieved a clickability rate of 91.3%, outperforming
personalized ads at 81.5% and initial ads at 64.2%. These results demonstrate that generic, untar-
geted advertisements, or initial ads, are significantly less effective in capturing consumer interest
compared to persona-specific, AI-driven strategies. Figure 12 extends this analysis by comparing
clickability rates across three AI models—Anthropic, Gemini, and DeepSeek—for each advertise-
ment type. The results show a consistent trend: hyper-personalized ads perform best across all
models, followed by personalized ads, while initial ads remain the least effective. Notably, the dif-
ferences between AI models are marginal, suggesting that the degree of personalization is a more
significant factor in ad effectiveness than the specific AI model used. For instance, across different
cleaning products, the clickability rates of hyper-personalized ads generated by Anthropic, Gemini,
and DeepSeek remain within a close range, reinforcing the conclusion that hyper-personalization is
a key driver of engagement, regardless of the underlying AI model. These findings underscore the
importance of AI-driven, persona-specific advertising strategies in modern digital marketing. By
tailoring advertisements to individual consumer preferences, businesses can significantly enhance
engagement, optimize advertising expenditures, and improve overall market competitiveness. In ad-
dition, radar charts (Figures 13 and 15) compare evaluation scores for six cleaning products—Tide
Detergent, Surf Excel Detergent (Detergents), Lysol All-Purpose Cleaner, Mrs. Meyer’s Clean Day
All-Purpose Cleaner (Multi-Purpose Cleaners), Clorox Disinfecting Bleach, and OdoBan Disinfec-
tant Concentrate (Antibacterial Cleaner)—with Ariel Detergent, Gain Detergent (Laundry Deter-
gents), Fabuloso Multi-Purpose Cleaner, Pine-Sol Cleaner (Multi-Purpose Cleaners), Ajax Disin-
fecting, and Microban Disinfectant (Antibacterial Cleaner)—across five dimensions: Helpfulness,
Correctness, Coherence, Complexity, and Verbosity. Each product’s performance is assessed using
Reward Model Scoring, LLM-as-Judge evaluations, and Human Evaluation methods on a scale of 1
to 5, revealing distinct patterns between automated metrics and human judgment.
6 C ONCLUSION
Our framework demonstrates the transformative potential of AI-driven, multimodal systems in revo-
lutionizing personalized advertising for competitive markets. By integrating advanced market anal-
ysis, persona-specific targeting, and competitive differentiation, we enable the creation of hyper-
personalized advertisements that enhance consumer engagement. The results validate the frame-
work’s scalability and compliance with privacy regulations through real-world experiments with
simulated persona behavior. Future work could extend this offline framework to real-time opera-
tions, enabling dynamic optimization based on live market data and consumer interactions. This
work paves the way for data-driven, AI-powered advertising in the evolving digital landscape.
REFERENCES
Alaa Awad, Denisa Roberts, Eden Dolev, Andrea Heyman, Zahra Ebrahimzadeh, Zoe Weil, Marcin
Mejran, Vaibhav Malpani, and Mahir Yavuz. adsformers: Personalization from short-term se-
quences and diversity of representations in etsy ads. arXiv preprint arXiv:2302.01255 , 2023.
9

Published as a conference paper at ICLR 2025
Tide
DetergentSurf Excel
DetergentLysol
All-Purpose CleanerMrs Meyer's
Clean DayColorox
Disinfecting BleachOdoBan
Disinfectant
Products020406080100Clickability Rate (%)66.0068.50
62.5073.50
55.4571.2583.0084.50 84.15 84.00
80.7583.9092.5094.25 93.7595.00
91.0093.25Comparison of Clickability Rates Across Advertisement Types
Initial Advertisement Personalized Advertisement Hyper-Personalized Advertisement
Figure 6: Comparison of average clickability rates across different advertisement types for vari-
ous products. The advertisement types include initial, personalized, and hyper-personalized ads,
with clickability rates calculated across multiple consumer personas within the simulated humanis-
tic colony of agents. Hyper-personalized advertisements consistently achieve the highest clickabil-
ity rates, followed by personalized ads, while initial ads show the lowest engagement. Error bars
indicate variability in the data across different personas, highlighting the effectiveness of tailored
advertising strategies in driving consumer engagement.
Benjamin Burger et al. Accelerating discovery in natural science laboratories with ai and robotics.
arXiv preprint arXiv:2501.06847 , 2024.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng
Wang, and Haofen Wang. Retrieval-augmented generation for large language models: A survey.
arXiv preprint arXiv:2312.10997 , 2023. URL https://arxiv.org/abs/2312.10997 .
Shreeyash Gowaikar, Srinivasan Iyengar, Sameer Segal, and Shivkumar Kalyanaraman. An agen-
tic approach to automatic creation of p&id diagrams from natural language descriptions. arXiv
preprint arXiv:2412.12898 , 2024.
Shailja Gupta, Rajesh Ranjan, and Surya Narayan Singh. A comprehensive survey of retrieval-
augmented generation (rag): Evolution, current landscape and future directions. arXiv preprint
arXiv:2410.12837 , 2024. URL https://arxiv.org/abs/2410.12837 .
Haoyu Han, Yu Wang, Harry Shomer, Kai Guo, Jiayuan Ding, Yongjia Lei, Mahantesh Halap-
panavar, Ryan A. Rossi, Subhabrata Mukherjee, Xianfeng Tang, Qi He, Zhigang Hua, Bo Long,
Tong Zhao, Neil Shah, Amin Javari, Yinglong Xia, and Jiliang Tang. Retrieval-augmented
generation with graphs (graphrag). arXiv preprint arXiv:2501.00309 , 2025. URL https:
//arxiv.org/abs/2501.00309 .
Elizabeth Hellier and Judy Edworthy. Hazard communication: The importance of consistent warn-
ing signal words. Applied Ergonomics , 43(2):276–282, 2012.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich K ¨uttler, Mike Lewis, Wen tau Yih, Tim Rockt ¨aschel, Sebastian Riedel, and Douwe
Kiela. Retrieval-augmented generation for knowledge-intensive nlp tasks. arXiv preprint
arXiv:2005.11401 , 2020. URL https://arxiv.org/abs/2005.11401 .
Chunyuan Li, Zhe Gan, Zhengyuan Yang, Jianwei Yang, Linjie Li, Lijuan Wang, Jianfeng Gao, et al.
Multimodal foundation models: From specialists to general-purpose assistants. Foundations and
Trends® in Computer Graphics and Vision , 16(1-2):1–214, 2024.
10

Published as a conference paper at ICLR 2025
Junichiro Niimi. Multimodal deep learning of word-of-mouth text and demographics to predict cus-
tomer rating: Handling consumer heterogeneity in marketing. arXiv preprint arXiv:2401.11888 ,
2024.
Joni Salminen, Sangkeun Jung, Shion Chowdhury, and Bernard J Jansen. Qualitative data-driven
personas: Designing an interactive system. In Proceedings of the 2024 ACM Conference on
Human Factors in Computing Systems , 2024.
Sakhinana Sagar Srinivas, Akash Das, Shivam Gupta, and Venkataramana Runkana. Accelerat-
ing manufacturing scale-up from material discovery using agentic web navigation and retrieval-
augmented ai for process engineering schematics design. arXiv preprint arXiv:2412.05937 , 2024.
Tianxin Wei, Bowen Jin, Ruirui Li, Hansi Zeng, Zhengyang Wang, Jianhui Sun, Qingyu Yin, Han-
qing Lu, Suhang Wang, Jingrui He, and Xianfeng Tang. Towards unified multi-modal personaliza-
tion: Large vision-language models for generative recommendation and beyond. arXiv preprint
arXiv:2403.10667 , 2024.
Han Yang, Chenxi Hu, Yichi Zhou, Xixian Liu, Yu Shi, Jielan Li, Guanzhi Li, Zekun Chen,
Shuizhou Chen, Claudio Zeni, et al. Mattersim: A deep learning atomistic model across ele-
ments, temperatures and pressures. arXiv preprint arXiv:2405.04967 , 2024.
Claudio Zeni, Robert Pinsler, Daniel Z ¨ugner, Andrew Fowler, Matthew Horton, Xiang Fu, Sasha
Shysheya, Jonathan Crabb ´e, Lixin Sun, et al. Mattergen: a generative model for inorganic mate-
rials design. arXiv preprint arXiv:2312.03687 , 2023.
Sheng Zhang, Jinchao Li, Xiang Li, Lei Li, and Maosong Sun. From persona to personalization: A
survey on role-playing large language models. arXiv preprint arXiv:2404.18231 , 2024.
0-20
20-40
40-60
60-80
80-100Age Group
OthersFemaleMale
Gender012345678
CountDiversity of the Humanisitc Agents Colony
Profession
Office Support
Management
Sales
Arts & Media
Healthcare
Education
Construction
Transportation
Engineering
Figure 7: Demographic Distribution of Humanistic Agents by Profession: This figure illustrates the
demographic diversity of agents within the Humanistic Agents Colony, focusing on the intersec-
tion of age group, gender, and profession. The colony spans a wide range of professions, including
Office Support, Management, and others, ensuring broad representation for consumer research and
marketing. This diversity enables the generation of highly tailored advertisements and marketing
content targeted at various occupational backgrounds, ensuring relevance across different profes-
sional contexts and improving the precision of outreach efforts.
11

Published as a conference paper at ICLR 2025
Complex
28.5%
Positive25.5%
Negative23.5%Neutral
22.5%Distribution of Emotional States
Figure 8: The figure displays the distribution of four distinct emotional states—Complex, Positive,
Negative, and Neutral—observed within a group of simulated agents. Understanding these varied
emotional profiles is essential for evaluating advertising campaign effectiveness across a range of
psychological responses. By assessing ad performance against agents exhibiting these emotional
states, we can refine messaging and build campaigns that resonate deeply with audiences, aligning
with their emotional contexts, from simple neutrality to complex emotional experiences.
English75.0%Spanish
12.5%Asian Languages
5.0%European Languages
3.5%Middle Eastern Languages
2.0%Other
2.0%Languages Spoken in the Humanistic Colony of Agents
Figure 9: The figure depicts the distribution of languages spoken by agents in the Humanistic
Colony. The personas represent a multilingual population, with English as the dominant language
(75.0%), followed by Spanish ( 12.5%), and smaller representations of Asian languages ( 5.0%), Eu-
ropean languages ( 3.5%), Middle Eastern languages ( 2.0%), and other languages ( 2.0%), ensuring
adaptability for global advertising campaigns. This linguistic diversity supports the creation of cul-
turally relevant messaging for diverse audiences.12

Published as a conference paper at ICLR 2025
Middle Class
36.0%
Upper Class35.5%Lower Class
28.5%Distribution of Socioeconomic Classes
Figure 10: The figure depicts the socioeconomic class distribution of agents. The personas are seg-
mented into Middle Class (36%), Upper Class (35.5%), and Lower Class (28.5%), ensuring broad
representation across various financial backgrounds. This segmentation allows for targeted mes-
saging, enabling advertisements to resonate with diverse financial priorities and spending behaviors
across different income levels.
Ariel
DetergentGain
DetergentFabuloso
Multi-Purpose CleanerPine Sol
CleanerAjax
Disinfecting BleachMicroban
Disinfectant
Products020406080100Clickability Rate (%)64.2069.80
58.9070.30
57.4568.7581.5082.70
79.9583.20
78.6580.9091.3093.45
90.7594.20
89.8092.55Comparison of Clickability Rates Across Advertisement Types
Initial Advertisement Personalized Advertisement Hyper-Personalized Advertisement
Figure 11: Comparison of clickability rates across advertisement types for household cleaning prod-
ucts. The figure illustrates the average clickability rates (in percentages) for three advertisement
types—initial, personalized, and hyper-personalized—calculated across multiple consumer personas
within the simulated humanistic colony of agents. The products analyzed include Ariel Detergent,
Gain Detergent (Laundry Detergents), Fabuloso Multi-Purpose Cleaner, Pine-Sol Cleaner (Multi-
Purpose Cleaners), Ajax Disinfecting, and Microban Disinfectant (Antibacterial Cleaner). The re-
sults demonstrate that hyper-personalized advertisements consistently achieved the highest clicka-
bility rates, followed by personalized ads, while initial ads showed the lowest engagement. Error
bars indicate variability in the data across different personas.
13

Published as a conference paper at ICLR 2025
Tide
DetergentSurf Excel
DetergentLysol
All-Purpose CleanerMrs Meyer's
Clean DayColorox
Disinfecting BleachOdoBan
Disinfectant
Products5060708090100Clickability Rate (%)
66.5068.00
62.0073.00
55.0071.0083.5084.00
83.5084.50
80.0083.0092.5094.25
93.7595.00
91.0093.25
65.0067.50
61.5072.50
54.5070.5082.5083.50
83.0084.00
79.5082.5091.5093.25
92.7594.00
90.0092.25
65.5068.25
62.2573.25
55.2571.2583.0084.25
83.7584.25
80.2583.2592.0093.75
93.2594.50
90.5092.75Comparison of LLM Performance Across Advertisement Types
Anthropic Initial
Anthropic Personalized
Anthropic Hyper-personalizedGemini Initial
Gemini Personalized
Gemini Hyper-personalizedDeepSeek Initial
DeepSeek Personalized
DeepSeek Hyper-personalized
Figure 12: This figure compares the average clickability rates across three AI models—Anthropic,
Gemini, and DeepSeek—for initial, personalized, and hyper-personalized advertisements across
seven different product categories: Tide Detergent, Surf Excel Detergent (Laundry Detergents),
Lysol All-Purpose Cleaner, Mrs. Meyer’s Clean Day (Multi-Purpose Cleaners), Clorox Disinfecting
Bleach, and OdoBan Disinfectant (Antibacterial Cleaner). Clickability rates are calculated as the
average across multiple consumer personas within the simulated humanistic colony of agents. The
results indicate that hyper-personalized ads consistently achieve the highest engagement across all
AI models, followed by personalized ads, while initial ads perform the worst. The relatively small
differences between AI models suggest that the level of personalization is a more significant factor
in ad effectiveness than the specific model used. Clickability rates generally range from the low 60%
to 95%, with variation between products.
14

Published as a conference paper at ICLR 2025
HelpfulnessCorrectness
Coherence
Complexity
Verbosity12345Tide Detergent
Reward Model
LLM as Judge
Human Evaluation
HelpfulnessCorrectness
Coherence
Complexity
Verbosity12345Surf Excel Detergent
Reward Model
LLM as Judge
Human Evaluation
HelpfulnessCorrectness
Coherence
Complexity
Verbosity12345Lysol All-Purpose Cleaner
Reward Model
LLM as Judge
Human Evaluation
HelpfulnessCorrectness
Coherence
Complexity
Verbosity12345Mrs Meyer's Clean Day All Purpose Cleaner 
Reward Model
LLM as Judge
Human Evaluation
HelpfulnessCorrectness
Coherence
Complexity
Verbosity12345Colorox Disinfecting Bleach
Reward Model
LLM as Judge
Human Evaluation
HelpfulnessCorrectness
Coherence
Complexity
Verbosity12345OdoBan Disinfectant Concentrate
Reward Model
LLM as Judge
Human EvaluationComparison of Scores for Each Product Across Evaluation Methods
Figure 13: Evaluation of AI-generated advertisements for cleaning and detergent products, includ-
ing Tide Detergent, Surf Excel Detergent (Laundry Detergents), Lysol All-Purpose Cleaner, Mrs.
Meyer’s Clean Day All-Purpose Cleaner (Multi-Purpose Cleaners), Clorox Disinfecting Bleach, and
OdoBan Disinfectant Concentrate (Antibacterial Cleaner). The advertisements are assessed using
three evaluation methods: Reward Model Scoring, LLM-as-Judge Evaluation, and Human Eval-
uation. Each product is evaluated across five key metrics: Helpfulness, Correctness, Coherence,
Complexity, and Verbosity. The figure highlights variations in scores across products and evalua-
tion methods, illustrating how AI-generated advertisements perform in different aspects of clarity,
engagement, and informativeness. These insights support the optimization of competitive, persona-
specific advertisements tailored to consumer preferences.
15

Published as a conference paper at ICLR 2025
HelpfulnessCorrectness
Coherence
Complexity
Verbosity12345Mrs. Meyer's Clean Day Dish Soap
Reward Model
LLM as Judge
Human Evaluation
HelpfulnessCorrectness
Coherence
Complexity
Verbosity12345Dawn Ultra Dishwashing Liquid
Reward Model
LLM as Judge
Human Evaluation
HelpfulnessCorrectness
Coherence
Complexity
Verbosity12345Pine-Sol Multi-Surface Cleaner
Reward Model
LLM as Judge
Human Evaluation
HelpfulnessCorrectness
Coherence
Complexity
Verbosity12345Fabuloso Multi-Purpose Cleaner
Reward Model
LLM as Judge
Human Evaluation
HelpfulnessCorrectness
Coherence
Complexity
Verbosity12345Lysol Disinfectant Spray
Reward Model
LLM as Judge
Human Evaluation
HelpfulnessCorrectness
Coherence
Complexity
Verbosity12345Clorox Disinfecting Wipes
Reward Model
LLM as Judge
Human EvaluationComparison of Scores for Each Product Across Evaluation Methods
Figure 14: Comparison of AI-generated advertisements for cleaning and detergent products, in-
cluding Mrs. Meyer’s Clean Day Dish Soap, Dawn Ultra Dishwashing Liquid (Dishwashing Liq-
uids), Pine-Sol Multi-Surface Cleaner, Fabuloso Multi-Purpose Cleaner (Multi-Purpose Cleaners),
Lysol Disinfectant Spray, and Clorox Disinfecting Wipes (Disinfectants). The advertisements are
evaluated across multiple dimensions—Helpfulness, Correctness, Coherence, Complexity, and Ver-
bosity—using Reward Model Scoring, LLM-as-Judge Evaluation, and Human Evaluation. The
figure highlights variations in scores across products and evaluation methods, providing insights
into the effectiveness of AI-generated advertisements tailored to different consumer preferences and
product categories.
16

Published as a conference paper at ICLR 2025
Example of a Personalized Advertising Campaign for a Simulated Consumer Persona: Table 1 intro-
duces the simulated persona “Alice”, while Tables 2, 3, and 4 illustrate her hyper-personalized adver-
tisement, ad clickability analysis, and AI-generated social media posts, respectively. As shown in Ta-
ble 1, the simulated persona “Alice” represents a middle-income consumer with value-driven finan-
cial behavior. To effectively engage Alice and similar consumer segments, the Personalized Market-
Aware Targeted Advertisement Generation (PAG) system leverages AI-driven hyper-personalization
to optimize advertisements based on her demographic profile, interests, and spending habits. Table 2
illustrates an example of a hyper-personalized advertisement generated for Alice. The ad aligns with
her interests in fashion and travel while emphasizing sustainability and affordability. It dynamically
adjusts messaging, visuals, and call-to-action strategies to enhance engagement, reflecting a person-
alized marketing approach tailored to her lifestyle preferences. To assess the effectiveness of such
targeted advertisements, Table 3 presents an analysis of ad clickability, evaluating strengths such
as direct personalization, emotional appeal, and visually engaging content. The results indicate a
high engagement score, with key areas for improvement identified to further optimize consumer
response, such as stronger fashion-centric messaging and promotional incentives. Beyond static
advertisements, Table 4 highlights AI-generated, platform-specific social media posts designed for
Twitter, Instagram, Facebook, and email marketing. These posts incorporate tailored messaging,
audience-specific engagement strategies, and platform-adapted content to ensure a seamless, cross-
channel advertising experience. By integrating hyper-personalization with audience insights, the
system enhances brand engagement and maximizes advertising effectiveness across diverse digital
ecosystems. Together, these components demonstrate the power of AI-driven, data-informed adver-
tising strategies in creating highly relevant, impactful marketing campaigns. By modeling consumer
personas like Alice and continuously optimizing advertisements through engagement metrics, busi-
nesses can enhance consumer interactions, improve conversion rates, and drive targeted outreach at
scale.
Attribute Value
Name Alice
Age 30
Gender Female
Interests Fashion, Travel
Socioeconomic Class Middle-income
Spending Power Moderate
Financial Behavior Value-driven
Language English
Cultural Context Western Europe
Emotional State Excited
Goal Explore
Table 1: Example of a simulated consumer persona, “Alice”, used in the Simulated Humanistic
Colony of Agents within the Personalized Market-Aware Targeted Advertisement Generation (PAG)
system. This persona represents a 30-year-old female from Western Europe with moderate spending
power and a value-driven financial approach. By modeling attributes such as interests, emotional
state, and cultural context, AI-driven systems optimize hyper-personalized advertisements tailored
to distinct consumer segments.
17

Published as a conference paper at ICLR 2025
Personalized Advertisement
Ad Title Experience the Magic of Family Moments with Surf Excel
Visual Description A vibrant, colorful image of a family enjoying a sunny picnic in a
beautiful park. The family members are dressed in trendy outfits, and
there are playful stains on their clothes visible as they laugh and share
joyful moments together. In the background, a stylish travel bag is placed
nearby, hinting at their adventurous spirit.
Headline “Adventure Awaits! Keep Your Family’s Style Spotless with Surf Excel”
Body Text Hey Alice!
Life’s too short to worry about stains when you’re making unforgettable
memories with your loved ones! At Surf Excel, we believe in celebrating
life and embracing every moment, whether you’re traveling to
breathtaking destinations or enjoying a lovely day outdoors.
What Makes Surf Excel Perfect for You:
•Eco-Friendly Cleaning : Our products are crafted with care for
both your clothes and the planet. Enjoy stain-free adventures
while doing your part for the environment!
•Stylish Solutions : With our sleek packaging and color-friendly
formulas, keeping your favorite outfits looking fresh and
fashionable has never been easier.
•Real Family Stories : Join families who trust Surf Excel to
tackle stains from every adventure. Check out testimonials that
highlight how our detergent has become a travel companion for
lasting memories!
Cost-Effective Performance : Get the quality and reliability you deserve
without breaking the bank. Invest in products that enrich your lifestyle and
keep your fashion game strong!
Call to Action “Shop Now and Get Ready for Your Next Adventure!”
Footer Include social media icons and a hashtag: #SurfExcelMoments — Visit
Us: www.surfexcel.com
Visual Note Include earthy tones and trendy design elements to appeal visually while
integrating eco-conscious graphics, like soft leaves or water droplets, to
emphasize sustainable practices.
Closing Note This ad resonates with your excitement for exploring life’s moments,
aligns with your interests in fashion and travel, and emphasizes
eco-friendly values, making it not just an advertisement but an invitation
to a lifestyle you love!
Table 2: Example of a hyper-personalized advertisement generated by the Personalized Market-
Aware Targeted Advertisement Generation (PAG) system. This advertisement is tailored for “Alice,”
a simulated persona with interests in fashion and travel, moderate spending power, and an eco-
conscious mindset. It incorporates dynamic content, including personalized messaging, tailored
visuals, and platform-specific adaptations, to enhance engagement and drive conversions.
18

Published as a conference paper at ICLR 2025
Ad Clickability Analysis
Overall Rating 85% Clickability Score
Strengths
1.Personalization The ad directly addresses “Alice”, which creates a personal
connection and makes it feel tailored to the recipient. This can
significantly enhance engagement.
2.Visual Appeal The description of the visual elements, such as a vibrant family
picnic and trendy outfits, aligns well with Alice’s interests in
fashion and travel. A colorful and joyful image can attract
attention and elicit positive emotions, which may lead to
increased click-through rates.
3.Clear Value Proposition The ad effectively communicates several benefits of using Surf
Excel, including eco-friendliness, stylish packaging, and
testimonials from real families. This not only highlights the
product’s features but also addresses a potential concern for
consumers—keeping clothes clean while having fun.
4.Emotional Appeal The ad taps into the excitement of making memories and
adventures, which resonates with Alice’s emotional state of
being excited. It invites her to embrace life and its messes,
framing stains as a natural part of family fun.
5.Strong Call to Action The call to action (“Shop Now and Get Ready for Your Next
Adventure!”) is inviting and motivating. It encourages
immediate action, which can lead to higher click rates.
Areas for Improvement
1.Focus on Fashion While fashion is mentioned, more emphasis on how Surf Excel
preserves trendy clothing could enhance resonance with
fashion-conscious consumers. Specific examples of fashionable
brands or styles that work well with Surf Excel might strengthen
the appeal further.
2.Additional Incentives Introducing a limited-time offer, discount, or free sample could
improve clickability. While the ad mentions cost-effectiveness, a
specific incentive could create urgency and better encourage
clicks.
3.Visual Enhancements The visual note provides good suggestions, but incorporating a
more specific visual example, such as showing the product being
used in the context of a fashionable outing or adventure, might
enhance interest.
Conclusion Overall, the ad is well-crafted and effectively targets Alice’s
interests and emotional state, resulting in a high clickability
score. However, slight adjustments could make it even more
compelling.
Table 3: Clickability analysis for a personalized advertisement targeting the simulated persona “Al-
ice.” The table highlights key strengths, including personalization, visual appeal, and emotional
engagement, contributing to an 85% clickability score. Areas for improvement, such as a stronger
emphasis on fashion-related messaging and additional incentives, are identified to further optimize
performance.
19

Published as a conference paper at ICLR 2025
Social Media Posts
Twitter ”Life’s too short to worry about stains while making memories! Embrace every
adventure with your loved ones and let Surf Excel keep your family stylish &
eco-friendly. Ready for a stain-free journey? #SurfExcelMoments ”
Instagram Family Adventures Await!
Dive into a world of fun and unforgettable moments with your loved ones! At
Surf Excel, we celebrate life’s joys while ensuring your family’s favorite outfits
stay spotless and stylish. Our eco-friendly cleaning solutions mean you can
adventure freely—no worrying about stains!
Picture this: a sunny picnic in the park, laughter echoing around, and beautiful
outfits shining bright, even with a few playful stains!
Join the Surf Excel family and share in the real stories of families who trust us
to keep their clothes fresh. Your journey to stain-free memories starts now!
Shop Now for Your Next Adventure!
#SurfExcelMoments #EcoFriendly #FamilyFun #StainFreeMemories
Facebook Hey friends!
Are you ready to create beautiful memories with your family without worrying
about those pesky stains? At Surf Excel, we believe every adventure deserves
to be celebrated! Whether you’re stylishly picnicking in the park or exploring
breathtaking destinations, we’ve got your back!
Why Choose Surf Excel?
• Eco-Friendly Cleaning
• Stylish Solutions
• Cost-Effective Performance
Join thousands of families who trust Surf Excel for their stain-fighting needs.
Check out our powerful products, designed to keep your clothes looking fresh
and vibrant through all of life’s adventures!
Shop Now and Get Ready for Your Next Adventure!
#SurfExcelMoments #Family #Adventure #EcoFriendly
Email Subject: Embrace Every Adventure with Surf Excel!
Dear [Recipient’s Name],
Life’s most precious moments are often filled with laughter, joy, and the
occasional mess! At Surf Excel, we understand the importance of creating
unforgettable memories with your loved ones without the worry of stains
ruining your favorite outfits.
Why Surf Excel is Right for You:
•Eco-Friendly Cleaning : Make a positive impact on the planet while
keeping your clothes looking pristine.
•Stylish Solutions : Our product packaging is not only sleek but
ensures your clothes remain fresh and fashionable.
•Real Family Stories : Join the community of families who rely on
Surf Excel to conquer stains from their most cherished adventures.
And let’s not forget our cost-effective performance that doesn’t compromise on
quality.
Ready to keep your family stylish and those memories stain-free?
Shop now and embrace every adventure with confidence and style! [Link
to shop]
Table 4: Example of AI-generated social media posts optimized for Twitter, Instagram, Facebook,
and email marketing. These posts leverage personalized messaging, audience-specific engagement
strategies, and platform-tailored content to maximize ad effectiveness. By aligning with the per-
sona’s interests and emotional state, the social media campaign enhances engagement, click-through
rates, and brand perception.
20

Published as a conference paper at ICLR 2025
A A PPENDIX
A.1 A DCOPY OPTIMIZATION
Ad Copy Optimization is a critical component of modern marketing strategies, designed to maxi-
mize the effectiveness of advertising campaigns by refining and testing multiple ad variations. This
process builds upon the Competitive Hyper-Personalized Advertisement System (CHPAS), which
generates persona-specific, competitive advertisements for products within the same category from
different manufacturers. Ad Copy Optimization enhances these ads by systematically evaluating
their effectiveness across diverse consumer personas, ensuring they resonate with target customers
while maintaining competitive differentiation. This optimization process is essential for improving
key performance metrics such as click-through rates (CTR), which measure the percentage of users
who click on an ad after viewing it, and conversions, which track the number of users who take a de-
sired action, such as making a purchase or signing up for a service. By addressing diverse consumer
preferences, platform-specific engagement requirements, and rapidly evolving market dynamics, Ad
Copy Optimization ensures that marketing messages maximize engagement and drive meaningful
customer action. To ground our experimental evaluation, we employ a diverse colony of Simu-
lated Humanistic Agentic Personas (SHAP), which assess CHPAS-generated ads. These personas
represent distinct consumer archetypes with well-defined demographic profiles, behavioral patterns,
and decision-making frameworks, including: (a) Analytical Evaluator, which focuses on technical
accuracy, factual consistency, and logical appeal; (b) Emotional Resonance Assessor, which mea-
sures emotional impact, brand connection, and relatability; (c) Cultural Context Validator, which
ensures cultural relevance, sensitivity, and appropriateness for diverse audiences; and (d) Consumer
Preference Analyzer, which evaluates alignment with consumer preferences and behavioral patterns.
The experimental framework systematically evaluates CHPAS-generated ad variations by aligning
them with the unique preferences of these personas and the unique selling points (USPs) of the
products. Audiences are segmented for randomized testing across platforms, ensuring a compre-
hensive evaluation of ad effectiveness through multi-dimensional analysis. Each agentic persona
rates advertisements across multiple dimensions using a standardized evaluation framework, includ-
ing: (a) Messaging Clarity (0–10); (b) Emotional Resonance (0–10) (c) Technical Accuracy (0–10);
(d) Cultural Fit (0–10). To enhance the robustness of the evaluation process, inter-persona valida-
tion is employed, where multiple agents cross-validate each other’s assessments using sophisticated
validation protocols. This ensures reliability and minimizes bias in the results while maintaining
statistical significance. Persona ratings are then analyzed with statistical rigor to identify the most
effective ads, ensuring that the findings are both significant and actionable for real-world implemen-
tation. The integration of these persona-based evaluations provides a deep understanding of ad effec-
tiveness, enabling businesses to iteratively refine advertisements and improve engagement metrics
through data-driven optimization. By leveraging these multi-dimensional insights, chemical product
manufacturers can reduce guesswork, adapt to evolving market trends, and allocate resources more
effectively to maximize return on investment (ROI) and market penetration. This approach under-
scores the critical role of data-driven, persona-validated advertising in modern marketing strategies
and competitive differentiation. In summary, Ad Copy Optimization serves as the refinement layer
for CHPAS-generated advertisements, enhancing their effectiveness by incorporating simulated hu-
manistic feedback, standardized evaluation metrics, and rigorous statistical analysis. This iterative
approach ensures that marketing efforts remain competitive, personalized, and aligned with the dy-
namic needs of diverse consumer segments, ultimately driving higher engagement and improved
conversion rates across multiple market segments.
A.2 O PEN-DOMAIN QUESTION ANSWERING (ODQA)
This section presents an advanced Optimized Retrieval-Augmented Generation (RAG) system Lewis
et al. (2020); Gupta et al. (2024); Gao et al. (2023); Han et al. (2025), designed to process and utilize
multimodal agglomerated knowledge from the Multimodal Agentic Advertisement Market Survey
(MAAMS) system for question-answering (QA) tasks. The knowledge synthesized by MAAMS
comprises market intelligence from diverse modalities, including text, images, video, financial data,
and market data, which is consolidated into a document-specific text output format for each product.
While the system primarily processes textual knowledge extracted from documents, it enables se-
mantic search and contextually accurate response generation for QA tasks. A QA system built on this
agglomerated knowledge base acts as a crucial interface between customers and technical product
21

Published as a conference paper at ICLR 2025
information, allowing users to receive precise and relevant answers without navigating lengthy doc-
umentation. The system excels in four key areas: (a) Providing direct access to product information;
(b) Offering comparative product insights; (c) Translating technical specifications into user-friendly
language; (d) Delivering specific usage guidance. By extracting and presenting relevant information
in an accessible format, the QA system enhances the customer experience, making technical product
information more actionable and user-friendly. The system begins by processing documents using an
automated text extraction method, ensuring the structured retrieval of textual content and metadata.
The extracted text is segmented into manageable, semantically coherent chunks with overlapping re-
gions to maintain contextual continuity. To improve context awareness, each chunk is summarized
for relevance in relation to other chunks within the same document, a process known as Contextual
Relevance Summarization (CRS). These augmented chunks prioritize semantically aligned content,
enhance long-context handling, and strengthen chain-of-thought reasoning. This refined retrieval in
RAG pipelines minimizes noise, improves precision, and enhances multi-hop retrieval and source
attribution. While summarization increases computational costs, the benefits of improved retrieval
accuracy and reasoning outweigh the trade-offs. Each augmented chunk is converted into a numer-
ical vector representation using OpenAI’s text-embedding-3-small model and indexed in a vector
store for efficient cosine similarity-based retrieval. User queries are embedded into the same vector
space and matched against stored embeddings to retrieve the most contextually relevant chunks. Re-
sponses are synthesized using OpenAI’s GPT-4o-mini model, integrating the top-K retrieved chunks
to enhance factual accuracy, contextual relevance, and coherence through iterative refinement. To
further improve response quality, the system incorporates an iterative self-reflection mechanism,
where LLM-as-Judge (using OpenAI’s GPT-4o) critiques generated responses for potential weak-
nesses, such as factual inaccuracies, missing details, or lack of clarity. Based on these critiques,
responses are iteratively revised until they meet either a predefined number of iterations or a qual-
ity threshold, ensuring optimal output. To evaluate performance, reference responses are generated
using GPT-4o, serving as a benchmark for assessing alignment and correctness. A comprehensive
evaluation framework measures retrieval and response quality using Precision, Recall, Mean Re-
ciprocal Rank (MRR), Mean Average Precision (MAP), BLEU, METEOR, ROUGE, and cosine
similarity. Faithfulness is assessed by comparing the semantic similarity between generated re-
sponses and retrieved chunks, ensuring alignment with source content, while relevance is evaluated
by comparing the embeddings of generated responses with original query embeddings. Addition-
ally, a state-of-the-art reward model (NVIDIA Nemotron-4-340b-reward) evaluates and scores the
final responses, optimizing for user satisfaction and practical utility. The reward model assesses
responses based on five key metrics: (a) Helpfulness (effectiveness in addressing the question); (b)
Correctness (accuracy and inclusion of pertinent facts); (c) Coherence (clarity and logical flow);
(d) Complexity (intellectual depth and domain expertise); (e) Verbosity (appropriate level of de-
tail relative to the question). By combining retrieval-augmented generation, self-refinement, and
multimodal processing, this AI-powered knowledge engine serves as a strategic decision-support
system, enabling companies to extract, synthesize, and optimize market insights. It automates com-
petitive analysis, facilitates data-driven decision-making, and retrieves multimodal knowledge for
benchmarking products, refining branding, generating targeted ad content, and ensuring regulatory
compliance. Marketing executives, product managers, brand analysts, and compliance teams lever-
age the system as an enterprise chatbot, API, semantic search engine, and advertising optimization
platform to enhance knowledge retrieval, consumer engagement, and strategic effectiveness. For
evaluation, a benchmark QA dataset of 2,500 QA pairs is created using GPT-4o, refined through
human review to ensure accuracy and relevance, serving as a benchmark for the Optimized RAG
system. In summary, this AI-powered engine serves as a strategic decision-support tool, automating
competitive analysis, improving branding, and enabling data-driven marketing, making it a scalable
and intelligent solution for enterprise use in marketing, compliance, and consumer engagement.
A.3 R ESULTS
Figure 16 presents the retrieval metrics used to evaluate the performance of the Optimized RAG
system across six key indicators. The Average Relevance Score (scale: 0-1) quantifies the alignment
of retrieved chunks with the query, where higher values indicate better relevance. The Retrieved
Chunk Count measures the number of chunks retrieved per query, which varies based on retrieval
settings. Precision@1 and Precision@3 (scale: 0-1) assess the proportion of relevant chunks within
the top-1 and top-3 retrieved results, respectively, with Precision@1 focusing on the relevance of
the first result and Precision@3 evaluating the top three results. Recall@3 (scale: 0-1) measures
22

Published as a conference paper at ICLR 2025
Analytical Evaluator
Emotional Resonance AssessorCultural Context Validator
Consumer Preference Analyzer
Evaluator Types0246810Advertisement Evaluation Score (0-10)9.2
8.58.78.98.88.9
8.58.79.0
8.68.8
8.58.79.1
8.98.8Product Advertisement Evaluation Scores by Evaluator Type
with Variation Across Ad Versions
Tide Detergent
Surf Excel
Lysol Cleaner
Mrs. Meyer's
Figure 15: A comparative analysis of product advertisements across different evaluator types, show-
ing mean evaluation scores and variations across multiple ad versions. Error bars represent standard
deviations in scores across different advertisement variations for each product-evaluator combina-
tion. The plot illustrates how each evaluator persona (Analytical Evaluator, Emotional Resonance
Assessor, Cultural Context Validator, and Consumer Preference Analyzer) assesses different ver-
sions of advertisements for each product, offering insights into both the average effectiveness and
consistency of advertising approaches.
Average Relevance Score
Retrieved Chunks CountPrecision at 1 Precision at 3Recall at 3
Mean Reciprocal Rank (MRR)Mean Average
 Precision (MAP)Similarity Score (Quality)012345
Retrieval Metrics
BLEU
 Score METEOR Score ROUGE-1 F-ScoreROUGE-2 F-ScoreROUGE-L F-ScoreSimilarity Score0.00.10.20.30.40.50.60.70.8Response Quality Metrics
Faithfulness ScoreRelevance Score0.00.20.40.60.8Other Metrics
Helpfulness CorrectnessCoherence ComplexityVerbosity0.00.51.01.52.02.53.03.5Nvidia Evaluation Scores
T echnical
 AccuracyClarity
CompletenessPractical Utility Source
 Integration0246810GPT Evaluation Scores
Avg GPT Score
Avg Nvidia Score
Avg Response Quality012345678Overall Evaluation Summary
Figure 16: A comprehensive evaluation of the Optimized RAG System. Subfigures 1–3 analyze
retrieval performance through precision, recall, and relevance, ensuring reliable and contextually
relevant responses. Subfigures 4–5 present AI-driven evaluations using NVIDIA’s Nemotron and
GPT-based scoring, assessing correctness, completeness, and practical utility. Subfigure 6 sum-
marizes overall performance, highlighting the balance between accurate retrieval and high-quality
response generation.
23

Published as a conference paper at ICLR 2025
the proportion of all relevant chunks retrieved within the top-3 results compared to the total relevant
chunks in the dataset, emphasizing retrieval coverage. Mean Reciprocal Rank (MRR) (scale: 0-1)
evaluates the rank position of the first relevant chunk, with higher values indicating faster retrieval
of relevant information. Mean Average Precision (MAP) (scale: 0-1) aggregates precision scores
across multiple recall levels, providing a comprehensive measure of retrieval performance. High
values in the Average Relevance Score, Precision@k, MRR, and MAP indicate effective retrieval,
ensuring high-quality context for response generation. While a high Retrieved Chunk Count im-
proves recall, it may introduce noise, whereas a low count risks limiting information coverage. This
figure demonstrates the system’s ability to optimize retrieval accuracy and relevance, which is cru-
cial for improving response synthesis. Response quality metrics assess the coherence, accuracy, and
informativeness of generated responses by comparing them to benchmark responses from GPT-4o.
The BLEU Score (scale: 0-1) measures n-gram overlap with reference answers, where higher values
indicate better alignment. The METEOR Score (scale: 0-1) accounts for synonym matching, flu-
ency, and recall, improving upon BLEU by considering linguistic variations. ROUGE-1, ROUGE-2,
and ROUGE-L F-Scores (scale: 0-1) assess token overlap, capturing unigram and bigram precision-
recall relationships. The Similarity Score (scale: 0-1) quantifies the semantic alignment between
generated and reference responses, ensuring contextual consistency. High values across these met-
rics indicate superior response generation with enhanced readability and factual alignment. Faith-
fulness and Relevance Scores measure response reliability and query alignment. The Faithfulness
Score (scale: 0-1) assesses how well the generated response adheres to retrieved source content,
ensuring factual integrity and reducing hallucination risks. The Relevance Score (scale: 0-1) evalu-
ates the semantic similarity between the generated response and the original query, determining how
well the system addresses user intent. Higher scores in both metrics indicate that the RAG system
remains grounded in retrieved evidence, improving trustworthiness and response credibility. The
NVIDIA evaluation scores present response assessments from NVIDIA’s Nemotron-4-340b-reward
model, which scores responses on a scale of 0-4, where higher values indicate superior quality. The
key evaluation metrics include Helpfulness, Correctness, Coherence, Complexity, and Verboseness.
Helpfulness measures how well responses address queries, while Correctness evaluates factual ac-
curacy and completeness. Coherence assesses logical structuring and readability, while Complexity
considers the depth of reasoning and domain knowledge integration. Verboseness ensures a bal-
ance between conciseness and necessary detail. Observed scores for individual responses range
from approximately 0.5 to 3.5 across these dimensions. Higher values indicate well-structured, in-
formative, and contextually accurate responses, while lower values suggest deficiencies in clarity,
completeness, or factual grounding. The GPT evaluation scores assess response quality using GPT-
4-based metrics, covering Technical Accuracy, Clarity, Completeness, Practical Utility, and Source
Integration. Technical Accuracy ensures domain-specific precision, Clarity measures readability
and logical flow, and Completeness evaluates the coverage of key aspects. Practical Utility assesses
real-world applicability, while Source Integration verifies grounding in retrieved evidence. Scores
range from 0 (poor) to approximately 10 (excellent), with higher values indicating more coherent,
informative, and factually robust responses. Lower scores suggest weaker grounding, incomplete
information,
A.4 O PEN-DOMAIN QUESTION ANSWERING (ODQA)
This section presents an advanced Optimized Retrieval-Augmented Generation (RAG) system Lewis
et al. (2020); Gupta et al. (2024); Gao et al. (2023); Han et al. (2025), designed to process and utilize
multimodal agglomerated knowledge from the Multimodal Agentic Advertisement Market Survey
(MAAMS) system for question-answering (QA) tasks. The knowledge synthesized by MAAMS
comprises market intelligence from diverse modalities, including text, images, video, financial data,
and market data, which is consolidated into a document-specific text output format for each product.
While the system primarily processes textual knowledge extracted from documents, it enables se-
mantic search and contextually accurate response generation for QA tasks. A QA system built on this
agglomerated knowledge base acts as a crucial interface between customers and technical product
information, allowing users to receive precise and relevant answers without navigating lengthy doc-
umentation. The system excels in four key areas: (a) Providing direct access to product information;
(b) Offering comparative product insights; (c) Translating technical specifications into user-friendly
language; (d) Delivering specific usage guidance. By extracting and presenting relevant information
in an accessible format, the QA system enhances the customer experience, making technical product
information more actionable and user-friendly. The system begins by processing documents using an
24

Published as a conference paper at ICLR 2025
automated text extraction method, ensuring the structured retrieval of textual content and metadata.
The extracted text is segmented into manageable, semantically coherent chunks with overlapping re-
gions to maintain contextual continuity. To improve context awarenesor reduced practical relevance.
The overall evaluation summary consolidates the performance of the Optimized RAG system, dis-
playing key evaluation metrics. The Avg GPT Score (scale: 0-10) reflects language model-based
evaluation, ensuring linguistic and contextual quality. The Avg NVIDIA Score (scale: 0-4) aggre-
gates reward model evaluations, capturing user-centric response effectiveness. The Avg Response
Quality (scale: 0-1) combines retrieval precision and response coherence, offering a comprehensive
measure of system performance. Higher averages across these metrics indicate a balanced trade-off
between accurate retrieval and high-quality response generation. In summary, this set of figures
presents a comprehensive evaluation of the Optimized RAG system, covering retrieval accuracy,
response quality, faithfulness, relevance, and benchark model-based assessments. In Figure 16,
Subfigures 1–3 analyze retrieval precision, response coherence, and factual grounding, ensuring the
system’s reliability and contextual relevance. Subfigures 4–5 showcase AI-driven evaluations from
NVIDIA’s Nemotron and GPT-based scoring, emphasizing correctness, completeness, and utility.
Subfigure 6 integrates these insights into an overall performance summary, highlighting the balance
between accurate retrieval and high-quality response generation.
A.5 M ATHEMATICAL MODELING
The proposed framework for multimodal analysis and personalized advertisement generation in
the chemical industry is formalized through a structured mathematical representation, captur-
ing the key components of the Multimodal Agentic Advertisement Market Survey (MAAMS),
Personalized Market-Aware Targeted Advertisement Generation (PAG), and Competitive Hyper-
Personalized Advertisement System (CHPAS). The MAAMS system integrates insights from
multiple specialized agents—Text, Image, Video, Finance, and Market—represented as M=
{Text,Image ,Video ,Finance ,Market }. Each agent m∈ M processes domain-specific raw data
Dm={dm
1, dm
2, . . . , dm
Nm}, such as search engine text, social media images, and financial reports,
into structured insights via a transformation function fm:Dm→ I m, where Im=fm(Dm).
The aggregated insights form a multimodal knowledge base K=S
m∈MIm, which is further
synthesized by a Meta-Agent Ameta, leveraging large language models (LLMs), to produce a mar-
ket intelligence report R=Ameta(K). This structured synthesis enables efficient data collection
and decision-oriented insights, driving personalized advertisement generation and competitive ad
differentiation in the PAG and CHPAS systems. The PAG system utilizes the market intelligence
report R, synthesized from the multimodal knowledge base K, to generate personalized advertise-
ments for Nconsumer personas, denoted as P={p1, p2, . . . , p N}. Each persona p∈ P is as-
sociated with a preference set Cp, capturing its traits, interests, and goals (e.g., Logical Strategist,
Visionary Trailblazer). The Advertisement Curator Agent Acurator generates persona-specific adver-
tisements ADpusing structured insights from R, formulated as Acurator(R,Cp) =ADp. These
advertisements are further optimized for specific platforms S={Twitter ,Instagram ,Facebook }
by the Social Media Agent Asocial, producing platform-adapted advertisements ADp,S, represented
asAsocial(ADp,S) =ADp,S. This ensures that advertisements are persona-specific, leveraging
refined insights from R, and optimized for platform constraints, cultural relevance, and linguis-
tic appropriateness. The CHPAS system extends personalization to competing products, denoted
asQ={q1, q2, . . . , q M}, where each qirepresents a specific product (e.g., laundry detergents
from different manufacturers). For each persona p, the system generates hyper-personalized ad-
vertisements ADp,qbased on affinity scores αp,qand competitive strengths βq. The affinity score
αp,qquantifies the alignment between persona preferences Cpand product features, computed as
αp,q=faffinity(Cp,R, q), where faffinity evaluates the relevance of product qto persona p. The com-
petitive strength βqcaptures the product’s unique selling points, computed as βq=fstrength (R, q),
where fstrength assesses the competitiv
A.6 O PEN-DOMAIN QUESTION ANSWERING (ODQA)
This section presents an advanced Optimized Retrieval-Augmented Generation (RAG) system Lewis
et al. (2020); Gupta et al. (2024); Gao et al. (2023); Han et al. (2025), designed to process and utilize
multimodal agglomerated knowledge from the Multimodal Agentic Advertisement Market Survey
(MAAMS) system for question-answering (QA) tasks. The knowledge synthesized by MAAMS
25

Published as a conference paper at ICLR 2025
comprises market intelligence from diverse modalities, including text, images, video, financial data,
and market data, which is consolidated into a document-specific text output format for each product.
While the system primarily processes textual knowledge extracted from documents, it enables se-
mantic search and contextually accurate response generation for QA tasks. A QA system built on this
agglomerated knowledge base acts as a crucial interface between customers and technical product
information, allowing users to receive precise and relevant answers without navigating lengthy doc-
umentation. The system excels in four key areas: (a) Providing direct access to product information;
(b) Offering comparative product insights; (c) Translating technical specifications into user-friendly
language; (d) Delivering specific usage guidance. By extracting and presenting relevant information
in an accessible format, the QA system enhances the customer experience, making technical product
information more actionable and user-friendly. The system begins by processing documents using an
automated text extraction method, ensuring the structured retrieval of textual content and metadata.
The extracted text is segmented into manageable, semantically coherent chunks with overlapping
regions to maintain contextual continuity. To improve context awarenes
A.7 O PEN-DOMAIN QUESTION ANSWERING (ODQA)
This section presents an advanced Optimized Retrieval-Augmented Generation (RAG) system Lewis
et al. (2020); Gupta et al. (2024); Gao et al. (2023); Han et al. (2025), designed to process and utilize
multimodal agglomerated knowledge from the Multimodal Agentic Advertisement Market Survey
(MAAMS) system for question-answering (QA) tasks. The knowledge synthesized by MAAMS
comprises market intelligence from diverse modalities, including text, images, video, financial data,
and market data, which is consolidated into a document-specific text output format for each product.
While the system primarily processes textual knowledge extracted from documents, it enables se-
mantic search and contextually accurate response generation for QA tasks. A QA system built on this
agglomerated knowledge base acts as a crucial interface between customers and technical product
information, allowing users to receive precise and relevant answers without navigating lengthy doc-
umentation. The system excels in four key areas: (a) Providing direct access to product information;
(b) Offering comparative product insights; (c) Translating technical specifications into user-friendly
language; (d) Delivering specific usage guidance. By extracting and presenting relevant information
in an accessible format, the QA system enhances the customer experience, making technical product
information more actionable and user-friendly. The system begins by processing documents using an
automated text extraction method, ensuring the structured retrieval of textual content and metadata.
The extracted text is segmented into manageable, semantically coherent chunks with overlapping
regions to maintain contextual continuity. To improve context awarenese advantages of product
q. The CHPAS system generates hyper-personalized advertisements as Acomp(R,Cp, q) =ADp,q,
ensuring that advertisements are not only tailored to individual preferences but also strategically
highlight each product’s competitive advantages. The effectiveness of the generated advertisements
is evaluated using AI-driven and human-aligned metrics to ensure they are engaging, relevant, and
persuasive. Specifically, the evaluation incorporates helpfulness H(ADp,q), correctness C(ADp,q),
and coherence R(ADp,q), while faithfulness F(ADp,q)measures alignment with retrieved market
intelligence, and relevance V(ADp,q)quantifies how well the advertisement matches consumer ex-
pectations. These scores are derived from NVIDIA Nemotron-4-340b-reward and LLM-as-Judge,
evaluation models. The optimization objective is to maximize the weighted sum of these metrics:
max
ADp,q(w1H(ADp,q) +w2C(ADp,q) +w3R(ADp,q) +w4F(ADp,q) +w5V(ADp,q)),
where w1, w2, w3, w4, w5are hyperparameters reflecting the relative importance of each factor. By
integrating multimodal AI analysis, retrieval-augmented market insights, and persona-driven opti-
mization, the framework ensures that advertisements are not only technically accurate and engaging
but also strategically competitive and culturally relevant.
A.8 E XPERIMENTAL SETUP
The experimental validation integrates real-world environments to evaluate our three-system frame-
work: (1) MAAMS (Multimodal Agentic Advertisement Market Survey) for market analysis, (2)
PAG (Personalized Market-Aware Targeted Advertisement Generation) for personalized multilin-
gual advertising, and (3) CHPAS (Competitive Hyper-Personalized Advertisement System) for com-
petitive ad differentiation. The MAAMS system is orchestrated by a Meta-Agent powered by GPT-
4o, responsible for cross-modal synthesis and market intelligence aggregation through specialized
26

Published as a conference paper at ICLR 2025
agents. The Text Agent (GPT-4o + SerpAPI) retrieves and analyzes consumer sentiment, brand
messaging, and regulatory compliance insights from Google, Bing, and Yahoo. The Image Agent
(GPT-4o + SerpAPI + Instagram and Pinterest APIs) conducts visual content analysis, fetching im-
ages from search queries to evaluate brand identity, packaging, and product perception. The Video
Agent (GPT-4o + YouTube API) extracts keyframes and transcripts from YouTube videos to assess
emotional appeal, storytelling effectiveness, and product highlights in advertisements. The Finance
Agent (GPT-4o + Bloomberg and Yahoo Finance) processes financial performance indicators, rev-
enue trends, and advertising expenditures using Bloomberg and Yahoo Finance data pipelines. The
Market Agent (GPT-4o + Statista and Google Trends via Pytrends) analyzes consumer behavior
trends, competitor benchmarking, and market demands using Statista and Google Trends APIs. The
PAG system generates multilingual, persona-driven advertisements using a Simulated Humanis-
tic Colony of Agents built on GPT-4o, simulating diverse consumer segments, including but not
limited to: the Logical Strategist, who prioritizes scientific innovation; the Visionary Trailblazer,
who focuses on sustainability and eco-conscious branding; the Harmonious Connector, who values
emotional connections and community impact; the Resilient Optimist, who responds to confidence-
building and problem-solving messaging; and the Organized Architect, who prefers efficiency and
structured communication. The system incorporates an Advanced Curator Agent (GPT-4o) that cre-
ates multilingual advertisements, ensuring cultural and linguistic alignment, and a Social Media
Agent (GPT-4o) that optimizes content for platform-specific engagement, refining advertisements
for Twitter, Instagram, and Facebook. The CHPAS system incorporates product differentiation
from various manufacturers, increases affinity scoring for persona-product matching, implements
cannibalization prevention mechanisms, and optimizes unique selling proposition (USP) empha-
sis for competing products, such as tailoring detergent advertisements differently for specific user
segments. Real-world testing encompasses data from over 50 fast-moving consumer goods (FMCG)
companies across 50+ product categories, with 2,000+ products spanning household care, cosmetics,
and pharmaceuticals. The system simulates diverse personas across occupational groups, emotional
profiles, language preferences (English, Spanish, Asian, European, Middle Eastern), and socioeco-
nomic classes. The evaluation framework employs comprehensive metrics, including clickability
rate tracking across initial, personalized, and hyper-personalized advertisements. Quality assess-
ment is conducted using the NVIDIA Nemotron-4-340b-reward model, LLM-as-Judge evaluation
with GPT-4o, and human evaluator benchmarking, analyzing five key dimensions: helpfulness (ac-
tionable information), correctness (factual accuracy), coherence (logical flow), complexity (level
of detail), and verbosity (conciseness). To enhance computational efficiency in AI-driven adver-
tisement generation, we adopt a structured pipeline where MAAMS is executed first to generate
and store market intelligence before subsequent PAG and CHPAS stages. This decoupled approach
minimizes redundant processing, reduces latency, and optimizes resource utilization by avoiding re-
peated execution of MAAMS during each ad iteration. Instead, MAAMS outputs serve as a cached
knowledge base, allowing PAG and CHPAS to efficiently generate persona-specific and competi-
tively optimized advertisements. This separation ensures scalable, high-performance ad generation
while maintaining adaptability to evolving market conditions.
A.9 R ELATED WORK
Advancements in artificial intelligence have significantly transformed personalized advertising, en-
abling the creation of highly targeted and engaging campaigns. However, advertising in specialized
domains, such as chemical products, presents unique challenges that remain underexplored. This
section reviews recent contributions to personalized advertising, multimodal data analysis, and AI-
driven consumer persona modeling, highlighting their relevance to our work.
A.9.1 P ERSONALIZED ADVERTISING
Personalized advertising has evolved from traditional demographic-based targeting to AI-driven
methods that enable real-time personalization. Recent work by Awad et al. (2023) introduced the
adSformers framework, which utilizes transformer-based architectures to dynamically model short-
term user behaviors and optimize ad recommendations. Similarly, Wei et al. (2024) proposed a
unified multimodal personalization system that integrates text, image, and interaction data for gen-
erative recommendations. While these approaches excel in consumer goods markets, their applica-
bility to technical domains—such as chemical products, where regulatory compliance and technical
accuracy are critical—has not been thoroughly investigated. Advertisements for chemical products
must effectively communicate complex technical details, including safety protocols and environ-
27

Published as a conference paper at ICLR 2025
mental impact, while maintaining a connection with consumers. This work addresses this gap by
introducing a framework that combines technical accuracy with personalized, engaging messaging.
A.9.2 M ULTIMODAL DATA ANALYSIS FOR MARKETING
The integration of multimodal data has enabled more comprehensive analyses of consumer behavior.
Niimi (2024) demonstrated the effectiveness of combining textual and demographic data for predict-
ing customer ratings, while Li et al. (2024) highlighted the importance of fusing visual and textual
modalities for sentiment analysis. Despite these advances, existing methods often treat modalities
independently, missing opportunities to derive holistic insights. For example, while text data can
reveal consumer sentiment and image data can assess visual appeal, integrating these insights with
video data for emotional engagement and financial data for market trends provides a more complete
picture of advertising effectiveness. This framework addresses this limitation by integrating text,
images, and video data alongside financial and market insights to comprehensively evaluate adver-
tising effectiveness. This multimodal approach ensures that advertisements are not only visually
appealing and impactful but also aligned with market dynamics and consumer preferences.
A.9.3 AI-D RIVEN CONSUMER PERSONA MODELING
Persona modeling is essential for effective personalized advertising. Salminen et al. (2024) proposed
a framework for generating personas using qualitative user data and AI, while Zhang et al. (2024) ex-
plored role-playing language models to simulate diverse personas. Building upon these approaches,
this work utilizes a simulated humanistic colony of agents representing distinct consumer personas
with well-defined characteristics, enabling the creation of highly targeted advertisements that align
with specific consumer preferences and decision-making patterns. By utilizing distinct personas
such as the Logical Strategist, Visionary Trailblazer, and Harmonious Connector, this framework
captures diverse consumer preferences and characteristics, allowing for the generation of tailored
advertisements that align with specific consumer segments.
A.9.4 C HALLENGES IN SPECIALIZED DOMAINS
Specialized markets, such as chemical products, require precise communication of technical details
and adherence to stringent regulatory requirements, including compliance with GHS and REACH
standards (Hellier & Edworthy, 2012). Prior studies have primarily focused on packaging and label-
ing to address these challenges but often overlook the emotional and cultural dimensions of adver-
tising. For example, while hazard symbols and usage instructions are critical for regulatory compli-
ance, they must be complemented by compelling messaging to effectively engage consumers. This
work bridges this gap by integrating regulatory compliance with personalized, culturally relevant
messaging, ensuring that advertisements fulfill both technical and consumer-centric objectives. This
approach enhances consumer trust and engagement while maintaining adherence to industry stan-
dards and regulations. In summary, while existing research has made significant strides in person-
alized advertising, multimodal analysis, and persona modeling, several gaps remain. The proposed
framework addresses these challenges by combining persona modeling, multimodal analysis, and
domain-specific considerations, advancing the state of personalized advertising in technical domains
such as chemical products. By integrating technical accuracy, consumer engagement, and regula-
tory compliance, this work provides a comprehensive solution for creating effective and engaging
advertisements in specialized markets.
A.10 S YNTHETIC EXPERIMENTATION FOR SCALABLE ADPERSONALIZATION : A
DATA-DRIVEN APPROACH TO COMPETITIVE ADVERTISING
The proliferation of digital platforms has heightened the demand for hyper-personalized advertis-
ing, especially when promoting multiple, same-category products from different companies to the
same user. This presents a challenge: how to balance personalization, avoid brand or product can-
nibalization, and effectively highlight each product’s unique selling points (USPs). To address this,
we leverage synthetic experiments—a data-driven, scalable, and privacy-compliant approach—to
systematically test and optimize advertising strategies. These experiments offer a structured frame-
work for simulating real-world complexities while maintaining control over critical variables. Our
synthetic experimentation framework supports competitive advertising by enabling companies to
test diverse ad configurations, customer responses, and market scenarios without the costs and risks
inherent in large-scale, real-world testing. This approach provides four primary advantages: (a)
28

Published as a conference paper at ICLR 2025
Controlled Environment for Systematic Testing: Synthetic experiments create a controlled environ-
ment where diverse market conditions, consumer personas, and product categories can be accurately
simulated. AI-powered optimization allows for dynamic evaluation of different ad strategies, en-
suring scalability and adaptability. Unlike real-world campaigns, where external noise (competitor
actions, seasonality, or economic fluctuations) complicates causal attribution, synthetic testing iso-
lates variables, allowing for precise measurement of their impact on ad performance. By modeling
interactions between competing ads, companies can optimize their delivery strategies, mitigating
negative consequences such as audience fatigue or brand dilution. (b) Cost-Effective and Scalable
Competitive Testing: Testing competitive advertising strategies in the real world is prohibitively
expensive, often requiring extensive A/B testing across multiple market segments. Synthetic ex-
periments eliminate this constraint by simulating and evaluating different advertising permutations
at scale. Companies can systematically assess how various combinations of competing products
impact consumer decision-making without incurring massive testing costs or exposing campaigns
to real-world risks. This is particularly valuable in industries where strategically nuanced product
differentiation (e.g., pricing, feature sets, or branding) is critical for market success. (c) Privacy
Compliance and Ethical Considerations: With increasing global data regulations such as GDPR and
CCPA, real-world consumer data collection poses significant compliance risks. Synthetic data gen-
eration enables rigorous testing of advertising models while preserving privacy and security. By
eliminating the need for direct reliance on actual consumer data, synthetic experiments allow brands
to ethically refine targeting strategies, ensuring compliance with data protection laws. Additionally,
they mitigate potential biases and ethical concerns associated with real-world consumer profiling,
fostering more responsible AI-driven advertising. (d) Accelerated AI Model Development and Edge
Case Testing: The development of AI-powered advertising systems demands rapid iteration, exten-
sive testing, and robust validation across diverse scenarios. Synthetic experiments generate thou-
sands of test cases, including rare but impactful edge cases—such as sudden market disruptions,
shifts in consumer sentiment, or emerging trends—that would take years to observe in real-world
data. By accelerating testing cycles, companies can refine AI-driven ad personalization models,
ensuring their robustness before real-world deployment. This significantly enhances adaptability,
allowing brands to respond proactively to evolving market conditions. Our framework ensures that
synthetic experiments preserve key statistical properties and behavioral patterns observed in real ad-
vertising data, while also enabling controlled manipulation of crucial variables. This methodology
provides a strong validation foundation prior to real-world deployment, where mistakes could be
costly and irreversible. Ultimately, these synthetic experiments serve as a proof of concept, demon-
strating the framework’s capability to generate personalized, competitive ad campaigns tailored to
specific customer segments. By bridging the gap between controlled testing and real-world adapta-
tion, synthetic experimentation paves the way for more effective, data-driven advertising strategies
that optimize engagement, relevance, and return on investment.
A.10.1 F RAMEWORK ARCHITECTURE
The hyper-personalized competitive ad optimization framework is designed as a modular and scal-
able system that integrates multiple components to tackle the challenge of advertising multiple same-
category products from competing companies. This framework is built on synthetic experimentation,
incorporating synthetic personas, AI-driven ad generation, and market-aware optimization to create
and refine competitive ad strategies. The system consists of seven key components, each playing
a crucial role in optimizing ad effectiveness. The first component, Market Research Data, pro-
vides foundational demographic insights (including information on income range, price sensitivity,
and tech-savviness across different age groups) to assess how different customer segments respond
to advertising strategies, product category attributes (outlining consumer purchasing behavior, loy-
alty trends, and the importance of eco-friendly considerations for various products), and competitor
intelligence (offering insights into brand perception, market share distribution, and sustainability
positioning for leading industry players). The second component, Persona Profiling, defines syn-
thetic customer personas based on a combination of demographic attributes, interests, values, and
purchasing behaviors, enabling targeted ad personalization. This module categorizes consumers
into distinct lifestyle segments (e.g., eco-conscious, luxury seeker, budget-conscious individuals),
ensuring that advertisements are tailored to resonate with their unique preferences. Furthermore,
an AI-driven persona affinity scoring system determines how well a specific product aligns with
a given persona by analyzing factors such as interest alignment, value compatibility, and income-
based price sensitivity. The third component, Product Analysis, systematically evaluates product
29

Published as a conference paper at ICLR 2025
features, pricing strategies, and brand positioning to inform competitive advertising strategies. By
assessing price sensitivity, consumer loyalty, and brand positioning, this module enables companies
to refine their messaging to highlight the most compelling attributes of their offerings. Additionally,
this component examines feature uniqueness by comparing a product’s specifications against those
of competing products to identify key differentiators. The fourth component, Competitive Analysis,
focuses on identifying unique selling points, price positioning, and brand strength relative to com-
petitors. This module employs market positioning assessment techniques such as the Herfindahl-
Hirschman Index (HHI) to determine market concentration levels, competitive advantage scoring
to evaluate a brand’s relative strength w.r.t competitors, and category growth simulations to predict
long-term trends in consumer demand. The fifth component, AI-Powered Ad Generation, is respon-
sible for leveraging large language models to generate product advertisements at two distinct levels.
The first level involves the creation of base advertisements, which are generic product-focused ad
copies highlighting fundamental features and benefits. The second level introduces an optimized
ad generation process, where base ads are refined and personalized based on competitive intelli-
gence and persona affinity analysis. This ensures that advertisements are not only feature-driven
but also tailored to the interests, values, and pain points of specific consumer segments. The sixth
component, AI-Based Ad Evaluation, is designed to assess the performance of generated ads using
advanced AI-driven scoring mechanisms. This evaluation is conducted through two primary meth-
ods. The first method utilizes large language models acting as evaluators to score advertisements
based on key criteria such as relevance, persuasiveness, emotional impact, clarity, and call-to-action
effectiveness. The second method incorporates an AI-powered evaluation model that measures ad-
ditional attributes such as helpfulness, correctness, coherence, complexity, and verbosity. This dual
evaluation system provides a comprehensive assessment of how well an advertisement performs in
engaging consumers and driving conversions. The seventh and final component, Simulation and It-
erative Ad Optimization, is dedicated to running synthetic market simulations that assess and refine
ad strategies be
A.11 O PEN-DOMAIN QUESTION ANSWERING (ODQA)
This section presents an advanced Optimized Retrieval-Augmented Generation (RAG) system Lewis
et al. (2020); Gupta et al. (2024); Gao et al. (2023); Han et al. (2025), designed to process and utilize
multimodal agglomerated knowledge from the Multimodal Agentic Advertisement Market Survey
(MAAMS) system for question-answering (QA) tasks. The knowledge synthesized by MAAMS
comprises market intelligence from diverse modalities, including text, images, video, financial data,
and market data, which is consolidated into a document-specific text output format for each product.
While the system primarily processes textual knowledge extracted from documents, it enables se-
mantic search and contextually accurate response generation for QA tasks. A QA system built on this
agglomerated knowledge base acts as a crucial interface between customers and technical product
information, allowing users to receive precise and relevant answers without navigating lengthy doc-
umentation. The system excels in four key areas: (a) Providing direct access to product information;
(b) Offering comparative product insights; (c) Translating technical specifications into user-friendly
language; (d) Delivering specific usage guidance. By extracting and presenting relevant information
in an accessible format, the QA system enhances the customer experience, making technical prod-
uct information more actionable and user-friendly. The system begins by processing documents
using an automated text extraction method, ensuring the structured retrieval of textual content and
metadata. The extracted text is segmented into manageable, semantically coherent chunks with
overlapping regions to maintain contextual continuity. To improve context awarenesfore they are
deployed in real-world campaigns. This simulation process leverages AI-driven iteration cycles to
optimize ad performance by testing various combinations of ad messaging, pricing strategies, and
competitive positioning. The system continuously tracks performance metrics, allowing advertisers
to iteratively refine their campaigns to maximize impact. By integrating synthetic personas, AI-
driven insights, and competitive benchmarking, this framework provides an automated, data-driven,
and continuously optimized approach to advertising. It enables product manufacturers to develop
highly personalized and competitive ads that drive higher engagement, conversion rates, and stronger
market positioning. The framework ensures privacy compliance by relying on synthetic data rather
than real consumer information, making it an ethical and scalable solution for modern advertising
challenges. The combination of AI-generated ad content, competitive market analysis, and real-
time performance tracking enables companies to refine their messaging dynamically, ensuring that
advertisements remain relevant in rapidly evolving market conditions. By systematically simulating
30

Published as a conference paper at ICLR 2025
consumer responses and testing ad variations, this approach minimizes the risks associated with tra-
ditional A/B testing and allows companies to experiment with innovative marketing strategies in a
controlled environment. This comprehensive, modular, and scalable architecture ensures that busi-
nesses can continuously refine their advertising strategies, optimize their market positioning, and
maximize the return on investment from their marketing efforts.
A.11.1 M ETHODOLOGY
The hyper-personalized competitive ad optimization framework operates as an offline, synthetic
experimental setup, where each stage follows a structured pipeline to simulate and analyze ad ef-
fectiveness in a controlled environment. The process begins with persona profiling, where synthetic
user personas are generated based on demographic attributes (age, income, lifestyle), psychographic
traits (interests, values), and behavioral tendencies (purchase frequency, price sensitivity, brand loy-
alty). These personas serve as structured input representations that guide subsequent stages. Next,
product analysis evaluates each product’s characteristics, including functional features (e.g., “or-
ganic ingredients”, “long-lasting fragrance”), price positioning (premium, mid-range, budget), and
brand perception (e.g., sustainability, innovation, quality), yielding a product-persona affinity score
that determines how well a product aligns with different personas. The competitive analysis stage
then integrates synthetic market data—including competing products, price trends, brand position-
ing, and market share—to generate a competitive positioning report highlighting differentiation op-
portunities and pricing strategies tailored to each persona. This analysis feeds into ad generation,
where two types of synthetic ads are created: base ads, which focus on general product attributes,
and optimized ads, which incorporate persona-specific customization, addressing user pain points
(e.g., concerns about harsh chemicals), reinforcing values (e.g., eco-friendliness), and highlighting
competitive advantages (e.g., larger volume or better pricing). These synthetic ads undergo AI-
powered evaluation using GPT-4 Omni and NVIDIA Nemotron-4-340B, which assess each ad based
on relevance, persuasiveness, emotional impact, clarity, and call-to-action effectiveness, producing
a scoring matrix ranking ads based on their effectiveness for different personas. Finally, the ad rank-
ing and deployment phase prioritizes ads based on their evaluation scores, persona-product affinity,
and competitive positioning, ensuring that each persona is matched with the most compelling adver-
tisement. As this is a synthetic experiment rather than a real-time system, ad performance feedback
is not derived from live deployment but is instead simulated using synthetic persona responses and
AI-based evaluation metrics. The system iterates offline, refining persona profiles, product analysis,
and ad optimization in a batch-based process, ensuring that insights from synthetic experiments can
inform future strategies for real-world deployment.
A.11.2 E XAMPLE SCENARIO
To illustrate the framework’s functionality, consider a scenario involving three shampoos from com-
peting companies, each targeting distinct customer personas: PureGlow Shampoo (premium, eco-
friendly), EconoFresh Shampoo (budget-friendly), and LuxeSilk Shampoo (luxury, high-quality).
The framework systematically applies synthetic experimentation, competitive analysis, and AI-
driven ad optimization to create targeted messaging that maximizes engagement for each persona
while ensuring competitive differentiation.
•Persona A: Eco-Conscious Professional
– Demographic & Lifestyle Insights : Values sustainability, prefers plant-based prod-
ucts, and seeks a shampoo free from harsh chemicals.
– Competitive Positioning : Compared to standard shampoos, PureGlow Shampoo
offers an eco-friendly, sulfate-free formula that outperforms competitors by us-
ing100% biodegradable packaging and ethically sourced ingredients . Unlike
EconoFresh , which contains synthetic cleansers, and LuxeSilk , which focuses on pre-
mium aesthetics, PureGlow directly aligns with this persona’s sustainability values.
– Generated Ad:
”Choose a shampoo that cares for your hair—and the planet. PureGlow
Shampoo is crafted with organic botanicals and zero synthetic additives, de-
livering a natural shine while protecting the environment. Make the switch
today!”
31

Published as a conference paper at ICLR 2025
•Persona B: Budget-Conscious Student
– Demographic & Lifestyle Insights : Prefers affordable products that offer long-
lasting use and strong cleansing power.
– Competitive Positioning : EconoFresh Shampoo delivers cost-effective, high-
foaming cleansing that lasts 30% longer than comparable brands. While Pure-
Glow is focused on premium eco-conscious buyers and LuxeSilk on luxury aesthetics,
EconoFresh prioritizes affordability and value for money .
– Generated Ad:
”Stretch your budget further without compromising quality! EconoFresh
Shampoo provides a deep clean with a refreshing scent, designed for daily
use at an unbeatable price. Get more washes for less!”
•Persona C: Luxury-Seeking Professional
– Demographic & Lifestyle Insights : Prefers high-quality formulations enriched with
premium ingredients for a refined haircare experience.
– Competitive Positioning : LuxeSilk Shampoo is infused with silk proteins and
botanical extracts , offering superior hydration and shine compared to competi-
tors. Unlike PureGlow , which prioritizes natural ingredients, or EconoFresh , which
focuses on affordability, LuxeSilk is designed for consumers who seek an indulgent,
high-performance formula .
– Generated Ad:
”Unleash the power of luxury with LuxeSilk Shampoo. Infused with silk pro-
teins, this rich formula deeply nourishes each strand, leaving your hair irre-
sistibly soft, radiant, and elegantly fragrant. Elevate your routine today!”
This scenario demonstrates how the framework dynamically optimizes ad personalization using
persona-product affinity analysis, competitive positioning, and AI-driven ad evaluation . By integrat-
ing these insights, the system maximizes ad relevance and effectiveness , ensuring each product is
strategically positioned against its competitors.
A.11.3 R ESULTS
We evaluate the AI-driven hyper-personalized competitive ad optimization framework for creating
highly personalized and effective advertisements and present the empirical results in Figures 17–21.
Figures 18–19 provide a high-level comparison of base ads versus optimized ads across multiple
personas, evaluated using the Nvidia Nemotron-4-340b-reward model (e.g., in terms of attributes
likecoherence ,complexity ,correctness ,helpfulness , and verbosity ) and LLM-as-a-Judge Met-
rics (e.g., clarity ,call-to-action effectiveness ,emotional impact ,persuasiveness , and relevance )
to assess and score the quality of generated advertisements. The results demonstrate that optimized
ads consistently outperform base ads across all metrics, highlighting the effectiveness of hyper-
personalization and competitive optimization. In Figure 18, metrics such as Clarity (4.00 vs. 3.56),
Call-to-Action Effectiveness (3.54 vs. 2.50), and Emotional Impact (3.74 vs. 2.58) show no-
table improvements for optimized ads (see; Figure 18) . For instance, Emotional Impact scores
are higher, indicating that optimized ads resonate more deeply with the personas’ values and pain
points. Persuasiveness (3.56 vs. 2.40) and Relevance (3.92 vs. 2.70) also see improvements,
demonstrating that optimized ads are more compelling and better tailored to the target audience.
In Figure 19, the Coherence metric (3.65 vs. 3.53) shows significant improvement for optimized
ads, indicating enhanced clarity and logical flow. Similarly, other Nvidia evaluation criteria such as
Helpfulness (2.94 vs. 2.30) and Correctness (2.92 vs. 2.29) also reflect higher scores for optimized
ads, suggesting better alignment with personas’ needs and expectations (refer; Figure 19). These
results underscore the value of hyper-personalization and competitive optimization in ad creation,
as optimized ads consistently deliver superior performance across both the Nvidia Reward model
and LLM-as-a-Judge evaluation frameworks. Figures 20-21 analyze ad performance across differ-
ent products, while earlier Figures 18-19 focused on ad performance across different personas. The
core message remains consistent: optimized ads consistently outperform base ads across all metrics,
delivering clearer, more compelling, and more relevant messaging, which drives better engagement
and performance. This consistency across both personas and products demonstrates the robustness
and scalability of the hyper-personalization and competitive optimization framework. The product-
level analysis provides additional insights into how different products benefit from optimization,
32

Published as a conference paper at ICLR 2025
further validating the effectiveness of the approach. Figure 17 compares the performance of opti-
mized ads versus base ads across LLM Metrics and Nvidia Metrics. In the LLM Metrics section,
Relevance (24.8%) and Emotional Impact (24.6%) show the highest improvements, indicating that
optimized ads better align with audience needs and evoke stronger emotional responses. Call to Ac-
tion Effectiveness improves by 16.8%, making optimized ads more compelling, while Clarity has
the smallest improvement at 5.9%, suggesting base ads were already clear (see; Figure 17). In the
Nvidia Metrics section, Correctness improves the most at 17.6%, reflecting improved factual accu-
racy, followed by Helpfulness (8.5%) and Verbosity (7.4%). Complexity andCoherence show the
lowest improvements at 6.9% and 1.8%, respectively, indicating base ads were already coherent and
appropriately complex (see; Figure 17). Overall, optimized ads perform better, with the most sig-
nificant gains in relevance, emotional impact, and correctness, while clarity and coherence remain
strong areas with limited room for improvement. This highlights the effectiveness of optimization
in enhancing audience engagement and action. The hyper-personalized competitive ad optimization
framework offers several critical insights for companies: Personalization is key : Tailoring ads to
each persona’s unique characteristics ensures maximum relevance and engagement. Leveraging
competitive advantages : Highlighting unique selling points (e.g., sustainability for eco-conscious
personas or affordability for budget-conscious ones) helps products stand out. AI-powered ad eval-
uation : Using AI-driven metrics (relevance, persuasiveness, emotional impact) ensures continuous
improvement. Strategic deployment : Prioritizing the most relevant ads for each persona optimizes
engagement. Competitive differentiation : Companies can emphasize product advantages to max-
imize positioning in the market. By following these principles, companies can effectively advertise
multiple same-category products, minimizing cannibalization and maximizing ad performance.
33

Published as a conference paper at ICLR 2025
Clarity
Call to Action EffectivenessEmotional Impact PersuasivenessRelevance051015202530Improvement (%)
5.9%21.3%24.8%
16.8%24.6%LLM Metrics
Coherence Complexity Correctness HelpfulnessVerbosity05101520Improvement (%)
1.8%6.9%17.6% 17.4%
8.5%Nvidia MetricsAverage Improvements: Optimized vs Base Ads
Figure 17: Average percentage improvements of hyper-personalized ads over base ads across all
products and personas. The figure presents the evaluation results from LLM-as-a-Judge (GPT-4o)
and reward model (Nemotron-4-340b-reward), showing improvements in key performance indica-
tors such as clarity ,call-to-action effectiveness ,emotional impact ,persuasiveness , and relevance
(LLM metrics) as well as coherence ,complexity ,correctness ,helpfulness , and verbosity (reward
model metrics). Hyper-personalized ads demonstrate consistent improvements across all dimen-
sions.
34

Published as a conference paper at ICLR 2025
Persona_1 Persona_2 Persona_3012345Score4.00
3.563.664.00 4.003.88Clarity
Ad Type
base
optimized
Persona_1 Persona_2 Persona_3012345Score3.24
2.502.803.543.64
3.18Call to Action Effectiveness
Ad Type
base
optimized
Persona_1 Persona_2 Persona_3012345Score3.34
2.58 2.643.74 3.68
3.26Emotional Impact
Ad Type
base
optimized
Persona_1 Persona_2 Persona_3012345Score3.48
2.402.923.56 3.56
3.16Persuasiveness
Ad Type
base
optimized
Persona_1 Persona_2 Persona_3012345Score3.50
2.702.903.92 3.92
3.50Relevance
Ad Type
base
optimizedLLM Metrics by Persona: Base vs Optimized
Figure 18: Comparison of base vs. hyper-personalized ads using LLM-as-a-Judge evaluation (GPT-
4o) across a subset of personas. This figure presents the performance comparison between base and
hyper-personalized ads assessed by GPT-4o, measuring clarity, call-to-action effectiveness, emo-
tional impact, persuasiveness, and relevance. Results are shown for a subset of personas to illustrate
the impact of hyper-personalization across different user segments.
35

Published as a conference paper at ICLR 2025
Persona_1 Persona_2 Persona_3012345Score3.58 3.53 3.58 3.60 3.64 3.65Coherence
Ad Type
base
optimized
Persona_1 Persona_2 Persona_30.00.51.01.52.02.5Score1.701.611.671.811.76 1.75Complexity
Ad Type
base
optimized
Persona_1 Persona_2 Persona_30.00.51.01.52.02.53.03.54.0Score2.63
2.292.442.92 2.922.82Correctness
Ad Type
base
optimized
Persona_1 Persona_2 Persona_30.00.51.01.52.02.53.03.54.0Score2.61
2.302.432.94 2.882.80Helpfulness
Ad Type
base
optimized
Persona_1 Persona_2 Persona_30.00.51.01.52.0Score1.481.441.511.641.57 1.58Verbosity
Ad Type
base
optimizedNVIDIA Metrics by Persona: Base vs Optimized
Figure 19: Comparison of base vs. hyper-personalized ads using reward model evaluation
(Nemotron-4-340b-reward) across a subset of personas. Evaluated using the Nemotron-4-340b-
reward model, this figure compares coherence, helpfulness, and correctness across personas. The
results represent a subset of personas to highlight how AI-driven ad personalization improves en-
gagement and message effectiveness.
36

Published as a conference paper at ICLR 2025
EcoClean Detergent LuxBrands Detergent ValueMax Detergent TechCare DetergentNaturePure Detergent012345Score3.77
3.334.00
3.673.93 4.00 4.00 3.93 3.93 3.93Clarity
Ad Type
base
optimized
EcoClean Detergent LuxBrands Detergent ValueMax Detergent TechCare DetergentNaturePure Detergent012345Score2.83
2.333.002.833.233.47 3.40 3.47 3.403.53Call to Action Effectiveness
Ad Type
base
optimized
EcoClean Detergent LuxBrands Detergent ValueMax Detergent TechCare DetergentNaturePure Detergent012345Score3.13
2.572.90
2.573.103.703.53 3.503.603.47Emotional Impact
Ad Type
base
optimized
EcoClean Detergent LuxBrands Detergent ValueMax Detergent TechCare DetergentNaturePure Detergent012345Score2.93
2.603.27
2.603.27 3.333.60
3.33 3.333.53Persuasiveness
Ad Type
base
optimized
EcoClean Detergent LuxBrands Detergent ValueMax Detergent TechCare DetergentNaturePure Detergent012345Score3.17
2.173.50
2.833.503.833.70 3.703.83 3.83Relevance
Ad Type
base
optimizedLLM Metrics by Product: Base vs Optimized
Figure 20: Comparison of base vs. hyper-personalized ads using LLM-as-a-Judge evaluation
(GPT-4o) across a subset of products. This figure evaluates base and hyper-personalized ads
across selected product categories using GPT-4o. Improvements in clarity, call-to-action effective-
ness, emotional impact, persuasiveness, and relevance reinforce the benefits of AI-driven hyper-
personalization for product-specific advertising strategies. The results focus on a subset of products
to provide a representative analysis.
37

Published as a conference paper at ICLR 2025
EcoClean Detergent LuxBrands Detergent ValueMax Detergent TechCare DetergentNaturePure Detergent012345Score3.593.493.59 3.59 3.58 3.59 3.66 3.66 3.673.57Coherence
Ad Type
base
optimized
EcoClean Detergent LuxBrands Detergent ValueMax Detergent TechCare DetergentNaturePure Detergent0.00.51.01.52.02.5Score1.671.601.731.62 1.661.73 1.761.81 1.78 1.78Complexity
Ad Type
base
optimized
EcoClean Detergent LuxBrands Detergent ValueMax Detergent TechCare DetergentNaturePure Detergent0.00.51.01.52.02.53.03.54.0Score2.62
2.162.62
2.352.512.732.93 2.98 2.972.80Correctness
Ad Type
base
optimized
EcoClean Detergent LuxBrands Detergent ValueMax Detergent TechCare DetergentNaturePure Detergent0.00.51.01.52.02.53.03.54.0Score2.57
2.172.62
2.382.512.752.92 2.93 2.962.82Helpfulness
Ad Type
base
optimized
EcoClean Detergent LuxBrands Detergent ValueMax Detergent TechCare DetergentNaturePure Detergent0.00.51.01.52.0Score1.481.411.521.48 1.491.601.561.61 1.60 1.62Verbosity
Ad Type
base
optimizedNVIDIA Metrics by Product: Base vs Optimized
Figure 21: Comparison of base vs. hyper-personalized ads using reward model evaluation
(Nemotron-4-340b-reward) across a subset of products. This figure presents performance differ-
ences between base and hyper-personalized ads evaluated with the Nemotron-4-340b-reward model.
Improvements in coherence, helpfulness, and correctness indicate the effectiveness of AI-driven
hyper-personalized ad strategies tailored to selected product categories. Results are presented for a
subset of products to illustrate the general trends in ad performance.
38

Published as a conference paper at ICLR 2025
Table 5: Comparative Analysis of EcoClean Detergent Advertisements: Content Personalization
Across Three Target Demographics
JSON Field Eco-Conscious Luxury (Persona 1) Active Lifestyle (Persona 2) Premium Innovation (Persona 3)
headline Transform Your Laundry with
EcoClean Detergent - The Luxury
Choice for the Eco-ConsciousAffordable EcoClean Detergent:
Perfect for Your Active LifestyleElevate Your Laundry Routine
with EcoClean Detergent -
Luxury Meets Sustainability
subheadline Discover an efficient, eco-friendly
cleaning solution that saves money
and protects the planet.Clean clothes, eco-friendly
choices, all within your budget!Innovative cleaning that
safeguards your wardrobe and
the environment.
mainBody Introducing EcoClean Detergent -
where luxury meets sustainability.
Designed for discerning
individuals like you, EcoClean
offers unparalleled cleaning
performance without
compromising on your values. Our
eco-friendly formula not only
penetrates deep into your favorite
fabrics, ensuring they stay vibrant
and fresh, but it also comes in
minimal, recyclable packaging to
reduce waste. Enjoy the peace of
mind that comes with knowing
you’re making a smart choice for
both your wardrobe and our
environment. Priced at just $20.00,
EcoClean embodies efficiency and
luxury, making it the perfect fit for
your eco-conscious lifestyle and
commitment to savings.As someone who values health and
sustainability, you deserve a
detergent that aligns with your
lifestyle. EcoClean Detergent
offers premium cleaning power at
just $20.00, making it a smart
investment for your
budget-conscious routine. Our
eco-friendly formula not only cares
for your clothes but also protects
the planet, ensuring you can focus
on your fitness journey without
compromising on quality or
performance. Experience the
difference of natural ingredients
that keep your activewear fresh and
ready for your next workout while
being gentle on both your family’s
health and the environment.Meet EcoClean Detergent, the
premium choice for conscious
consumers who value both
luxury and innovation. Our
advanced formula not only
delivers unbeatable cleaning
power but also provides
long-lasting protection for your
favorite fabrics, ensuring they
look new wash after wash. You
deserve a detergent that not only
meets your high standards but
also aligns with your
commitment to sustainability.
With EcoClean, every wash is a
step towards a cleaner planet,
without compromising on
quality. Experience the perfect
blend of performance and
environmental
responsibility—because you
shouldn’t have to choose
between luxury and
sustainability.
callToAction Join the movement towards a
cleaner, greener future. Order
EcoClean today and elevate your
laundry experience!Make the switch to EcoClean
today and enjoy a cleaner, greener
laundry experience that fits your
budget!Invest in Quality for Your Home
- Shop Now for Only $20.00!
targetedBenefits • Exceptional cleaning power that
outperforms traditional
detergents, giving you more
value for your money
• Eco-friendly ingredients and
minimal packaging that align
with your values and concern for
the environment
• Long-lasting protection for your
clothes, helping you save on
replacements while keeping
them looking new• Exceptional cleaning power that
beats traditional detergents while
being easy on your wallet
• Eco-friendly ingredients that
prioritize your health and the
well-being of the planet
• Long-lasting freshness for your
workout gear, so you can feel
confident and ready for anything• Unmatched cleaning
performance that rivals even
the strongest brands
• Long-lasting fabric protection
that saves you money over
time
• Eco-friendly ingredients that
resonate with your values of
innovation and luxury
39

Published as a conference paper at ICLR 2025
A.11.4 V ISUALIZATION ANALYSIS OF SYNTHETIC MARKETING DATA
Table 5 illustrates personalized advertisements for EcoClean Detergent across three synthetic per-
sonas: Eco-Conscious Luxury, Active Lifestyle, and Premium Innovation consumers. The table
breaks down messaging components (headlines, subheadlines, main body, calls-to-action, and tar-
geted benefits) demonstrating how the same product is marketed differently to each persona. While
Eco-Conscious Luxury messaging emphasizes sustainability and premium quality, Active Lifestyle
focuses on affordability and performance, and Premium Innovation stresses advanced technology
with environmental responsibility. This comparison showcases how synthetic persona generation
informs targeted advertising strategies. The suite of visualizations (Figures 22, 23, 24, 25, 26, 27,
28, 29, 30, and 31) provides comprehensive insights into the effectiveness of synthetic data gener-
ation for advertising analysis. Figure 22 reveals distinct pricing patterns across synthetic product
categories, while Figure 23 demonstrates the simulated market segment distribution, showing how
different products are positioned across market tiers. The correlation between synthetic products
and generated customer personas is examined in Figure 24, which highlights varying degrees of in-
terest alignment across different synthetic customer segments. The distribution of company values
is analyzed in Figure 25, providing insights into how different brand values can be positioned in
marketing strategies. Figures 26 and 27 together offer a detailed view of pricing strategies relative
to synthetic customer income levels, revealing potential market opportunities and gaps in the simu-
lated environment. The alignment between company values and customer interests is visualized in
Figure 28, while Figure 29 quantifies product-lifestyle fit across different synthetic customer seg-
ments. Figure 30 maps how effectively generated product features address synthetic customer pain
points, identifying areas for potential product development and marketing focus. Finally, Figure 31
provides a comprehensive view of simulated market segmentation strategies, combining price dis-
tribution, market segment distribution, and company-specific targeting approaches. Together, these
visualizations demonstrate the power of synthetic data in understanding market dynamics, customer
preferences, and advertising opportunities, offering valuable insights for real-world marketing strate-
gies without the need for extensive real-world data collection or testing.
detergenttoothpaste
Category1015202530354045Price ($)Price Distribution by Product Category
Figure 22: Price Distribution by Product Category. This visualization reveals the price variation
within detergent and toothpaste categories, helping identify pricing patterns and potential market
positioning opportunities. The box plots show median prices, quartiles, and outliers, enabling strate-
gic pricing decisions based on category-specific market dynamics.
40

Published as a conference paper at ICLR 2025
premium
60.0%
budget20.0%
mid-range20.0%Product Distribution by T arget Market
Figure 23: Product Distribution by Target Market. A pie chart showing the proportion of products
targeting different market segments (premium, budget, mid-range). This visualization helps identify
potential market gaps and assess the balance of product offerings across different consumer seg-
ments.
41

Published as a conference paper at ICLR 2025
Persona_1 Persona_2
PersonasEcoClean Detergent
LuxBrands Detergent
ValueMax Detergent
T echCare Detergent
NaturePure Detergent
EcoClean T oothpaste
LuxBrands T oothpaste
ValueMax T oothpaste
T echCare T oothpaste
NaturePure T oothpasteProducts0 1
1 1
0 2
1 0
1 1
1 0
0 1
1 0
0 1
1 1Product-Persona Interest Correlation
0.000.250.500.751.001.251.501.752.00
Figure 24: Product-Persona Interest Correlation Heatmap. This visualization maps the alignment
between product target interests and persona interests, highlighting potential marketing opportunities
and product-customer fit. Darker colors indicate stronger alignment between products and specific
customer personas.
sustainabilityefficiencyluxury quality
affordabilityreliabilityinnovationhealth
Values0.00.51.01.52.02.53.03.54.0CountCompany Values Distribution
Figure 25: Company Values Distribution. Bar chart showing the frequency of different company
values across the product portfolio. This visualization helps understand the overall brand positioning
and identify potential gaps in value propositions, guiding future marketing and branding strategies.
42

Published as a conference paper at ICLR 2025
EcoCleanLuxBrands NaturePureT echCare ValueMax
Company020406080100Price ($)Average Product Price by Company vs Average Persona Income
Avg Persona Income (scaled 1:1000): $104,138.87
Figure 26: Average Product Price by Company vs Average Persona Income. This comparison be-
tween company pricing strategies and customer income levels helps assess product affordability and
market positioning. The red dashed line represents scaled average persona income, providing con-
text for pricing decisions.
Budget Mid-range Premium
Price Segment020406080100120Average Price ($)
n=3n=3n=4Product Price Segments vs Persona Income Levels
Persona_1 Income (scaled 1:1000)
Persona_2 Income (scaled 1:1000)
Figure 27: Price Segments vs Persona Income Levels. This visualization segments products into
price tiers (Budget, Mid-range, Premium) and overlays persona income levels, revealing how well
current pricing aligns with customer purchasing power. Bar annotations show the number of prod-
ucts in each segment.
43

Published as a conference paper at ICLR 2025
luxury savings fitness health environment technology
T arget Interestsefficiency
affordability
luxury
quality
sustainability
innovation
health
reliabilityCompany Values1 3 1 1 1 1
0 0 1 1 1 1
1 1 0 0 2 0
1 1 0 0 2 0
0 2 0 1 2 3
1 1 1 0 1 0
0 0 0 0 2 2
0 0 1 1 1 1Company Values vs T arget Interests Alignment
0.00.51.01.52.02.53.0
Figure 28: Company Values vs Target Interests Alignment Heatmap. This visualization shows how
company values align with target customer interests, revealing opportunities for more effective mar-
keting messaging. The intensity of colors indicates the strength of alignment between values and
interests.
budget conscious busy professional
Lifestyle SegmentEcoClean LuxBrands ValueMax T echCare NaturePureCompany3.0 1.0
3.0 4.0
1.0 4.0
1.0 1.0
4.0 2.0Product-Lifestyle Matching Score
1.01.52.02.53.03.54.0
Figure 29: Product-Lifestyle Matching Score Heatmap. This visualization quantifies how well dif-
ferent companies’ products match with various lifestyle segments, considering factors like price
alignment, shared values, and interests. Higher scores (darker colors) indicate better product-
lifestyle fit.
44

Published as a conference paper at ICLR 2025
Stress from work-life balance Need for convenient solutionsWorried about value for moneyNeeds long-lasting products
Pain PointsPremium toothpaste formula
Long-lasting protection
Premium detergent formula
Eco-friendly ingredientsProduct Features0 0 0 0
0 0 0 1
0 0 0 0
0 0 0 0Product Features vs Persona Pain Points Alignment
0.00.20.40.60.81.0
Figure 30: Product Features vs Persona Pain Points Alignment Heatmap. This visualization maps
how well current product features address identified customer pain points. The heatmap helps iden-
tify gaps in product offerings and opportunities for product development to better meet customer
needs.
premiumbudget
mid-range
target_market1015202530354045pricePrice Distribution by T arget Market
premium
60.0%
budget20.0%mid-range20.0%T arget Market Distribution
budget mid-range premium
target_marketEcoClean
LuxBrands
NaturePure
T echCare
ValueMaxcompany0 0 2
0 0 2
0 0 2
0 2 0
2 0 0Company Focus by T arget Market
0.000.250.500.751.001.251.501.752.00
Figure 31: Market Segment Analysis Multi-plot. This comprehensive visualization combines
three views: (1) price distribution across target markets, (2) market segment distribution, and (3)
company-specific market targeting. Together, these plots provide insights into market segmentation
strategies and competitive positioning.
45