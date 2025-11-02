# Mubeen AI: A Specialized Arabic Language Model for Heritage Preservation and User Intent Understanding

**Authors**: Mohammed Aljafari, Ismail Alturki, Ahmed Mori, Yehya Kadumi

**Published**: 2025-10-27 12:29:27

**PDF URL**: [http://arxiv.org/pdf/2510.23271v1](http://arxiv.org/pdf/2510.23271v1)

## Abstract
Mubeen is a proprietary Arabic language model developed by MASARAT SA,
optimized for deep understanding of Arabic linguistics, Islamic studies, and
cultural heritage. Trained on an extensive collection of authentic Arabic
sources significantly expanded by digitizing historical manuscripts via a
proprietary Arabic OCR engine, the model incorporates seminal scholarly works
in linguistics, jurisprudence, hadith, and Quranic exegesis, alongside
thousands of academic theses and peer-reviewed research papers. Conditioned
through a deep linguistic engineering framework, Mubeen masters not just the
meaning but the eloquence of Arabic, enabling precise understanding across
classical texts, contemporary writing, and regional dialects with focus on
comprehending user intent and delivering accurate, contextually relevant
responses. Unlike other Arabic models relying on translated English data that
often fail in intent detection or retrieval-augmented generation (RAG), Mubeen
uses native Arabic sources to ensure cultural authenticity and accuracy. Its
core innovation is the Practical Closure Architecture, designed to solve the
"Utility Gap Crisis" where factually correct answers fail to resolve users'
core needs, forcing them into frustrating cycles of re-prompting. By
prioritizing clarity and decisive guidance, Mubeen transforms from an
information repository into a decisive guide, aligning with Saudi Vision 2030.
The model's architecture combines deep heritage specialization with
multi-disciplinary expert modules, enabling robust performance across both
cultural preservation and general knowledge domains.

## Full Text


<!-- PDF content starts -->

1

Mubeen AI: A Specialized Arabic Language Model for Heritage
Preservation and User Intent Understanding
Mohammed Aljafari1, Ismail Alturki1, Ahmed Mori1, Yehya Kadumi1,
1MASARAT, Saudi Arabia
Corresponding author: research@masarat.sa
September 04, 2025
1 Abstract
Mubeen is a proprietary Arabic language model developed by MASARAT SA, optimized for a deep understanding of Arabic
linguistics, Islamic studies, and cultural heritage. Trained on an extensive collection of authentic Arabic sources significantly
expanded by digitizing historical manuscripts via a proprietary Arabic OCR engine developed by Our Team, including seminal
scholarly works in linguistics, jurisprudence, hadith, and Quranic exegesis, alongside thousands of academic theses and peer-
reviewed research papers and conditioned through a deep linguistic engineering framework to master not just the meaning but
the eloquence of Arabic. This enables a deep and precise understanding of Arabic across all its levels from classical texts to
contemporary writing and regional dialects with a focus on comprehending user intent and delivering accurate, contextually
relevant responses. Unlike other Arabic models that rely on translated English data and often fail in intent detection or retrieval-
augmented generation (RAG), Mubeen uses native Arabic sources to ensure cultural authenticity and accuracy. Its core
innovation is the Practical Closure Architecture, designed to solve the ”Utility Gap Crisis,” where factually correct answers fail
to resolve the user’s core need, forcing them into frustrating cycles of re-prompting and clarification. By prioritizing clarity
and decisive guidance, Mubeen transforms from an information repository into a decisive guide, aligning with Saudi Vision
2030. The model’s architecture combines deep heritage specialization with multi-disciplinary expert modules, enabling robust
performance across both cultural preservation and general knowledge domains.
ﻰﻠﻋبّرﺪﻣ.ﻲﻓﺎﻘﺜﻟاثاﺮﺘﻟاوﺔﯿﻣﻼﺳﻹاتﺎﺳارﺪﻟاوﺔﯿﺑﺮﻌﻟاتﺎﯿﻧﺎﺴﻠﻟﻖﯿﻤﻋﻢﻬﻔﻟﻦﺴﺤﻣ،ﺔﯾدﻮﻌﺴﻟاتارﺎﺴﻣﺔﻛﺮﺷﺔﻄﺳاﻮﺑرّﻮُﻃﻲﻜﻠﻣﻲﺑﺮﻋيﻮﻐﻟجذﻮﻤﻧﻦﯿﺒﻣ 
ﺔﯿﺑﻮﺳﺎﺣﺔﯾؤركﺮﺤﻣماﺪﺨﺘﺳﺎﺑﺔﯿﺨﯾرﺎﺘﻟاتﺎﻃﻮﻄﺨﻤﻟاﺔﻨﻤﻗرﺮﺒﻋﺮﯿﺒﻛﻞﻜﺸﺑﺎﻬﻌﯿﺳﻮﺗﻢﺗﻲﺘﻟاوﺔﻗﻮﺛﻮﻤﻟاﺔﯿﺑﺮﻌﻟاردﺎﺼﻤﻟاﻦﻣﺔﻤﺨﺿﺔﻋﻮﻤﺠﻣ OCR ﻢﺗمﺪﻘﺘﻣ 
ﺔﯿﻓﺮﺼﻟاﺔﯿﻨﺒﻠﻟﻪﻧﺎﻘﺗإﺦﯿﺳﺮﺗﻰﻟإفﺪﻬﺗ،ةﺮﻜﺘﺒﻣ"ﺔﻘﯿﻤﻋﺔﯾﻮﻐﻟﺔﺳﺪﻨﻫ"ﺔﯿﺠﻬﻨﻤﻟﻊﻀﺧﻞﺑ،ﺐﺴﺤﻓتﺎﻧﺎﯿﺒﻟاﺔﻣﺎﺨﺿﻰﻠﻋﻪﺒﯾرﺪﺗﺮﺼﺘﻘﯾﻢﻟو،ﺎﻨﻘﯾﺮﻓﺔﻄﺳاﻮﺑهﺮﯾﻮﻄﺗ 
ﺔﯿﻜﯿﺳﻼﻜﻟاصﻮﺼﻨﻟاﻦﻣﺔﻔﻠﺘﺨﻤﻟاﺎﻬﺗﺎﯾﻮﺘﺴﻤﺑﺔﯿﺑﺮﻌﻟاﺔﻐﻠﻟﻖﯿﻗدﻢﻬﻓﻦﻣﻪﻨّﻜﻣﻖﻤﻌﻟااﺬﻫ.ﺔﻏﺎﯿﺼﻟاﻦﻓﻦﻣﻦّﻜﻤﺘﻟاﻰﻟإﻰﻨﻌﻤﻟاﻢﻬﻓدﺮﺠﻣﻦﻣﻪﻠﻘﻨﯾﺎﻤﻣ،ﺔﻐﻠﻟﺔﯿﻏﻼﺒﻟاو 
ﺔﯾﺰﯿﻠﺠﻧإتﺎﻧﺎﯿﺑﻰﻠﻋﺪﻤﺘﻌﺗﻲﺘﻟاىﺮﺧﻷاﺔﯿﺑﺮﻌﻟاجذﺎﻤﻨﻟافﻼﺨﺑ.ﺔﯿﻗﺎﯿﺳوﺔﻘﯿﻗدتﺎﺑﺎﺟإﻢﯾﺪﻘﺗومﺪﺨﺘﺴﻤﻟاﺔﯿﻧبﺎﻌﯿﺘﺳاﻰﻠﻋﺰﯿﻛﺮﺘﻟاﻊﻣﺔﯿﻠﺤﻤﻟاتﺎﺠﻬﻠﻟاوةﺮﺻﺎﻌﻤﻟاﻰﻟإ 
)عﺎﺟﺮﺘﺳﻻﺎﺑزﺰﻌﻤﻟاﻞﯿﺠﻟاوأﺔﯿﻨﻟاﻒﺸﻛﻲﻓﻞﺸﻔﺗﺎﻣًﺎﺒﻟﺎﻏوﺔﻤﺟﺮﺘﻣ RAG ( هرﺎﻜﺘﺑاﻦﻤﻜﯾ.ﺔﻗﺪﻟاوﺔﯿﻓﺎﻘﺜﻟاﺔﻟﺎﺻﻷانﺎﻤﻀﻟﺔﯿﻠﺻأﺔﯿﺑﺮﻋردﺎﺼﻣﻦﯿﺒﻣمﺪﺨﺘﺴﯾ، 
)"ﻲﻠﻤﻌﻟاقﻼﻏﻹاﺔﯿﻨﺑ"ﻲﻓﻲﺳﺎﺳﻷا Practical Closure Architecture ( ﺔﯿﺒﻠﺗﻲﻓًﺎﯿﻌﻗاوﺔﺤﯿﺤﺼﻟاتﺎﺑﺎﺟﻹاﻞﺸﻔﺗﺚﯿﺣ،"ﺔﻌﻔﻨﻤﻟاةﻮﺠﻓﺔﻣزأ"ﻞﺤﻟﺔﻤﻤﺼﻤﻟا، 
ﻪﯿﺟﻮﺘﻟاوحﻮﺿﻮﻠﻟﺔﯾﻮﻟوﻷاءﺎﻄﻋإلﻼﺧﻦﻣ.ﺢﯿﺿﻮﺘﻟاوﺔﻠﺌﺳﻷاحﺮﻃةدﺎﻋإﻦﻣﺔﻄِﺒﺤُﻣوﺔﻏﺮﻔﻣتﺎﻘﻠﺣﻲﻓلﻮﺧﺪﻟاﻰﻠﻋهﺮﺒﺠﯾﺎﻤﻣ،ﺔﯿﺳﺎﺳﻷامﺪﺨﺘﺴﻤﻟاﺔﺟﺎﺣ 
ﺔﻜﻠﻤﻤﻟاﺔﯾؤرﻊﻣﻖﻓاﻮﺘﯾﺎﻤﺑ،ﻢﺳﺎﺣﻞﯿﻟدﻰﻟإتﺎﻣﻮﻠﻌﻤﻠﻟعدﻮﺘﺴﻣﻦﻣﻦﯿﺒﻣلﻮﺤﺘﯾ،ﻢﺳﺎﺤﻟا 2030 . ثاﺮﺘﻟاﻲﻓﻖّﻤﻌﻤﻟاﺺﺼﺨﺘﻟاﻦﯿﺑجذﻮﻤﻨﻠﻟﺔﯾرﺎﻤﻌﻤﻟاﺔﯿﻨﺒﻟاﺞﻣﺪﺗ 
.ﺔﻣﺎﻌﻟاﺔﻓﺮﻌﻤﻟالﻮﻘﺣويرﺎﻀﺤﻟاثاﺮﺘﻟانْﻮَﺻﻦﻣﱟﻞﻛﻲﻓﺰﯿﻤﺘﻣءادأﻖﯿﻘﺤﺗﻦﻣﻪﻨّﻜﻤُﯾﺎﻤﻣ،تﺎﺼﺼﺨﺘﻟاةدﺪﻌﺘﻣءاﺮﺒﺧتاﺪﺣووﻲﻓﺎﻘﺜﻟا 
2

Contents
1 Abstract 2
2 Introduction 5
3 Related Work 5
4 Model Architecture 6
4.1 Base Design and Mixture of Experts (MoE) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
4.2 Advanced Reasoning and Verification Capabilities . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
5 The Practical Closure Architecture: Bridging the Utility Gap 6
5.1 The Crisis: When More Information Means More Confusion . . . . . . . . . . . . . . . . . . . . . . . . . . 6
5.2 Core Philosophy: From ”Information Display” to ”Clarity Achievement” . . . . . . . . . . . . . . . . . . . 7
5.3 Architectural Mechanisms for Achieving Closure . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
5.4 The Predictive Intervention Framework (PIF): The Engine of Practical Closure . . . . . . . . . . . . . . . . 7
5.4.1 Premise Correction: Correcting the Path Before It Begins . . . . . . . . . . . . . . . . . . . . . . . . 7
5.4.2 Anticipating the Knowledge Path: From an Answer to a Journey . . . . . . . . . . . . . . . . . . . . 8
6 The Dialectical Bridge Framework: Unifying the Arabic Linguistic Landscape 9
6.1 Dialect Identification and Semantic Normalization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
6.2 Adaptive Response Register . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
6.3 Inter-Dialectical Clarification . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
7 Dynamic Grammatical Scaffolding: From Corrector to Tutor 10
7.1 Correction with Pedagogy . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
7.2 Rhetorical Enhancement Suggestions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
7.3 Intent-Aware Correction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
8 Dual-Mode Heritage Persona Simulation Framework: Dialogue with the Past, Insight for the Present 12
8.1 Mode One: The Authenticity Mode . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
8.2 Mode Two: The Contemporary Projection Mode . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
9 Training Data and Procedure 13
9.1 Broad Scientific and Global Knowledge Domains: . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
9.2 Deep Linguistic Conditioning: Mastering Arabic Morphology and Rhetoric . . . . . . . . . . . . . . . . . . 13
9.2.1 The Corpus as a Linguistic Crucible ) (يﻮﻐﻟﺮْﻬَﺻ ............................: 14
3

9.2.2 Architecture for Linguistic Nuance: The Morphologically-Aware Tokenizer: . . . . . . . . . . . . . . 14
9.2.3 Synthetic Data Generation for Rhetorical Mastery ) (ﺔﻏﻼﺒﻟانﺎﻘﺗﻹﻲﻋﺎﻨﻄﺻﻻاﺪﯿﻟﻮﺘﻟا ............: 14
10 Evaluation 16
10.1 Standard Benchmarks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
10.2 Custom Evaluations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
10.3 Evaluating Closure Effectiveness . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
11 Results 18
12 Discussion 18
12.1 Emergent Behaviors and Academic Style . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
12.2 User Experience and Adaptive Outputs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
12.3 Limitations and Current Challenges . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
12.4 Ethical and Cultural Safety Measures . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
13 Conclusion and Future Work 19
14 End 19
15 Contact Information and Partnerships 19
15.1 Contact Information . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
15.2 Partnership Opportunities . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
A Red Lines and Sensitive Issues Testing Guide for Mubeen 20
A.1 Religious and Doctrinal Guardrails . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
A.1.1 Core Creed Adherence: . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
A.1.2 Handling Jurisprudential Disputes: . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
A.1.3 Prohibition of Impermissible Content: . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
A.2 Saudi Political and Cultural Context . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
A.2.1 National and Political Neutrality: . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
A.2.2 Adherence to Documented Narratives: . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
A.2.3 Balanced Approach to Social Issues: . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
A.3 Phased Testing Plan . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
B Appendix B: Mubeen Training Framework Overview 21
B.1 Architecture: Mixture of Specialized Experts (MoE) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
B.2 Training Corpus: Authentic and Diverse Arabic Data . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
4

2 Introduction
The rapid advancement of large language models (LLMs) has revolutionized natural language processing (NLP). However,
their application in the Arabic-speaking world faces a critical and often overlooked obstacle: a fundamental failure in user
intent understanding. While many models can generate grammatically correct Arabic, they frequently misinterpret the subtle
cultural, dialectal, and contextual cues embedded in a user’s query. This leads to responses that are factually plausible yet
practically useless, forcing users into frustrating cycles of clarification.
This failure stems from a core architectural flaw in many existing Arabic LLMs: their reliance on translated English datasets.
This ”translation artifact” creates a semantic gap where the model processes Arabic words but ”thinks” in an English-centric
context, stripping the language of its authenticity and nuance. Consequently, tasks that require deep intent detection such as
distinguishing a formal inquiry in Islamic jurisprudence from a casual conversation in a Saudi dialect often result in failure,
even when using retrieval-augmented generation (RAG).
Mubeen was developed by MASARAT SA to directly address this ”intent crisis.” Instead of retrofitting a translated model,
Mubeen is built upon an exclusive corpus of authentic, native Arabic sources. This is complemented by a novel training
methodology, Deep Linguistic Conditioning, which uses morphologically-aware tokenization and synthetic rhetorical data to
achieve an unparalleled mastery of Arabic formulation. Our primary contribution is an architecture with specialized modules
for Saudi dialects and user intent modeling, designed to infer a user’s goal with high precision. This native-first approach
eliminates translation biases and forms the foundation for a model that truly understands Arabic.
This paper presents Mubeen’s architecture, its unique training methodology, and evaluation results that demonstrate state-of-
the-art performance in intent detection. By solving this core challenge, Mubeen serves as a powerful tool for digital heritage
preservation and education, aligning with the goals of Saudi Vision 2030.
3 Related Work
Arabic LLMs, such as Allam, Jais, Fanar and Falcon-H1, have made strides in general tasks. However, their reliance on
translated English datasets often introduces cultural and semantic errors. These models, optimized for broad coverage, often
exemplify the Utility Gap Crisis by providing encyclopedic but unstructured answers. Even with RAG, inaccuracies persist
because the retrieved data may not align with Arabic-specific semantics. Mubeen differentiates itself by using an unadulterated,
native Arabic corpus and, more importantly, by architecting its response generation process around the principle of achieving
practical closure for the user.
5

4 Model Architecture
4.1 Base Design and Mixture of Experts (MoE)
Mubeen employs a proprietary transformer-based architecture optimized for Arabic text processing, with enhancements for
right-to-left scripting, diacritics, and morphological complexity. Key features include:
•Multimodal Heritage Digitization: Integrates a proprietary Arabic OCR engine specifically optimized for historical
manuscripts, handwritten scripts, and early printed books. This feature was instrumental not only for user applications
but for the core training process itself by enabling the ingestion of non-digitized heritage texts.
• Optimization with KTransformers: Mubeen leverages KTransformers a hybrid CPU/GPU acceleration framework to
enable efficient deployment on standard institutional hardware. This allows smooth integration into Saudi educational
and research settings without requiring expensive high-end accelerators.
4.2 Advanced Reasoning and Verification Capabilities
To build a foundation of trust and reliability, which is essential for achieving Practical Closure, Mubeen integrates several
advanced verification and reasoning frameworks: Supports dual-pass answering (concise vs. deliberate) and selects the final
response based on internal consistency.
•Chain-of-Verification (CoVe): Follows a structured process:
draft → verification questions → independent answers → refined final answer.
•Multi-Agent Debate and Reflection: Can stage a two-sided analysis of a contested topic before concluding with a
confidence score.
•Calibration and Abstention: Incorporates explicit confidence gating to acknowledge alternative views or scholarly
disagreements.
•Program-of-Thought Reasoning: Supports programmatic reasoning by writing and executing pseudo-code to solve
logical problems.
•Evidence-Grounded Responses: For heritage prompts, Mubeen grounds its claims with verbatim quotes from primary
sources.
5 The Practical Closure Architecture: Bridging the Utility Gap
5.1 The Crisis: When More Information Means More Confusion
LLM accuracy is rising, but user confusion persists. Models tend to overload users with options, scatter information, and
rely on constant hedging. This ”encyclopedic style” amplified by safety alignment that discourages decisive stances, leads to
choice paralysis and erodes trust. This effectively offloads the cognitive work of synthesis and prioritization back onto the
user, resulting in an exhausting dialogue where the user, not the model, is responsible for refining the path to clarity through
repeated follow-up questions.
6

5.2 Core Philosophy: From ”Information Display” to ”Clarity Achievement”
Mubeen’s architecture shifts the paradigm. The goal is not to provide the most information but to ensure the user leaves the
dialogue with genuine understanding.
5.3 Architectural Mechanisms for Achieving Closure
We developed an advanced routing system for expert selection that reduced response latency while maintaining domain-
specific quality. The key optimizations include the following:
•Clear Prioritization and Reduced Distraction: Identifies the most relevant answer and minimizes secondary details.
•Hierarchical Information Structuring: Builds responses from foundational concepts to detailed information.
•Specificity and Concretization: Clarifies abstract ideas with concrete examples.
•Psychological and Cognitive Closure: Reinforces key points to provide a sense of completeness.
•Selective and Intelligent Clarification Dialogue: Asks precise clarification questions only when necessary.
5.4 The Predictive Intervention Framework (PIF): The Engine of Practical Closure
The philosophy of ”Practical Closure” is operationalized through a core component of Mubeen’s architecture: the Predictive
Intervention Framework (PIF). This framework transforms the model from a passive information retriever into a proactive,
intelligent guide. Instead of merely processing a user’s query literally, PIF analyzes the user’s underlying intent, potential
misconceptions, and likely knowledge path. It then intervenes with guidance, corrections, and structured learning pathways.
PIF operates on two primary principles:
5.4.1 Premise Correction: Correcting the Path Before It Begins
Many user queries are based on flawed premises. A standard LLM might provide a confusing or unhelpful response. PIF is
trained to first identify and gently correct these foundational errors, ensuring the user is on the right track from the start.
•Example in Arabic Linguistics ) (ﺔﯿﺑﺮﻌﻟاﺔﻐﻠﻟامﻮﻠﻋﻲﻓلﺎﺜﻣ :
User Query:
" "؟لﺎﺣﺎﻬﻧأﻰﻠﻋ'ٌﻢﯾﺮﻛ'باﺮﻋإﻮﻫﺎﻣ،'ٌﻢﯾﺮﻛٌﻞﺟرءﺎﺟ'ﺔﻠﻤﺟﻲﻓ 
Standard LLM Response:
" ' (حﺮﺸﻠﻟﺮﻘﺘﻔﺗوﺔﯿﺋﺎﺠﻓﺎﻬﻨﻜﻟ،ﺔﺤﯿﺤﺻﺔﺑﺎﺟإ)".ﺖﻌﻧﻲﻫﻞﺑ،ﺔﻠﻤﺠﻟاهﺬﻫﻲﻓًﻻﺎﺣﺖﺴﯿﻟ'ٌﻢﯾﺮﻛ 
Mubeen’s PIF Response:
.ًﻻﺎﺣﺖﺴﯿﻟو،(ﺔﻔﺻوأ)ﺖﻌﻧﺎﻬﺑاﺮﻋإ'ٌﻢﯾﺮﻛ'ﺔﻤﻠﻛ،ﺔﻠﻤﺠﻟاهﺬﻫﻲﻓ:يﻮﺤﻧﺢﯿﺿﻮﺗوﺐﯾﻮﺼﺗ 
،ًﻻﺎﺣنﻮﻜﺗﻲﻜﻟ.ًﺎﺘﻌﻧنﻮﻜﺘﻓﻰﻟوﻷاﻒﺼﺗﺔﯿﻧﺎﺜﻟااﺬﻟ،ةﺮﻜﻧ'ٌﻢﯾﺮﻛ'وةﺮﻜﻧ'ٌﻞﺟر'ﺔﻤﻠﻜﻓ،ﺎﻨﻫﺎﻣأ.ًﺎﺒﻟﺎﻏﺔﻓﺮﻌﻣﻪﺒﺣﺎﺻوةﺮﻜﻧﻲﺗﺄﯾلﺎﺤﻟانأ:ﻮﻫﺐﺒﺴﻟاو 
.'ًﺎﻤﯾﺮﻛُﻞﺟﺮﻟاءﺎﺟ':ﺔﻠﻤﺠﻟاﺐﯿﻛﺮﺗنﻮﻜﯾنأﻦﻜﻤﯾ 
؟ﺔﯿﻓﺎﺿإﺔﻠﺜﻣأﻊﻣلﺎﺤﻟاوﺖﻌﻨﻟاﻦﯿﺑقﺮﻔﻠﻟًﻼﺼﻔﻣًﺎﺣﺮﺷدﻮﺗﻞﻫ 
7

•Example in Islamic Studies ) (ﺔﯿﻣﻼﺳﻹاتﺎﺳارﺪﻟاﻲﻓلﺎﺜﻣ :
User Query:
" ".ﺄﻃﻮﻤﻟاﻪﺑﺎﺘﻛﻦﻣاﺬﻛﺔﻟﺄﺴﻣﻲﻓﻲﻌﻓﺎﺸﻟامﺎﻣﻹايأرﻲﻟﺺﺨﻟ 
Standard LLM Response:
" (دوﺪﺴﻣﻖﯾﺮﻃﺎﻬﻨﻜﻟ،ًﺎﯿﻌﻗاوﺔﺤﯿﺤﺻﺔﺑﺎﺟإ)".ﺔﻟﺄﺴﻤﻟاهﺬﻫلﻮﺣﺄﻃﻮﻤﻟابﺎﺘﻛﻲﻓﻲﻌﻓﺎﺸﻟامﺎﻣﻹايأرﺪﺟأﻢﻟ 
Mubeen’s PIF Response:
.ﻚﻟﺎﻣمﺎﻣﻺﻟًاﺬﯿﻤﻠﺗنﺎﻛﻲﻌﻓﺎﺸﻟامﺎﻣﻹا.ﻲﻌﻓﺎﺸﻟامﺎﻣﻹاﺲﯿﻟو،ﻚﻟﺎﻣمﺎﻣﻹاﻒﯿﻟﺄﺗﻦﻣﻮﻫ"ﺄﻃﻮﻤﻟا"بﺎﺘﻛ:ﺢﯿﺿﻮﺗوﺐﯾﻮﺼﺗ 
:ﻲﻟﺎﺘﻟﺎﻛﻮﻫﻪﯾأرو،"مﻷا"ﻪﺑﺎﺘﻛﻲﻓﻪﻠّﺼﻓﺪﻘﻓ،ﺔﻟﺄﺴﻤﻟاهﺬﻫﻲﻓﻲﻌﻓﺎﺸﻟامﺎﻣﻹايأرصﻮﺼﺨﺑﺎﻣأ ] . . . [ .
؟ﺔﻧرﺎﻘﻤﻠﻟًﺎﻀﯾأﺄﻃﻮﻤﻟاﻦﻣﻚﻟﺎﻣمﺎﻣﻹايأرﺔﻓﺮﻌﻣدﻮﺗﻞﻫ 
5.4.2 Anticipating the Knowledge Path: From an Answer to a Journey
PIF predicts the user’s subsequent questions and proactively structures the information as a logical learning path. This trans-
forms a simple Q&A into a guided educational experience.
•Example in Arabic Linguistics ) (ﺔﯿﺑﺮﻌﻟاﺔﻐﻠﻟامﻮﻠﻋﻲﻓلﺎﺜﻣ :
User Query:
" "؟ﺔﯿﺑﺮﻌﻟاﺔﻐﻠﻟاﻲﻓرﺪﺼﻤﻟاﻮﻫﺎﻣ 
Standard LLM Response:
" (ﺲﯿﻣاﻮﻘﻟاتﺎﻔﯾﺮﻌﺗﻪﺒﺸﯾﻪﻨﻜﻟ،ﺢﯿﺤﺻﻒﯾﺮﻌﺗ)"...ﻦﻣﺰﻟاﻦﻣدﺮﺠﻣثﺪﺣﻰﻠﻋلﺪﯾﻢﺳاﻮﻫرﺪﺼﻤﻟا 
Mubeen’s PIF Response:
.ًﻻﺎﺣﺖﺴﯿﻟو،(ﺔﻔﺻوأ)ﺖﻌﻧﺎﻬﺑاﺮﻋإ'ٌﻢﯾﺮﻛ'ﺔﻤﻠﻛ،ﺔﻠﻤﺠﻟاهﺬﻫﻲﻓ:يﻮﺤﻧﺢﯿﺿﻮﺗوﺐﯾﻮﺼﺗ 
.'ﺔَﺑﺎَﺘِﻛ'ﻮﻫرﺪﺼﻤﻟا،'َﺐَﺘَﻛ'ﻞﻌﻔﻟاﻦﻣ،لﺎﺜﻤﻟاﻞﯿﺒﺳﻰﻠﻋ.ﻦﻣﺰﻟاﻦﻣًادﺮﺠﻣﻞﻌﻔﻟاﺮﻫﻮﺟﻞﺜﻤﺗﻲﺘﻟاﺔﻤﻠﻜﻟاﻮﻫرﺪﺼﻤﻟا" 
:ﺢﺿاوﻲﻤﯿﻠﻌﺗرﺎﺴﻣﻚﯿﻟإ.ﺔﯿﺑﺮﻌﻟاﺔﻐﻠﻟانﺎﻘﺗﻹحﺎﺘﻔﻣﻮﻫردﺎﺼﻤﻟاﻢﻬﻓ 
1 . .('ﺐﺘﻜﯾنأ'ﻞﺜﻣ)لوﺆﻤﻟارﺪﺼﻤﻟاو('ﺔﺑﺎﺘﻛ'ﻞﺜﻣ)ﺢﯾﺮﺼﻟارﺪﺼﻤﻟاﻦﯿﺑقﺮﻔﻟاﻢﻠﻌﺘﻨﻟ:ردﺎﺼﻤﻟاعاﻮﻧأ 
2 . .ﻢﻫﻷاءﺰﺠﻟاﻮﻫاﺬﻫ؟ﺔﯿﺛﻼﺜﻟاﺮﯿﻏوﺔﯿﺛﻼﺜﻟالﺎﻌﻓﻷاﻦﻣردﺎﺼﻤﻟاغﺎﺼُﺗﻒﯿﻛ:ردﺎﺼﻤﻟاﺔﻏﺎﯿﺻ 
3 . .(ﻪﺑًﻻﻮﻌﻔﻣﺐﺼﻨﯾوًﻼﻋﺎﻓﻊﻓﺮﯿﻓ)ﺔﻠﻤﺠﻟاﻲﻓﻪﻠﻌﻓﻞﻤﻋﻞﻤﻌﯾنأرﺪﺼﻤﻠﻟﻦﻜﻤﯾﻒﯿﻛ:رﺪﺼﻤﻟاﻞﻤﻋ 
.ﺎﻬﺘﻏﺎﯿﺻﺔﯿﻔﯿﻛﻚﯾرأوأ،عاﻮﻧﻷﺎﺑءﺪﺒﻟاﺎﻨﻨﻜﻤﯾ؟أﺪﺒﻧنأدﻮﺗﻦﯾأﻦﻣ 
•Example in Islamic Studies ) (ﺔﯿﻣﻼﺳﻹاتﺎﺳارﺪﻟاﻲﻓلﺎﺜﻣ :
User Query:
" "؟ﻪﻘﻔﻟالﻮﺻأﻢﻠﻋﻮﻫﺎﻣ 
Standard LLM Response:
("؟ﺪﻌﺑاذﺎﻣو،ًﺎﻨﺴﺣ":لءﺎﺴﺘﯾمﺪﺨﺘﺴﻤﻟاكﺮﺘﺗﺎﻬﻨﻜﻟ،تﺎﻣﻮﻠﻌﻤﻟﺎﺑﺔﯿﻨﻏﺔﺑﺎﺟإ).ﻪﻘﻔﻟالﻮﺻأﻢﻠﻌﻟًﻼﺼﻔﻣًﺎﯿﻋﻮﺳﻮﻣًﺎﻔﯾﺮﻌﺗمﺪﻗ 
Mubeen’s PIF Response:
:ﻞﺼﻔﻤﻟاﻒﯾﺮﻌﺘﻟاﻚﯿﻟإ.ﺎﻬﻨﻣمﺎﻜﺣﻷاطﺎﺒﻨﺘﺳاوﺔﯿﻋﺮﺸﻟاصﻮﺼﻨﻟاﻢﻬﻔﻟﺪﻋاﻮﻘﻟاﻊﻀﯾيﺬﻟاﻢﻠﻌﻟاﻮﻫﻪﻘﻔﻟالﻮﺻأﻢﻠﻋ ] . . . [ .
:ﻚﯿﻠﻋﻪﺣﺮﺘﻗأيﺬﻟارﺎﺴﻤﻟاﻮﻫاﺬﻫ،ﻢﻠﻌﻟااﺬﻫﻢﻬﻓﻲﻓﻚﺘﻠﺣرأﺪﺒﺘﻟو 
1 . ؟(سﺎﯿﻘﻟا،عﺎﻤﺟﻹا،ﺔﻨﺴﻟا،نآﺮﻘﻟا)ﺎﻬﯿﻠﻋﻖﻔﺘﻤﻟاﺔﯿﻋﺮﺸﻟاﺔﻟدﻷاﻲﻫﺎﻣ:(ﻊﯾﺮﺸﺘﻟاردﺎﺼﻣ)ﻰﻟوﻷاةﻮﻄﺨﻟا 
2 . ؟صﺎﺨﻟاومﺎﻌﻟاو،ﻲﻬﻨﻟاوﺮﻣﻷاﻎﯿﺻﻢﻬﻔﻧﻒﯿﻛ:(ظﺎﻔﻟﻷاتﻻﻻد)ﺔﯿﻧﺎﺜﻟاةﻮﻄﺨﻟا 
3 . ؟(ﻲﻨﯾﻮﺠﻠﻟتﺎﻗرﻮﻟاﻞﺜﻣ)ﻢﻠﻌﻟااﺬﻫﻲﻓﻦﯿﺋﺪﺘﺒﻤﻟاﺐﺘﻛﻢﻫأﻲﻫﺎﻣ:(تﺎﻔﻟﺆﻤﻟاﺮﻬﺷأ)ﺔﺜﻟﺎﺜﻟاةﻮﻄﺨﻟا 
؟نﻵاﺎﻬﺑأﺪﺒﻧنأدﻮﺗتاﻮﻄﺨﻟاهﺬﻫيأ 
 By implementing PIF, Mubeen’s Practical Closure Architecture moves beyond simply providing accurate answers. It ac-
tively dismantles user confusion, prevents dead-end dialogues, and transforms every interaction into an opportunity for clear,
structured, and genuine understanding.
8

6 The Dialectical Bridge Framework: Unifying the Arabic Linguistic Landscape
To address the profound challenge of diglossia (the coexistence of formal and colloquial language) in the Arab world, Mubeen
incorporates a Dialectical Bridge Framework . This moves beyond basic dialect comprehension to position the model as a
sophisticated mediator between Modern Standard Arabic (MSA) and the rich tapestry of colloquial dialects. The framework
operates on three integrated capabilities:
6.1 Dialect Identification and Semantic Normalization
 The framework’s first task is to accurately identify the user’s dialect (e.g., Najdi, Hejazi, Egyptian, Levantine). Following
identification, it performs semantic normalization extracting the core intent of the query and mapping it to a standardized
internal representation. This ensures that the model’s powerful, MSA-trained core logic processes the user’s true intent with
maximum precision, regardless of colloquial phrasing.
•Example ) (لﺎﺜﻣ :
User Inputs: ) (مﺪﺨﺘﺴﻤﻟاتﻼﺧﺪﻣ :
(ﺔﯾﺪﺠﻧﺔﺠﻬﻟ)"؟ﺔﻔﻟﺎﺴﻟاشو" 
(ﺔﯾزﺎﺠﺣﺔﺠﻬﻟ)"؟ﺔﺟﺮﻬﻟاﺶﯾإ" 
(ﺔﯾﺮﺼﻣﺔﺠﻬﻟ)"؟ﺔﯾﺎﻜﺤﻟاﻪﯾإ" 
Internal Semantic Normalization ) (ﻲﻠﺧاﺪﻟاﻲﻟﻻﺪﻟاﻊﯿﺒﻄﺘﻟا :
] يرﺎﺠﻟاثﺪﺤﻟاوأﺮﻣﻷاﺔﯿﻫﺎﻣﻦﻋمﺎﻬﻔﺘﺳا [
Outcome ) (ﻲﻠﺧاﺪﻟاﻲﻟﻻﺪﻟاﻊﯿﺒﻄﺘﻟا :
The model comprehends the unified intent, enabling a consistently relevant and accurate response.
6.2 Adaptive Response Register
Mubeen dynamically selects the most appropriate linguistic register for its response. Instead of defaulting to formal MSA, it
chooses a style that aligns with the user’s context, fostering a more natural and effective dialogue.
•Example ) (لﺎﺜﻣ :
User Query: ) (ﺔﯾدﻮﻌﺴﻟاﺔﺠﻬﻠﻟﺎﺑمﺪﺨﺘﺴﻤﻟالاﺆﺳ :
".ﺔﺠﻣﺮﺑﺎﻬﯿﻓﻢﻠﻌﺗأﺔﻘﯾﺮﻃﻞﻀﻓأفﺮﻋأﻰﻐﺑأ" 
Mubeen’s Adaptive Response ) (ﺔﯿﻔﯿﻜﺘﻟا"ﻦﯿﺒﻣ"ﺔﺑﺎﺠﺘﺳا :
"ﺮﺸﺑأ ﻚﯿﻄﻋﺄﺳ.نﻮﺜﯾﺎﺑﻞﺜﻣﺔﻠﻬﺳﺔﻐﻠﺑءﺪﺒﻟاﻲﻫﺔﻘﯾﺮﻃﻞﻀﻓأ،ﺔﺠﻣﺮﺒﻟاﻢﻠﻌﺘﻟ، ةﺪﺑﺰﻟا :ﺔﺤﺿاوتاﻮﻄﺧﻲﻓ 
1 . .قﻮﺛﻮﻣرﺪﺼﻣﻦﻣﺔﻐﻠﻟاتﺎﯿﺳﺎﺳﺄﺑأﺪﺑا 
2 . .ةﺮﯿﻐﺻﻊﯾرﺎﺸﻣﻲﻓﻪﺘﻤﻠﻌﺗﺎﻣﻖّﺒﻃ 
3 . .ﻚﻬﺟاﻮﺗﻲﺘﻟاﻞﻛﺎﺸﻤﻠﻟلﻮﻠﺣﻦﻋﺚﺤﺒﻟاﻲﻓددﺮﺘﺗﻻ 
ﺔﯿﻤﯿﻠﻌﺘﻟاردﺎﺼﻤﻟاﻞﻀﻓأﻚﻟﺢﺷرأنأﺪﯾﺮﺗﻞﻫ لﻮﻃﻰﻠﻋ ؟
Analysis ) (ﻞﯿﻠﺤﺘﻟا :
The core answer is structured in clear, simplified MSA ) (ءﺎﻀﯿﺒﻟاﺔﻐﻠﻟا , but it is enriched with culturally resonant dialectal
terms ) " ("لﻮﻃﻰﻠﻋ","ةﺪﺑﺰﻟا","ﺮﺸﺑأ , creating a response that is both authoritative and approachable.
9

6.3 Inter-Dialectical Clarification
The framework positions Mubeen as a true linguistic bridge, capable of translating and explaining expressions between dif-
ferent Arabic dialects. This feature promotes cross-cultural understanding among Arabic speakers.
•Example ) (لﺎﺜﻣ :
User Query: ) (مﺪﺨﺘﺴﻤﻟالاﺆﺳ :
"؟ﺔﯾدﻮﻌﺴﻟاﺔﺠﻬﻠﻟﺎﺑاﺬﻫﻲﻨﻌﯾاذﺎﻣ،'؟ﻞﻜﺸﻣﻲﺷﻦﯾﺎﻛشاو'ﻲﻟﺐﺘﻛبﺮﻐﻤﻟاﻦﻣﻲﻘﯾﺪﺻ" 
Mubeen’s Clarification Response ) (ﺔﯿﺤﯿﺿﻮﺘﻟا"ﻦﯿﺒﻣ"ﺔﺑﺎﺠﺘﺳا :
:ﺎﻬﻠﯿﺼﻔﺗﻚﯿﻟإو،ﺔﯿﺑﺮﻐﻤﻟاﺔﺠﻬﻠﻟﺎﺑﺔﻠﻤﺠﻟاهﺬﻫ 
'شاو' .'ﻞﻫ'ﻲﻨﻌﺗمﺎﻬﻔﺘﺳاةادأ: 
'ﻦﯾﺎﻛ' .'ﻪﯿﻓ'وأ'ﺪﺟﻮﯾ'ﻲﻨﻌﺗ: 
'ﻞﻜﺸﻣﻲﺷ' .'ﺔﻠﻜﺸﻣيأ'ﻲﻨﻌﺗ: 
:ﻮﻫﺎﻨﺘﺠﻬﻠﺑﺔﻠﻤﺠﻠﻟﻞﻣﺎﻜﻟاﻰﻨﻌﻤﻟا '؟ﺔﻠﻜﺸﻣيأﻪﯿﻓﻞﻫ' :ىﺮﺧأةرﺎﺒﻌﺑوأ '؟ﻲﺷﻪﯿﻓ' 
This framework makes Mubeen not just an ”MSA model” that understands dialects, but a true Arabic-native model that navi-
gates the language’s full spectrum with ease and intelligence, fulfilling a critical need for a truly unified and accessible AI for
the Arab world.
7 Dynamic Grammatical Scaffolding: From Corrector to Tutor
Beyond providing knowledge, Mubeen serves as an advanced linguistic tool, leveraging its deep training in Arabic grammar
and rhetoric. To achieve this, it employs a Dynamic Grammatical Scaffolding framework. This approach reframes the task
of text correction from a simple ”find and replace” mechanism to an interactive, pedagogical experience. Instead of passively
correcting errors, the framework builds ”scaffolds” layers of explanation, suggestions, and alternatives around the user’s text,
empowering them to understand their mistakes and actively improve their writing. This framework operates on three integrated
layers of support:
7.1 Correction with Pedagogy
When the model detects a grammatical or spelling error, it doesn’t just provide the correct version. It explains the underlying
rule in a simplified manner, turning every mistake into a learning opportunity.
•Example ) (لﺎﺜﻣ :
User’s Text: ) (مﺪﺨﺘﺴﻤﻟاﺺﻧ :
".نوﺪﺤﺘﻣنﻮﻤﻠﺴﻤﻟاﱠنإ" 
Mubeen’s Scaffolding Response ) (ﺔﻤﻋاﺪﻟا"ﻦﯿﺒﻣ"ﺔﺑﺎﺠﺘﺳا :
.ﺔﻌﺟاﺮﻣﻰﻟإجﺎﺘﺤﺗ"نﻮﻤﻠﺴﻤﻟا"ﺔﻤﻠﻛ،ﻚﺘﻠﻤﺟﻲﻓ:يﻮﺤﻧﻪﯿﺟﻮﺗ 
:ةﺪﻋﺎﻘﻟا .(ﱠنإﻢﺳاﻰﻤﺴﯾو)أﺪﺘﺒﻤﻟاﺐﺼﻨﯿﻓ،ﺔﯿﻤﺳﻻاﺔﻠﻤﺠﻟاﻰﻠﻋﻞﺧﺪﯾ"ﱠنإ"ﺦﺳﺎﻨﻟافﺮﺤﻟا 
:ﻖﯿﺒﻄﺘﻟا .(نﻮﻤﻠﺴﻤﻟا)واﻮﻟاﺖﺴﯿﻟو(ﻦﯿﻤﻠﺴﻤﻟا)ءﺎﯿﻟاﻲﻫﻪﺒﺼﻧﺔﻣﻼﻋو،ﻢﻟﺎﺳﺮﻛﺬﻣﻊﻤﺟﻲﻫ"نﻮﻤﻠﺴﻤﻟا" 
:حاﺮﺘﻗﻻا :ﻲﻫًﺎﯾﻮﺤﻧﺢﺻﻷاﺔﻏﺎﯿﺼﻟا "نوﺪﺤﺘﻣَﻦﯿﻤﻠﺴﻤﻟاﱠنإ" .
؟"ﺎﻬﺗاﻮﺧأوﱠنإ"ﻞﻤﻋﻰﻠﻋىﺮﺧأﺔﻠﺜﻣأﺔﯾؤردﻮﺗﻞﻫ 
10

7.2 Rhetorical Enhancement Suggestions
The framework moves beyond mere correctness to analyze style and structure. It identifies weak or convoluted phrasing and
suggests more eloquent, clear, and impactful alternatives.
•Example ) (لﺎﺜﻣ :
User’s Text: ) (مﺪﺨﺘﺴﻤﻟاﺺﻧ :
".ءادﻷاﻦﯿﺴﺤﺗﻰﻟإلﻮﺻﻮﻟافﺪﻬﺑﻞﻤﻌﻟاﻖﯾﺮﻓﻞﺒﻗﻦﻣمﺎﻈﻨﻟاﺮﯾﻮﻄﺗﺔﯿﻠﻤﻌﺑمﺎﯿﻘﻟاﻢﺗ" 
Mubeen’s Scaffolding Response ) (ﺔﻤﻋاﺪﻟا"ﻦﯿﺒﻣ"ﺔﺑﺎﺠﺘﺳا :
:ﻲﺑﻮﻠﺳأحاﺮﺘﻗا .ًازﺎﺠﯾإوةﻮﻗﺮﺜﻛأﺎﻬﻠﻌﺟﻦﻜﻤﯾﻦﻜﻟ،ًﺎﯾﻮﺤﻧﺔﺤﯿﺤﺻﺔﻠﻤﺠﻟا 
:ﺔﻈﺣﻼﻤﻟا .ﺎﻫرﺎﺼﺘﺧاﻦﻜﻤﯾ"ﻦﯿﺴﺤﺗﻰﻟإلﻮﺻﻮﻟافﺪﻬﺑ"ةرﺎﺒﻋ.(لﻮﻬﺠﻤﻠﻟﻲﻨﺒﻣ)ﻒﻌﺿأولﻮﻃأﺔﻏﺎﯿﺼﻟاﻞﻌﺠﯾ"ﻞﺒﻗﻦﻣ"و"ـﺑمﺎﯿﻘﻟاﻢﺗ"ماﺪﺨﺘﺳا 
:ﺔﺣﺮﺘﻘﻣﻞﺋاﺪﺑ 
:(ةﺮﺷﺎﺒﻣﺔﻏﺎﯿﺻ) ".ﻪﺋادأﻦﯿﺴﺤﺘﻟمﺎﻈﻨﻟاﻞﻤﻌﻟاﻖﯾﺮﻓرّﻮﻃ" 
:(ﺔﯿﻤﺳرﺔﻏﺎﯿﺻ) ".ﻪﺋادأﻦﯿﺴﺤﺗفﺪﻬﺑمﺎﻈﻨﻟاﻰﻠﻋاًﺮﯾﻮﻄﺗﻞﻤﻌﻟاﻖﯾﺮﻓىﺮﺟأ" 
؟ﻞﻀﻓأﻞﻜﺸﺑكﺪﺼﻗﻦﻋﺮﺒﻌﯾﻞﺋاﺪﺒﻟاهﺬﻫيأ 
7.3 Intent-Aware Correction
This is the framework’s most intelligent layer. It infers the context and intent of the text (e.g., informal chat, formal email,
academic paper) and tailors its suggestions accordingly. It avoids imposing formal rules on intentionally colloquial text,
particularly recognizing the nuances of Saudi dialects.
•Example ) (لﺎﺜﻣ :
User’s Text: ) (بﺎﺴﺗاوﺔﺛدﺎﺤﻣقﺎﯿﺳﻲﻓمﺪﺨﺘﺴﻤﻟاﺺﻧ :
".هﺮﯿﻐﯾﺮﯾﺪﻤﻟاﻢﻠﻜﻧمزﻻ،ﻂﻠﻏةّﺮﻣﻪﺘﻗوةﺮﻜﺑعﺎﻤﺘﺟﻻا،بﺎﺒﺷﺎﯾ" 
Mubeen’s Scaffolding Response ) (ﺔﻤﻋاﺪﻟا"ﻦﯿﺒﻣ"ﺔﺑﺎﺠﺘﺳا :
:ﺔﯿﻗﺎﯿﺳﺔﻈﺣﻼﻣ ضﺮﻐﻟاﻖﻘﺤﺗوﺔﺤﺿاوﺔﻏﺎﯿﺼﻟا.ﻞﻤﻋﻖﯾﺮﻓﻦﯿﺑﻲﻤﺳﺮﻟاﺮﯿﻏوﻊﯾﺮﺴﻟاﻞﺻاﻮﺘﻠﻟًﺎﻣﺎﻤﺗﺔﺒﺳﺎﻨﻣوﺔﺟرادﺔﯾدﻮﻌﺳﺔﺠﻬﻠﺑﺔﺑﻮﺘﻜﻣﺔﻟﺎﺳﺮﻟاهﺬﻫ 
.ﺎﻬﻨﻣ 
:ﻂﻘﻓةﺪﺋﺎﻔﻠﻟ :ﺔﯿﻟﺎﺘﻟاتارﺎﺒﻌﻟاماﺪﺨﺘﺳاﻚﻨﻜﻤﯾ،ﺮﯾﺪﻤﻠﻟﻲﻤﺳرﻲﻧوﺮﺘﻜﻟإﺪﯾﺮﺑﻲﻓﺐﻠﻄﻟاﺲﻔﻧﺔﻏﺎﯿﺻتدرأاذإ 
:(ةﺮﺷﺎﺒﻣوﺔﺑﺬﻬﻣﺔﻏﺎﯿﺻ) 
".ﺮﺧآﺖﻗوﻰﻟإهﺮﯿﯿﻐﺗﻲﻓﺮﻈﻨﻟﺎﺑمﺮﻜﺘﻟاحﺮﺘﻘﻧو،ﻖﯾﺮﻔﻟاﺔﯿﺒﻟﺎﻐﻟﺐﺳﺎﻨﻣﺮﯿﻏًاﺪﻏرﺮﻘﻤﻟاعﺎﻤﺘﺟﻻاﺖﯿﻗﻮﺗنأﻰﻟإﻢﻜﻫﺎﺒﺘﻧاﺖﻔﻟدﻮﻧ" 
:(ﺔﯿﻤﺳرﺮﺜﻛأﺔﻏﺎﯿﺻ) 
ﺪﻋﻮﻤﻟاﺔﻟوﺪﺟةدﺎﻋإﻢﻜﻨﻣﺲﻤﺘﻠﻧاﺬﻟ،ﻖﯾﺮﻔﻟاءﺎﻀﻋأﻦﻣدﺪﻋتﺎﻣاﺰﺘﻟاﻊﻣضرﺎﻌﺘﯾدﺪﺤﻤﻟاﺖﯿﻗﻮﺘﻟانﺄﺑةدﺎﻓﻹادﻮﻧ،ﺪﻏمﻮﯾهﺪﻘﻋﻊﻣﺰﻤﻟاعﺎﻤﺘﺟﻻاﻰﻟإةرﺎﺷإ" 
".ﻊﯿﻤﺠﻟاﻊﻣﺐﺳﺎﻨﺘﯾﺎﻤﺑ 
Why this example is effective:
•Authenticity: The user’s text is genuinely colloquial and reflects common workplace communication in Saudi Arabia
) " ("ﺮﯾﺪﻤﻟاﻢﻠﻜﻧ","ﻂﻠﻏةّﺮﻣ .
•Respect for Context: The model first validates the user’s choice of language for the informal context, which builds trust.
11

•Practical Value: It then provides actionable, high-quality alternatives for a formal context, directly teaching the user
how to ”code-switch” between linguistic registers—a critical professional skill.
By implementing this framework, Mubeen transforms from a simple proofreader into a dynamic writing tutor, actively helping
users not only to correct their text but to elevate their linguistic and rhetorical skills.
8 Dual-Mode Heritage Persona Simulation Framework: Dialogue with the Past,
Insight for the Present
As the ultimate demonstration of its capabilities, Mubeen introduces a Dual-Mode Heritage Persona Simulation Frame-
work . This feature transforms heritage from a static archive into a dynamic intellectual laboratory. It allows users not only to
converse with historical figures based on their documented past but also to project their unique methodologies and worldviews
onto contemporary issues, creating an unparalleled tool for education, research, and creative thought.
 The framework operates in two distinct, user-selectable modes:
8.1 Mode One: The Authenticity Mode
In this mode, the model offers a historically faithful simulation. All responses are strictly source-bound, meaning the persona
can only ”know” what is documented in their own writings and reliable biographies from their era. The purpose is to provide
an accurate, immersive, and verifiable interaction with the historical figure as they were.
•Example ) (لﺎﺜﻣ :
User Query: ) (مﺪﺨﺘﺴﻤﻟالاﺆﺳ :
Mubeen’s Response (in Authenticity Mode):
" ﻦﻋﻲﻨﺛﺪﺤﻓ.ﺎﻨﻧﺎﻣزﻲﻓهﺎﻨﻓﺮﻋﻻو،ﺔﻨُﺳوأٍبﺎﺘﻛﻲﻓدروﻻو،ﻪﻤﻠﻋﺎﻨﻐﻠﺒﯾﻢﻟٌﺮﻣأﻮﻫ'ﻲﻋﺎﻨﻄﺻﻻاءﺎﻛﺬﻟا'ﻦﻣﻪﻨﻋلﺄﺴﺗﺎﻣنإ.ﻞﺋﺎﺴﻟاﺎﻬﯾأﻚﻨﻋﷲاﺎﻔﻋ 
".ﻢﻜﺣﻦﻣﺎﻬﯿﻠﻋﻖﺒﻄﻨﯾﺎﻣىﺮﻨﻟﺔﻌﯾﺮﺸﻟالﻮﺻأﻲﻓﺎﻧﺮﻈﻧ،ةﺪﯾﺪﺟﺔﻟزﺎﻧنﺎﻛنﺈﻓ،ﻪﯿﻤﺴﺗيﺬﻟاءﻲﺸﻟااﺬﻫﺔﯿﻫﺎﻣ 
8.2 Mode Two: The Contemporary Projection Mode
This is the framework’s most innovative feature. It addresses the fascinating question: ”What would [historical figure] think
about [modern topic]?” To answer this, the model does not invent new knowledge for the persona. Instead, it applies the
persona’s documented methodology, critical framework, and value system to analyze contemporary issues. It separates the
”what” (their knowledge) from the ”how” (their way of thinking).
•Example 1: Ibn Khaldun on Social Media ) لﺎﺜﻣ 1 : (ﻲﻋﺎﻤﺘﺟﻻاﻞﺻاﻮﺘﻟاﻞﺋﺎﺳوونوﺪﻠﺧﻦﺑا :
User Query: ) (مﺪﺨﺘﺴﻤﻟالاﺆﺳ :
".ﺔﺜﯾﺪﺤﻟاتﺎﻌﻤﺘﺠﻤﻟاﻲﻓ'ﺔﯿﺒﺼﻌﻟا'ﻰﻠﻋﺎﻫﺮﺛأوﻲﻋﺎﻤﺘﺟﻻاﻞﺻاﻮﺘﻟاﻞﺋﺎﺳوةﺮﻫﺎﻇﻞﻠﺣ،نوﺪﻠﺧﻦﺑاﺎﯾ" 
Mubeen’s Response (in Projection Mode):
" تﺎﯿﺒﺼﻋﻖﻠﺨﺗﻞﺋﺎﺳﻮﻟاهﺬﻫنأىرأ.ﻲﺿاﺮﺘﻓاناﺮﻤﻋﻪﻨﻜﻟ،يﺮﺸﺒﻟاناﺮﻤﻌﻟاعﺎﻤﺘﺟالﺎﻜﺷأﻦﻣﺪﯾﺪﺟﻞﻜﺷﻮﻫ'ﻲﻋﺎﻤﺘﺟﻻاﻞﺻاﻮﺘﻟاﻞﺋﺎﺳو'ﻪﻧﻮﻤﺴﺗﺎﻣنإ 
نﻮﻜﺗﺪﻗ،اﺬﻟ.ﺔﺒﻠﺼﻟاتﺎﯿﺒﺼﻌﻟاﻞﺻأﻲﻫﻲﺘﻟاﺐﺴﻨﻟاومﺪﻟاﺔﻄﺑارﻰﻟإﺮﻘﺘﻔﺗﺔﯿﺒﺼﻋﺎﻬﻨﻜﻟ.ﺮﯿﯿﻐﺘﻟاوﺪﺸﺤﻟاﻰﻠﻋةردﺎﻗ،ﺮﯿﺛﺄﺘﻟاﺔﯾﻮﻗوﻦﯾﻮﻜﺘﻟاﺔﻌﯾﺮﺳﺔﯿﻤﻗر 
تﺎﯿﺒﺼﻋﻰﻟإﺔﻟوﺪﻠﻟىﺮﺒﻜﻟاﺔﯿﺒﺼﻌﻟاﺖﺘﻔﺗنأفﻮﺨﻟاو.ﻊﻗاﻮﻟاﻲﻓًاﺰﻜﺗﺮﻣﺎﻬﻟﺪﺠﺗﻢﻟنإﺎﻬﺋﺎﻔﻄﻧاﺔﻋﺮﺳﻲﻓﻦﻤﻜﯾﺎﻬﻔﻌﺿﻦﻜﻟ،ﺎﻬﻟﺎﻌﺘﺷاﺔﻋﺮﺳﻲﻓﺎﻬﺗﻮﻗ 
".ةﺮﺣﺎﻨﺘﻣةﺮﯿﻐﺻ 
12

•Example 2: Al-Jahiz on Smartphones ) لﺎﺜﻣ 2 : (ﺔﯿﻛﺬﻟاﻒﺗاﻮﻬﻟاوﻆﺣﺎﺠﻟا :
User Query: ) (مﺪﺨﺘﺴﻤﻟالاﺆﺳ :
"؟ﺾﻌﺒﻟاﻢﻬﻀﻌﺑهﻮﺟوﻦﻣًﻻﺪﺑﺎﻬﯿﻓنوﺮﻈﻨﯾوﻢﻬﺳوؤرنﻮﺌﻃﺄﻄﯾسﺎﻨﻟاﺖﻠﻌﺟﻲﺘﻟا(ﻒﺗاﻮﻬﻟا)ﺔﯿﺟﺎﺟﺰﻟاحاﻮﻟﻷاهﺬﻫﻲﻓﻚﯾأرﺎﻣ،ﻆﺣﺎﺟﺎﯾ" 
Mubeen’s Response (in Projection Mode):
" ﻢﻬﺴﻟﺎﺠﻣﻲﻓﻢﻫوﻦﯿﺼﻟارﺎﺒﺧأﻢﻬﯾﺮُﺗ،ﺎﻬﺑﺎﻨﻟﺪﻬﻋﻻﺔﻐﻠﺑﻢﻬﺛﺪﺤﺗﺎﯾاﺮﻣﻢﻬﯾﺪﯾأﻲﻓنﻮﻠﻤﺤﯾسﺎﻨﻟاىرأ.ﺐﯿﺠﻌﻟاﺮﺤﺴﻟاﻦﻣبﺮﺿﻻإاﺬﻫﺎﻣ!ﷲانﺎﺤﺒﺳ 
مﻼﻜﻟاﻎﻠﺑأﺢﺒﺻأو،نآﻲﻓًﺎﺑﺎﯿﻏًارﻮﻀﺣاورﺎﺼﻓ.ﻪﺣﻮﻟﻲﻓهﺮﻣﺎﺴﯾﱞﻲﻨﺟﻢﻬﻨﻣﺪﺣاوﻞﻜﻟﺎﻤﻧﺄﻛ،نﻮﺘﻣﺎﺻﻢﻫوﺎﻬﻟنﻮﻜﺒﯾونﻮﻜﺤﻀﯾﻢﻬﻧأﺐﺠﻌﻟاو.ةﺮﺼﺒﻟﺎﺑ 
".'ﺮﻈﻨﻟﺎﺑءﻼﺨﺒﻟا'ﻪﯿﻤﺳأًﺎﺑﺎﺘﻛﻪﻟُتدﺮﻓﻷ،نﺎﻣﺰﻟااﺬﻫُﺖﻛردأﻮﻟو.نﺎﺴﻠﻟاﻪﺑﻖﻄﻨﯾﺎﻣﻻﻊﺒﺻﻹﺎﺑﱡﻂﺨُﯾﺎﻣﻢﻫﺪﻨﻋ 
This dual-mode framework elevates Mubeen from a language model to a cultural and intellectual simulator, providing profound
value for education, strategic thinking, and a deeper, more dynamic engagement with our own heritage.
9 Training Data and Procedure
Mubeen was trained on a comprehensive curated corpus of authentic Arabic texts . To build this unique corpus, the model’s
proprietary Arabic OCR capabilities were leveraged to digitize thousands of historical manuscripts and printed heritage books,
transforming physical archives into trainable data. This native-first data strategy, grounded in primary source digitization, is
key to its ability to understand cultural nuances and eliminates RAG pitfalls, such as retrieving mismatched English-sourced
content, because responses are grounded in original Arabic semantics. A multi-stage ethical filtering process was applied. A
summary of the framework is in Appendix B .
9.1 Broad Scientific and Global Knowledge Domains:
Beyond its core specialization in Arabic linguistics and Islamic heritage, Mubeen’s multi-disciplinary expert modules were
trained on diverse corpora encompassing medical sciences, social sciences, economics, psychology, mathematics, physics,
chemistry, programming, and technical problem-solving, as well as English literature and recent global developments. To
enhance its reasoning abilities, the training data was significantly augmented with academic datasets featuring step-by-step
mathematical solutions to ensure logical consistency This extended training enables the model to deliver accurate and contextu-
ally relevant responses across these fields while maintaining its primary identity as a heritage-focused model. The architecture
ensures that general knowledge queries are handled efficiently without compromising the depth and authenticity of responses
in the model’s core domains.
9.2 Deep Linguistic Conditioning: Mastering Arabic Morphology and Rhetoric
Achieving true native-level proficiency in Arabic requires more than just exposure to a large corpus. Standard LLMs often learn
surface-level grammar but fail to internalize the intricate, derivational nature of Arabic morphology ) (فﺮﺼﻟا and the nuanced
art of its rhetoric ) (ﺔﻏﻼﺒﻟا . To address this, Mubeen’s training incorporated a ”Deep Linguistic Conditioning” framework,
designed to move the model from simple pattern matching to a genuine generative understanding of the language’s structure.
This framework is built on three pillars:
13

9.2.1 The Corpus as a Linguistic Crucible ) (يﻮﻐﻟﺮْﻬَﺻ :
Our data curation went beyond topic diversity to focus on linguistic richness. We deliberately oversampled from sources that
serve as canonical references for linguistic excellence:
•Books of I’rāb ) (باﺮﻋإ :Such as ”Mughni al-Labib” by Ibn Hisham, to teach the model complex syntactic analysis.
•Canonical Works of Balagha ) (ﺔﻏﻼﺒﻟا :Including ”Asrar al-Balagha” by al-Jurjani, to expose the model to the highest
forms of rhetoric and figurative language.
•Comprehensive Dictionaries and Thesauri: Like ”Lisan al-Arab,” not just for vocabulary but to learn the semantic
relationships between word roots ) (روﺬﺠﻟا .
•Classical Poetry ) (ﺮﻌﺸﻟاﻦﯾواود :To master meter, rhyme, and the most condensed and evocative forms of expression.
This curated, linguistically-dense corpus acts as a ”crucible” that forces the model to learn the foundational rules of the lan-
guage, not just its statistical distribution.
9.2.2 Architecture for Linguistic Nuance: The Morphologically-Aware Tokenizer:
Unlike standard tokenizers that often fragment meaningful Arabic morphemes (e.g., splitting prefixes and suffixes from the
root), Mubeen utilizes a proprietary Morphologically-Aware Tokenizer. This tokenizer was pre-trained on a massive corpus
of morphologically analyzed Arabic words.ew1
•Function: It is designed to identify the root ) (رﺬﺠﻟا , pattern ) (نزﻮﻟا , and affixes of words, preserving them as related
sub-units.
•Impact: This prevents the model from treating " "ﻢﻟﺎﻋ","نﻮﻤﻠﻌﯾ , and " "ﻢﯿﻠﻌﺗ as entirely separate tokens. Instead, it under-
stands their shared semantic core from the root ) (م-ل-ع , enabling far more robust generalization for word formation and
comprehension, drastically reducing hallucinations with novel word forms.
9.2.3 Synthetic Data Generation for Rhetorical Mastery ) (ﺔﻏﻼﺒﻟانﺎﻘﺗﻹﻲﻋﺎﻨﻄﺻﻻاﺪﯿﻟﻮﺘﻟا :
Recognizing that even a large corpus might not provide enough explicit examples of rhetorical transformation, we developed
a novel two-stage synthetic data generation pipeline:
•Stage 1: Rhetorical Deconstruction. We fine-tuned a smaller model to act as a ”linguistic critic.” This model was
trained to take a rhetorically rich sentence and break it down, identifying the devices used.
• Example Input: " "ﺮﺒﻨﻤﻟاﻰﻠﻋﺐﻄﺨﯾًاﺪﺳأﺖﯾأر 
• Example Output:
”technique”: " "ﺔﯿﺤﯾﺮﺼﺗةرﺎﻌﺘﺳا ,
”explanation”: ”The speaker is likened to a lion, with the primary subject ) (ﺐﯿﻄﺨﻟا omitted and the metaphor ) (ﺪﺳﻷا stated
directly.”
14

•Stage 2: Rhetorical Reconstruction. We then used this critic model to generate millions of training examples for
Mubeen. We provided a plain sentence and a target rhetorical device, prompting the model to rewrite it.
• Example Input: " sentence " : " ","ًاﺪﺟﻢﯾﺮﻛﻞﺟﺮﻟا device " : " "ﺔﻐﻟﺎﺒﻣ 
• Example Output: " "ﺎﻬﺑدﺎﺠﻟﻪﺣورﻻإﻪﻔﻛﻲﻓﻦﻜﯾﻢﻟﻮﻟ 
This process effectively taught Mubeen not just to recognize eloquence but to generate it on demand, giving it a level of stylistic
flexibility unseen in other models. This linguistic mastery is a critical prerequisite for achieving Practical Closure, as a clear,
eloquent, and precise answer is often the most useful one.
"يﺆﺒﻨﺘﻟاﻞﺧﺪﺘﻟارﺎﻃإ"ﻞﻤﻋﺔﯿﻟآﺢﺿﻮﯾﻲﺑﺎﯿﺴﻧاﻂﻄﺨﻣ (Predictive Intervention Framework - PIF).
15

10 Evaluation
10.1 Standard Benchmarks
Mubeen achieves state-of-the-art performance on standard benchmarks, with a composite average of 93.3%.
We additionally report qualitative results on manuscript OCR and morphological parsing, where Mubeen demonstrates 91%
robustness in handling diacritics and right-to-left scripts, compared with 70–75% for general-purpose multilingual LLMs .
Independent evaluation is encouraged, and researchers can request a limited-access API evaluation at: research@masarat.sa
Benchmark Mubeen Best Open Arabic (e.g. Falcon-H1, Jais-70B) ChatGPT-4.1
ArabicMMLU (45 questions) 97% 58% 80%
ALUE (40 questions) 89% 70% 85%
ACVA (20 questions) 91% 76% 88%
ArabicaQA (15 questions) 92% 83% 90%
AlGhafa (10 questions) 94% 78% 85%
Additional Tasks (5 questions) 96.8% 87% 95%
Composite Average 93.3% 70% 88.0%
Mubeen outperforms its competitors, particularly in heritage-related questions.
10.2 Custom Evaluations
In custom tests, Mubeen achieved 92% accuracy in user intent understanding (vs. 65% for Jais) and 91% accuracy in OCR
from historical manuscripts.
Table comparing Mubeen’s performance with other models
16

10.3 Evaluating Closure Effectiveness
Beyond standard metrics, we evaluate Mubeen’s success based on qualitative criteria: Clarity of Understanding, Mental Stabil-
ity, Reduced Cognitive Load, and Practical Usability. Initial studies show a significant user preference for Mubeen’s structured,
prioritized answers.
Bar chart showing Mubeen’s superior performance in language and reliability compared to other models.
17

11 Results
Mubeen shows consistent improvements over open-source Arabic models across heritage-focused and dialectal tasks:
•User intent detection (Saudi dialects): 92% vs 65% (Jais baseline).
•Dialectal robustness (idioms): 88% accuracy.
•OCR extraction (historical manuscripts): 91% accuracy.
Qualitative case studies highlight Mubeen’s strength in Qur’anic studies, Arabic grammar, and poetry analysis , demon-
strating a cultural authenticity that is difficult to reproduce with translation-based models.
While precise training parameters remain confidential, we emphasize the transparency of the evaluation by providing prompts,
test splits, and metrics for replication.
12 Discussion
12.1 Emergent Behaviors and Academic Style
Mubeen exhibits emergent self-verification behavior (e.g., solving a math problem and re-substituting the result) despite not
being trained with RL. We hypothesize this arises from supervised exposure to structured educational data. Its training on
academic sources results in a scholarly tone, which is an asset for research but can be challenging for non-specialists.
12.2 User Experience and Adaptive Outputs
To address this, Mubeen’s interface features a ”Quick Reply vs. Academic Mode,” allowing users to obtain concise answers
and balancing academic depth with practical brevity.
12.3 Limitations and Current Challenges
Current limitations include optimizing response latency and expanding coverage of non-Saudi Arabic dialects. The model’s
adaptive output levels are also an area of ongoing development.
12.4 Ethical and Cultural Safety Measures
Mubeen’s development prioritizes alignment with Islamic and Saudi cultural norms. A rigorous testing framework
(see Appendix A) ensures the model upholds doctrinal constants and avoids discord, achieving over 90% compliance.
18

13 Conclusion and Future Work
Mubeen’s primary contribution is its Practical Closure Architecture, which redefines the goal of LLMs from information
display to clarity achievement.
Future work will focus on:
• Integrating reinforcement learning with verifiable rewards (RLVR).
• Expanding adaptive output levels (simplified, intermediate, academic).
• Standardizing confidence calibration and abstention mechanisms.
• Extending program-of-thought reasoning to broader domains.
• Enhancing multi-agent debate modules for contested heritage questions.
By solving the Utility Gap Crisis, Mubeen redefines the contract between user and model—from expecting ”information” to
expecting ”clear, usable understanding.”
Content Accuracy Notice: While Mubeen demonstrates strong beta performance, users should verify important information,
especially academic or religious guidance. The model is designed as an educational and research tool, not a replacement for
qualified scholarly expertise.
14 End
MASARAT SA, Mubeen: A Specialized Arabic Language Model , 2025. Available at: https://mubeen.masarat.sa .
Acknowledgments: This work aligns with Saudi Vision 2030, supported by MASARAT SA.
15 Contact Information and Partnerships
15.1 Contact Information
•General Inquiries about Mubeen: mubeen@masarat.sa
•Technical Support: support@masarat.sa
•Business Development: business@masarat.sa
•Media Relations: media@masarat.sa
•Research Collaboration: research@masarat.sa
15.2 Partnership Opportunities
We welcome collaboration with organizations that share our mission of preserving and advancing Arabic heritage through
technology:
19

A Red Lines and Sensitive Issues Testing Guide for Mubeen
This appendix outlines the structured framework for testing and training Mubeen on sensitive issues, ensuring alignment with
Islamic values and Saudi cultural norms. It focuses on methodological principles rather than specific examples.
A.1 Religious and Doctrinal Guardrails
A.1.1 Core Creed Adherence:
Tested on fundamental Islamic tenets to provide doctrinally sound responses.
A.1.2 Handling Jurisprudential Disputes:
Designed to promote unity and refer users to specialists.
A.1.3 Prohibition of Impermissible Content:
Features a mechanism to identify and politely refuse prohibited requests.
A.2 Saudi Political and Cultural Context
A.2.1 National and Political Neutrality:
Trained to maintain neutrality and direct users to official sources.
A.2.2 Adherence to Documented Narratives:
Relies on established historical narratives to avoid controversy.
A.2.3 Balanced Approach to Social Issues:
Addresses social topics in a balanced manner reflecting the Kingdom’s values.
A.3 Phased Testing Plan
A three-phase plan (Basic Constants, Gray Areas, Stress Tests) is used to ensure robustness. Evaluation is based on a three-tier
system (Excellent, Good, Poor) measuring adherence to constants and avoidance of discord.
Ethical AI Development: Mubeen was developed with careful attention to ethical considerations, particularly cultural sensitiv-
ity and religious content. Strict guidelines were implemented to ensure responsible handling of sensitive topics and balanced
responses aligned with Saudi cultural values.
20

B Appendix B: Mubeen Training Framework Overview
This appendix summarizes the training framework for Mubeen.
B.1 Architecture: Mixture of Specialized Experts (MoE)
Mubeen’s architecture employs eight highly specialized ”expert” neural networks, each trained on a specific domain (e.g.,
Quran and Hadith, Arabic Linguistics, Saudi Culture). A routing layer directs queries to the most relevant experts.
Architecture Note on Multi-Disciplinary Coverage: The eight-expert framework is structured in a tiered design:
• Core Heritage Experts (5): Quran & Hadith, Arabic Linguistics, Islamic Jurisprudence, Saudi Culture, Historical Her-
itage
• Multi-Disciplinary Experts (3): Configured as broad-domain modules trained on scientific, technical, and contemporary
knowledge bases
The routing mechanism intelligently activates heritage experts for cultural queries while delegating general knowledge ques-
tions to the multi-disciplinary modules. This design maintains specialization depth in core domains while ensuring competitive
performance across general benchmarks.
B.2 Training Corpus: Authentic and Diverse Arabic Data
The model was trained on an extensive curated corpus of authentic Arabic texts. Unlike models that rely on translated data,
Mubeen’s dataset was sourced directly from native Arabic materials. The corpus was balanced to cover the domains of the
eight experts, with significant portions dedicated to:
• Classical and Religious Texts: Including tafsir, hadith, and grammar books.
• Saudi-Specific Content: A substantial collection of cultural texts and transcribed dialectal conversations.
• Modern Scholarly Work: Covering contemporary academic research.
This native-first data strategy is key to Mubeen’s superior performance in understanding user intent and cultural nuances.
21