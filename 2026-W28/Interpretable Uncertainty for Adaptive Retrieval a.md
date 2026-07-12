# Interpretable Uncertainty for Adaptive Retrieval and Reasoning in Question Answering

**Authors**: Ritajit Dey, Iadh Ounis, Graham McDonald

**Published**: 2026-07-08 13:12:39

**PDF URL**: [https://arxiv.org/pdf/2607.07380v1](https://arxiv.org/pdf/2607.07380v1)

## Abstract
Large language models (LLMs) achieve a strong performance in question answering (QA), but remain prone to hallucinations and suffer from limited transparency. Retrieval-augmented generation (RAG) can improve factuality, yet decisions about when and how to retrieve from external resources are typically based on opaque policies or computationally inefficient multi-step prompting procedures. We propose an uncertainty-aware framework for adaptive QA based on explicit signals derived from LLM internal representations. We distinguish between knowledge insufficiency and knowledge ambiguity or conflict, and efficiently estimate these from hidden states in a single forward pass. These signals guide system behaviour: RAG is triggered when knowledge is insufficient, while additional reasoning is applied when ambiguity or conflict is high. By grounding adaptive decisions in decomposed and efficiently estimable uncertainty signals, this approach provides a transparent and practical alternative to existing retrieval and reasoning strategies supporting the design of interpretable user-facing tools.

## Full Text


<!-- PDF content starts -->

Interpretable Uncertainty for Adaptive Retrieval and Reasoning
in Question Answering
Ritajit Dey
r.dey.1@research.gla.ac.uk
University of Glasgow
Glasgow, UKIadh Ounis
iadh.ounis@glasgow.ac.uk
University of Glasgow
Glasgow, UKGraham McDonald
graham.mcdonald@glasgow.ac.uk
University of Glasgow
Glasgow, UK
Abstract
Large language models (LLMs) achieve a strong performance in
question answering (QA), but remain prone to hallucinations and
suffer from limited transparency. Retrieval-augmented generation
(RAG) can improve factuality, yet decisions about when and how to
retrieve from external resources are typically based on opaque poli-
cies or computationally inefficient multi-step prompting procedures.
We propose an uncertainty-aware framework for adaptive QA based
on explicit signals derived from LLM internal representations. We
distinguish between knowledge insufficiency and knowledge ambi-
guity or conflict, and efficiently estimate these from hidden states in
a single forward pass. These signals guide system behaviour: RAG
is triggered when knowledge is insufficient, while additional rea-
soning is applied when ambiguity or conflict is high. By grounding
adaptive decisions in decomposed and efficiently estimable uncer-
tainty signals, this approach provides a transparent and practical
alternative to existing retrieval and reasoning strategies supporting
the design of interpretable user-facing tools.
CCS Concepts
•Information systems →Information retrieval;Question
answering;•Computing methodologies →Language models.
Keywords
Question Answering, Retrieval-Augmented Generation, Informa-
tion Retrieval, Uncertainty Estimation, Large Language Models,
Interpretability
1 Introduction
Question answering (QA) is a central task in information retrieval
and natural language processing, requiring systems to produce ac-
curate answers to natural language queries. Recent progress has
been driven by large language models (LLMs), which generate an-
swers directly by leveraging knowledge acquired during large-scale
pretraining [ 18]. Despite strong empirical performance, such mod-
els are prone to hallucinated outputs [ 8], can rely on incomplete
or outdated internal knowledge [ 13], and suffer from limited trans-
parency in how answers are produced [2, 24].
Retrieval-augmented generation (RAG) [ 13] has emerged as a
dominant paradigm to address these issues by grounding gener-
ation in externally retrieved evidence. By decoupling knowledge
access from answer generation, RAG-based systems can improve
factuality and provide a more interpretable link between outputs
and supporting information. However, retrieval is not universally
beneficial: for queries where relevant knowledge is already well-
represented in the LLM, retrieval may introduce noise, increaselatency, and degrade answer quality [ 14]. This has motivated grow-
ing interest in adaptive retrieval strategies that determine whether
and how to retrieve external evidence [1, 7, 25].
Existing approaches to adaptive retrieval typically rely on im-
plicit learned decision policies [ 1,7,25], or structured multi-step
reasoning-based decision mechanisms [ 17]. The former are opaque
and do not disentangle different sources of uncertainty, while the
latter rely on multi-step prompting and are computationally in-
efficient. As a consequence, it is difficult to expose interpretable,
user-facing cues to end-users. This highlights the need for deci-
sion mechanisms that are both interpretable and efficient, while
explicitly modelling different sources of uncertainty. In QA set-
tings, uncertainty may arise due to knowledge insufficiency, where
the model lacks the required information, or due to knowledge
ambiguity or conflict, where multiple plausible interpretations or
conflicting evidence exist [ 6,11]. To illustrate knowledge insuf-
ficiency, consider the question “Who is the head coach of Forest
Green Rovers?”. Such long-tail facts about obscure teams may not be
reliably represented in the model’s training data; as a result, reason-
ing alone is unlikely to recover the correct answer (Robbie Savage),
and external retrieval is required. By contrast, knowledge conflicts
can be illustrated by the following question “Who is the head coach
of Chelsea?”. The training data may contain multiple historically
correct answers (e.g., Thomas Tuchel, Graham Potter, Mauricio
Pochettino, Enzo Maresca), each strongly associated with Chelsea
Football Club. This results in competing internal representations,
even though the data includes the currently correct answer (Enzo
Maresca) [ 9]. Increasing test-time compute [ 19] (i.e., enabling the
LLM to reason via Chain-of-Thought [ 23] or Self-Consistency [ 22])
can improve accuracy in such cases by allowing the LLM to se-
lect among competing candidate answers, as the reasoning process
encourages the enumeration of relevant facts followed by the appli-
cation of explicit selection criteria such as temporal recency. Hence,
the aforementioned two cases call for different responses: knowl-
edge insufficiency may benefit from external retrieval to fill gaps
in the LLM’s knowledge, while ambiguity or conflict may require
additional reasoning or increased test-time computation to select
correctly from competing candidate answers.
In this work, we propose an uncertainty-aware framework for
QA that explicitly distinguishes between knowledge insufficiency
and knowledge ambiguity or conflict, and uses these signals to guide
and explain retrieval and reasoning decisions. Our key idea is that
hidden states of LLMs can be used to efficiently estimate these two
forms of uncertainty within a single forward pass, avoiding the need
for repeated sampling or auxiliary models. Based on these estimates,
we introduce a simple decision mechanism: retrieval is triggered
when knowledge insufficiency is high, while additional reasoning
arXiv:2607.07380v1  [cs.IR]  8 Jul 2026

WExIR ’26, July 24, 2026, Melbourne, Australia Dey et al.
Figure 1: The proposed uncertainty-aware QA framework.
methods (such as Chain-of-Thought [ 23] or Self-Consistency [ 22])
are applied when ambiguity or conflict is high. An illustration of
the proposed QA pipeline is given in Figure 1.
The ultimate objective is to provide a transparent and principled
basis for adaptive retrieval-augmented generation in knowledge-
intensive tasks, grounding routing and intervention decisions in in-
terpretable uncertainty signals, while remaining efficient for multi-
step or agentic settings where repeated sampling or prompting is
impractical. These signals allow a QA system, regardless of its com-
plexity, to surface the nature of the underlying knowledge problem
at each step (e.g., missing vs. conflicting or ambiguous knowledge),
and to use this information to drive behaviour as well as generate
step-level explanations for why specific components (e.g., retrieval,
disambiguation, clarification, abstention) are invoked. By design,
this enables decision behaviour that is directly interpretable, and
where needed, explainable, rather than inferred post-hoc.
2 Methodology & Results
We propose an uncertainty-aware framework for adaptive QA that
estimates two complementary signals from LLM hidden states:
knowledge insufficiency and knowledge ambiguity or conflict, and
uses them to guide retrieval and reasoning decisions.
Uncertainty Estimation from Hidden States:We propose
estimating knowledge insufficiency and knowledge ambiguity or
conflict for a given question directly from LLM hidden states using
regression probes. We train lightweight regression probes over
hidden states [3–5] to estimate two quantities:
Knowledge insufficiencyis estimated as the number of occur-
rences of the fact in the pretraining corpus of the LLM, following
Kandpal et al. [ 9] and Kang et al. [ 10], with fewer occurrences
indicating higher insufficiency and uncertainty [ 8]. We propose
training a regression probe to predict these occurrence counts from
LLM hidden states.
Knowledge ambiguity/conflictcaptures the degree to which
multiple competing answers are supported by the data. For a given
question, we conceptualise this quantity as the entropy of the oc-
currences of the different versions of the fact in the pretraining
corpus, following Tomov et al. [ 20]. Higher entropy corresponds to
increased LLM uncertainty [ 20]. We propose training a regression
probe to estimate this entropy value from LLM hidden states.Both probes operate on hidden states from a single forward
pass, enabling efficient uncertainty estimation without additional
sampling or prompting.
Adaptive Decision Mechanism:We propose using estimated
uncertainty signals to control the behaviour of the QA system
through a simple threshold-based decision mechanism. When a
question is passed to the LLM, the hidden states of the LLM can
be extracted and used to estimate knowledge insufficiency and
ambiguity or conflict scores, which in turn inform downstream
processing decisions. If knowledge insufficiency exceeds a given
threshold, the system can trigger retrieval-augmented generation
(RAG) to supplement missing information. On the other hand, when
ambiguity or conflict is high, the system may instead allocate ad-
ditional test-time computation [ 19] (e.g., Chain-of-Thought [ 23]
or Self-Consistency [22]) to resolve competing candidate answers.
When both signals are low, the model can answer directly without
additional intervention.
Preliminary Results:We conduct experiments on factoid QA
using the NQ dataset [ 12] with Llama-2-7b-chat [ 21], using Se-
mantic Entropy (SE) [ 11] and Weighted Entropy Production Rate
(WEPR) [ 16] to selectively trigger RAG. Results show that our frame-
work significantly outperforms (according to McNemar’s test) both
the LLM-only baseline (+5.9% for SE, +4.7% for WEPR) and the
always-on RAG baseline (+3.3% for SE, +2.1% for WEPR). This sup-
ports the promise of uncertainty-driven adaptive QA.
3 Conclusions
We advocated a shift in adaptive question answering (QA) from
opaque policies toward decisions grounded in explicit, interpretable
uncertainty. By distinguishing between knowledge insufficiency
and knowledge ambiguity or conflict, we outlined a framework
where different forms of uncertainty drive different actions – re-
trieval to acquire missing information, and additional reasoning to
resolve competing alternatives. This decomposition enables inter-
pretable decision-making by exposing the type of knowledge prob-
lem the system is facing and justifying its behaviour accordingly,
thereby providing both users and system designers with step-level
explanations, as needed. We further argued that interpretability
should not come at the cost of efficiency. Our approach highlights
the potential to estimate these signals within a single forward pass,
avoiding expensive sampling or auxiliary components, which is
critical for deployment in complex settings. More broadly, we advo-
cated for treating interpretability as a design principle rather than
a post-hoc tool in QA systems. This allows users to understand
why the system searched or reasoned further, and enables system
designers to audit and diagnose behaviour in terms of missing or
conflicting knowledge.
References
[1]Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2024.
Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.
InThe Twelfth International Conference on Learning Representations, ICLR 2024,
Vienna, Austria, May 7-11, 2024.
[2]Emily M. Bender, Timnit Gebru, Angelina McMillan-Major, and Shmargaret
Shmitchell. 2021. On the Dangers of Stochastic Parrots: Can Language Models Be
Too Big?. InProceedings of the 2021 ACM Conference on Fairness, Accountability,
and Transparency, FAccT 2021, Toronto, Canada, March 3-10, 2021.
[3]Alexis Conneau, Douwe Kiela, Holger Schwenk, Loïc Barrault, and Antoine
Bordes. 2018. What You Can Cram into a Single Vector: Probing Sentence

Interpretable Uncertainty for Adaptive Retrieval and Reasoning in Question Answering WExIR ’26, July 24, 2026, Melbourne, Australia
Embeddings for Linguistic Properties. InProceedings of the 56th Annual Meeting
of the Association for Computational Linguistics, ACL 2018, Melbourne, Australia,
July 15-20, 2018, Volume 1: Long Papers.
[4]John Hewitt and Christopher D. Manning. 2019. A Structural Probe for Finding
Syntax in Word Representations. InProceedings of the 2019 Conference of the
North American Chapter of the Association for Computational Linguistics: Human
Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019,
Volume 1 (Long and Short Papers).
[5]Nora Hollenstein, Federico Pirovano, Ce Zhang, Lena Jäger, and Lisa Beinborn.
2021. Multilingual Language Models Predict Human Reading Behavior. InPro-
ceedings of the 2021 Conference of the North American Chapter of the Association
for Computational Linguistics: Human Language Technologies, NAACL-HLT 2021,
Online, June 6-11, 2021.
[6]Eyke Hüllermeier and Willem Waegeman. 2021. Aleatoric and Epistemic Uncer-
tainty in Machine Learning: An Introduction to Concepts and Methods.Machine
Learning110, 3 (2021), 457–506.
[7]Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong Park.
2024. Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language
Models. InProceedings of the 2024 Conference of the North American Chapter of the
Association for Computational Linguistics: Human Language Technologies (Volume
1: Long Papers), NAACL 2024, Mexico City, Mexico, June 16-21, 2024.
[8]Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii,
Yejin Bang, Andrea Madotto, and Pascale Fung. 2023. Survey of Hallucination in
Natural Language Generation.ACM Comput. Surv.55, 12, Article 248 (2023).
[9]Nikhil Kandpal, Haikang Deng, Adam Roberts, Eric Wallace, and Colin Raffel.
2023. Large language models struggle to learn long-tail knowledge. InInterna-
tional Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu,
Hawaii, USA.
[10] Cheongwoong Kang and Jaesik Choi. 2023. Impact of Co-occurrence on Fac-
tual Knowledge of Large Language Models. InFindings of the Association for
Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023.
[11] Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. 2023. Semantic Uncertainty:
Linguistic Invariances for Uncertainty Estimation in Natural Language Genera-
tion. InThe Eleventh International Conference on Learning Representations, ICLR
2023, Kigali, Rwanda, May 1-5, 2023.
[12] Tom Kwiatkowski et al .2019. Natural Questions: A Benchmark for Question
Answering Research.Transactions of the Association for Computational Linguistics
7 (2019), 452–466.
[13] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented Generation for
Knowledge-Intensive NLP Tasks. InAdvances in Neural Information Processing
Systems 33: Annual Conference on Neural Information Processing Systems 2020,
NeurIPS 2020, December 6-12, 2020, virtual.
[14] Alex Mallen, Akari Asai, Zexuan Zhong, et al .2023. When Not to Trust Lan-
guage Models: Investigating Effectiveness of Retrieval-Augmented Generation.
InProceedings of the 61st Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), ACL 2023.
[15] Sara Vera Marjanovic, Haeun Yu, Pepa Atanasova, Maria Maistro, Christina
Lioma, and Isabelle Augenstein. 2024. DYNAMICQA: Tracing Internal Knowledge
Conflicts in Language Models. InFindings of the Association for Computational
Linguistics: EMNLP 2024, Miami, Florida, USA, November 12-16, 2024.
[16] Charles Moslonka, Hicham Randrianarivo, Arthur Garnier, and Emmanuel Mal-
herbe. 2026. Learned Hallucination Detection in Black-Box Llms Using Token-
Level Entropy Production Rate. InAdvances in Information Retrieval - 48th Eu-
ropean Conference on Information Retrieval, ECIR 2026, Delft, The Netherlands,
March 29 - April 2, 2026, Proceedings, Part I. Springer Nature Switzerland.
[17] Vy Nguyen, Ziqi Xu, Jeffrey Chan, Estrid He, Feng Xia, and Xiuzhen Zhang.
2026. Hallucinate Less by Thinking More: Aspect-Based Causal Abstention for
Large Language Models. InFortieth AAAI Conference on Artificial Intelligence,
Thirty-Eighth Conference on Innovative Applications of Artificial Intelligence, Six-
teenth Symposium on Educational Advances in Artificial Intelligence, AAAI 2026,
Singapore, January 20-27, 2026.
[18] Fabio Petroni, Tim Rocktäschel, Sebastian Riedel, Patrick Lewis, Anton Bakhtin,
Yuxiang Wu, and Alexander Miller. 2019. Language Models as Knowledge Bases?.
InProceedings of the 2019 Conference on Empirical Methods in Natural Language
Processing and the 9th International Joint Conference on Natural Language Pro-
cessing, EMNLP-IJCNLP 2019, Hong Kong, China, November 3-7, 2019.
[19] Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. 2024. Scaling LLM Test-
Time Compute Optimally can be More Effective than Scaling Model Parameters.
arXiv:2408.03314
[20] Tim Tomov, Dominik Fuchsgruber, Tom Wollschläger, and Stephan Günnemann.
2026. The Illusion of Certainty: Uncertainty Quantification for LLMs Fails Under
Ambiguity. arXiv:2511.04418
[21] Hugo Touvron et al .2023. Llama 2: Open Foundation and Fine-Tuned Chat
Models. arXiv:2307.09288
[22] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang,
Aakanksha Chowdhery, and Denny Zhou. 2023. Self-Consistency ImprovesChain of Thought Reasoning in Language Models. InThe Eleventh International
Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023.
[23] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei
Xia, Ed Chi, Quoc Le, and Denny Zhou. 2022. Chain-of-Thought Prompting
Elicits Reasoning in Large Language Models. InAdvances in Neural Information
Processing Systems 35: Annual Conference on Neural Information Processing Systems
2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022.
[24] Sarah Wiegreffe and Yuval Pinter. 2019. Attention is not not Explanation. In
Proceedings of the 2019 Conference on Empirical Methods in Natural Language Pro-
cessing and the 9th International Joint Conference on Natural Language Processing,
EMNLP-IJCNLP 2019, Hong Kong, China, November 3-7, 2019.
[25] Di Wu et al .2024. REPOFORMER: selective retrieval for repository-level code
completion. InForty-first International Conference on Machine Learning, ICML
2024, Vienna, Austria, July 21-27, 2024.