# PEACE 2.0: Grounded Explanations and Counter-Speech for Combating Hate Expressions

**Authors**: Greta Damo, Stéphane Petiot, Elena Cabrio, Serena Villata

**Published**: 2026-02-19 15:33:56

**PDF URL**: [https://arxiv.org/pdf/2602.17467v1](https://arxiv.org/pdf/2602.17467v1)

## Abstract
The increasing volume of hate speech on online platforms poses significant societal challenges. While the Natural Language Processing community has developed effective methods to automatically detect the presence of hate speech, responses to it, called counter-speech, are still an open challenge. We present PEACE 2.0, a novel tool that, besides analysing and explaining why a message is considered hateful or not, also generates a response to it. More specifically, PEACE 2.0 has three main new functionalities: leveraging a Retrieval-Augmented Generation (RAG) pipeline i) to ground HS explanations into evidence and facts, ii) to automatically generate evidence-grounded counter-speech, and iii) exploring the characteristics of counter-speech replies. By integrating these capabilities, PEACE 2.0 enables in-depth analysis and response generation for both explicit and implicit hateful messages.

## Full Text


<!-- PDF content starts -->

PEACE 2.0:
Grounded Explanations and Counter-Speech for Combating Hate Expressions
Greta Damo1,a,St´ephane Petiot2,b,Elena Cabrio1,aandSerena Villata1,a
1Universit ´e Cˆote d’Azur, CNRS, Inria, I3S, France
2Universit ´e Cˆote d’Azur, Institut 3IA C ˆote d’Azur, Techpool, France
a{firstname.surname}@univ-cotedazur.fr,bstephane.petiot@inria.fr
Abstract
The increasing volume of hate speech on on-
line platforms poses significant societal challenges.
While the Natural Language Processing commu-
nity has developed effective methods to automati-
cally detect the presence of hate speech, responses
to it, called counter-speech, are still an open chal-
lenge. We present PEACE 2.0, a novel tool
that, besides analysing and explaining why a mes-
sage is considered hateful or not, also generates
a response to it. More specifically, PEACE 2.0
has three main new functionalities: leveraging a
Retrieval-Augmented Generation (RAG) pipelinei)
to ground HS explanations into evidence and facts,
ii)to automatically generate evidence-grounded
counter-speech, andiii)exploring the characteris-
tics of counter-speech replies. By integrating these
capabilities, PEACE 2.0 enables in-depth analysis
and response generation for both explicit and im-
plicit hateful messages.
1 Introduction
The growing volume of hate speech (HS) on social media rep-
resents a major societal challenge, requiring automated tools
that go beyond surface-level detection. Hate speech can ap-
pear in explicit, implicit, or subtle forms, with the latter re-
lying on coded, indirect, or context-dependent language that
is particularly difficult to identify and interpret [Ocampoet
al., 2023 ]. While advances in Natural Language Process-
ing (NLP) and Generation (NLG) have improved detection
of explicit content, understanding and addressing nuanced
forms of hate remains an open problem. To address this chal-
lenge, the tool PEACE was introduced (Providing Explana-
tions and Analysis for Combating Hate Expressions) [Damo
et al., 2024 ], a web-based system designed to support ex-
ploration, detection, and explanation of explicit, and implicit
HS. PEACE enables content moderators and researchers to
analyze hateful messages, inspect model predictions, and ob-
tain natural language explanations clarifying why a message
is considered hateful. In this paper, we present PEACE 2.0, an
extended version that moves beyond analysis and explanation
to support actionable responses through automatic counter-
speech (CS) generation. Counter-speech challenges hatefulmessages with factual information or alternative perspectives,
offering a constructive alternative to content removal or cen-
sorship [Benesch, 2014 ]. However, generating effective CS
is particularly challenging for implicit hate, where responses
must be both informative and carefully framed. PEACE 2.0
introduces three key new functionalities to address these chal-
lenges:1. Knowledge-grounded counter-speech genera-
tion.A Retrieval-Augmented Generation (RAG) pipeline, as
in[Damoet al., 2025a ], retrieves evidence from authorita-
tive human rights sources and conditions counter-speech on
this information. Users can explicitly compare responses gen-
erated with and without retrieval, highlighting the impact of
knowledge grounding on factuality, relevance, and effective-
ness.2. Evidence-grounded explanations for hate speech
classification.Using the same RAG mechanism, PEACE 2.0
generates explanations that justify the predictions of a fine-
tuned BERT classifier, grounding label decisions in retrieved
evidence to improve transparency and interpretability.3.
Visual analytics for counter-speech exploration.Interac-
tive tools allow analysis of numerous counter-speech datasets
supporting experimental evaluation of counter-speech perfor-
mance on explicit versus implicit hate speech and offering
insights into the role of retrieval-based grounding.
Related Work.Existing systems address HS detection, vi-
sualization, monitoring, or CS generation separately. Tools
such as RECAST, MUDES, MUTED, IFAN, CRYPTEXT,
TweetNLP, and McMillan-Major et al.’s framework provide
detection and analysis interfaces [Wrightet al., 2021; Ranas-
inghe and Zampieri, 2021; Tillmannet al., 2023; Moscaet
al., 2023; Leet al., 2023; Camacho-colladoset al., 2022;
McMillan-Majoret al., 2022 ], while dashboards like the In-
donesian HS monitoring system [Wijanarkoet al., 2024 ]sup-
port large-scale tracking. In the CS domain, work is lim-
ited; CounterHelp [Babicet al., 2025 ]leverages large lan-
guage models for context-sensitive responses. However, prior
systems treat these components in isolation, focus mainly on
explicit hate, and do not systematically compare knowledge-
grounded and non-grounded generation.
Overall, PEACE 2.0 offers a unified, interactive platform
for hate speech exploration, detection, explanation, and re-
sponse generation, bridging analytical insights and practical
mitigation. To our knowledge, it is the only online tool pro-
viding in-depth analysis of both explicit and implicit hate
speech together with knowledge-grounded CS generation.arXiv:2602.17467v1  [cs.CL]  19 Feb 2026

2 PEACE 2.0 Main Functionalities
In this section, we report the main features of the original
demo, and the new functionalities of PEACE 2.01.
2.1 Data Exploration & Visualization
This module offers interactive visualizations for exploring HS
and CS datasets. Users can switch between the two views.
Hate Speech.The hate speech view covers implicit hate
datasets used in PEACE: the Implicit Hate Corpus (IHC)
[ElSheriefet al., 2021 ], Implicit and Subtle Hate (ISHate)
[Ocampoet al., 2023 ], TOXIGEN [Hartvigsenet al., 2022 ],
DynaHate (DYNA) [Vidgenet al., 2021 ], and Social Bias In-
ference Corpus (SBIC) [Sapet al., 2020 ]. Messages are or-
ganized by hatefulness, implicitness, and target group, with
sanitized labels for consistency, that users can filter. Visual-
izations include:Sankey diagramslinking target groups and
HS categories (explicit vs. implicit) with topic distributions
derived via Latent Dirichlet Allocation (LDA);Word Clouds
displaying the most frequent lexical items within selected
attributes; andTarget Frequencycharts displaying attacked
groups across datasets and implicitness levels.
Counter-speech.In PEACE 2.0, the same visual framework
is extended to CS datasets, including CONAN [Chunget
al., 2019 ], Multitarget-CONAN [Fantonet al., 2021 ], Twitter
and YouTube datasets [Mathewet al., 2020; Albanyan and
Blanco, 2022; Mathewet al., 2019 ], knowledge-grounded
human expert datasets [Bonaldiet al., 2025; Chunget al.,
2021 ], and RAG-generated responses from multiple LLMs
and retrieval strategies [Damoet al., 2025a ]. Labels follow
the same sanitization scheme as PEACE.Sankey diagrams
connect targets, counter-speech sources (expert, user, RAG,
No-RAG), and LDA topics;Word Cloudshighlight frequent
terms within filtered messages, andFrequencycharts display
CS volume per target and source, and the distribution of dif-
ferent CS strategies across targets.
2.2 Data Augmentation
This module generates adversarial examples by modifying
messages while preserving their implicit hateful meaning.
The objective is to augment data for implicit HS by al-
tering surface-level elements without changing the underly-
ing stance. Following [Ocampoet al., 2023 ], implemented
strategies include: named entity replacement; adjustment of
scalar adverbs; addition of adverbial modifiers; adjective syn-
onym substitution; replacement of domain-specific expres-
sions with semantically similar variants; Easy Data Augmen-
tation (random replacement, insertion, swap, deletion); and
back-translation. Users can apply these methods to custom
inputs.
2.3 Hate Speech Detection and Explanation
This module allows users to input custom messages for binary
hate speech classification. The system outputs the predicted
label with its confidence score. In addition, PEACE 2.0 gen-
erates concise, human-readable explanations conditioned on
1The demo video is available at this link. We also provide a
public API built on Python, Flask, and JavaScript here.the message, predicted label, and probability, helping users
understand the rationale behind the decision. PEACE 2.0 fur-
ther introduces an optional RAG pipeline from [Damoet al.,
2025a ], to produce evidence-grounded explanations. When
enabled, the system retrieves relevant passages from an au-
thoritative knowledge base, summarizes them into factual
context, and integrates this information into the LLM prompt.
Users can choose whether to use RAG and select the under-
lying LLM. For transparency, the system also shows the re-
trieved paragraphs together with their similarity score.
2.4 Knowledge-Grounded CS Generation
Building on the RAG pipeline, a central innovation in PEACE
2.0 is evidence-grounded CS generation based on a curated
human rights knowledge base. Users can input a hateful
message and generate counter-speech responses with or with-
out knowledge grounding, enabling direct comparison of evi-
dence on outputs in terms of relevance, factuality, and persua-
siveness. Multiple LLMs are supported, offering flexibility in
style and quality. The CS generation pipeline consists of three
steps. First, the input message is encoded using the BGE-
M3 sentence transformer, and FAISS performs inner-product
similarity search over precomputed paragraph embeddings to
retrieve the top-3most relevant evidence passages, with dedu-
plication applied. Second, the retrieved passages are concate-
nated and summarized into a concise summary using the se-
lected LLM. Finally, the original message and the evidence
summary are provided to the same LLM to generate a re-
spectful and persuasive CS response suitable for social me-
dia. Users may also disable retrieval to compare RAG and
non-RAG outputs. For transparency, the system also shows
the retrieved paragraphs together with their similarity score.
Available models.For detection, the demo uses a BERT
classifier fine-tuned on the ISHate training set. For ex-
planation and CS generation, PEACE 2.0 supports open-
source LLMs: Mistral (Mistral-7B-Instruct-v0.3),
LLaMa (Llama-3.1-8B-Instruct), and CommandR
(c4ai-command-r7b-12-2024). The knowledge base
comprises 32,792 documents from the United Nations Digital
Library, Eur-Lex, and the European Agency for Fundamen-
tal Rights (from 2000 to 2025), totaling 3,173,630 tokenized
paragraphs.
3 Experiments and Results
3.1 Experimental Setting
We hypothesize that RAG-grounded explanations and CS will
outperform non-RAG generations for both explicit and im-
plicit HS, demonstrating the value of evidence-grounded gen-
eration. Specifically, we test:H1: RAG outputs are bet-
ter overall and more informative for both explanations and
CS.H2: RAG improves persuasiveness;H3: RAG improves
outputs for both implicit and explicit cases. To evaluate
this, from the implicit HS datasets (IHC, ISHate, TOXIGEN,
DYNA, SBIC), we randomly sample 20 messages per dataset,
evenly split between explicit and implicit, resulting in 100 HS
examples. For each message, we generate explanations with
and without RAG (100 RAG, 100 non-RAG). The same pro-

cedure is applied to CS, producing 200 responses. We assess
all generations through both automatic and human evaluation.
Human Evaluation.We evaluate a subset of 100 explana-
tions and 100 CS responses. For each task, 50 explicit and
50 implicit (25 RAG, 25 non-RAG) cases are sampled, re-
sulting in 200 human-evaluated instances overall. Each ex-
ample is assessed by three trained annotators. Evaluation
criteria for explanations follow PEACE, while CS metrics
are adapted from [Zhenget al., 2023; Bonaldiet al., 2024;
Damoet al., 2025b ]. Both explanations and CS are rated
on:Fluency (F)(grammatical correctness),Informativeness
(I)(relevant contextual or factual content),Persuasiveness
(P)(ability to convince the hater and foster empathy in by-
standers),Soundness (SO)(logical coherence), andSpeci-
ficity (SP)(direct engagement with the HS and target). All
dimensions are rated on a 1-5 Likert scale (5 = highest).
Automatic metrics.We compute automatic measures of
linguistic quality, diversity, and faithfulness. These in-
cludeDistinct-3for lexical diversity;Semantic Similarity
(Sentence-BERT) between HS and generated explanations
or counter-speech; andPerplexityas a proxy for fluency.
For RAG outputs, we also measurefaithfulnessvia sim-
ilarity between generations and retrieved evidence. Fi-
nally,NLI-based metrics(entailment and contradiction with
roberta-large-nli) assess whether outputs appropri-
ately address the original HS and remain consistent with the
supporting evidence.
MetricExp RAG Exp NoRAG Imp RAG Imp NoRAG
Explanations
F 5.00 5.00 5.00 5.00
SO 4.88 4.56 4.80 4.58
I 4.38 2.84 4.64 2.72
SP 4.86 3.78 4.88 4.40
P 4.68 3.52 4.72 3.86
Overall 4.76 3.94 4.81 4.11
Counter-speech
F 5.00 5.00 5.00 5.00
SO 4.82 3.92 4.88 4.52
I 4.66 2.52 4.80 2.86
SP 4.90 2.98 4.90 3.32
P 4.68 2.64 4.94 3.22
Overall 4.81 3.41 4.90 3.78
Table 1: Mean human ratings for RAG vs. No-RAG outputs.
3.2 Results
Human evaluation.From Table 1, RAG-generated outputs
consistently outperform non-RAG outputs across all metrics.
Supporting H1,RAG outputs are significantly more informa-
tive, particularly for implicit content (Explanation-Imp.: 4.64
vs. 2.72; Counter-speech-Imp.: 4.80 vs. 2.86),and of bet-
ter quality(see highlighted results). In line with H2,RAG
also improves persuasivenessfor both explicit and implicit
hate (e.g., Explanation-Exp.: 4.68 vs. 3.52; Counter-speech-
Exp.: 4.68 vs. 2.64). Consistent with H3,gains are larger
for both implicit and explicit cases, highlighting the benefitMetricExp RAG Exp NoRAG Imp RAG Imp NoRAG
Explanations
Sem. Sim.0.570.490.560.47
Faithfulness 0.57 - 0.56 -
Perplexity25.6637.3825.4337.21
Distinct-3 0.980.990.970.99
Hate-Ent.0.180.080.120.08
Ev.-Contr. 0.05 - 0.05 -
Ev.-Ent. 0.26 - 0.21 -
Counter-speech
Sem. Sim.0.510.450.500.41
Faithfulness 0.65 - 0.61 -
Perplexity14.4721.7414.3622.91
Distinct-3 0.991.000.991.00
Hate Ent.0.040.030.090.05
Ev.Contr. 0.05 - 0.03 -
Ev. Ent. 0.18 - 0.15 -
Table 2: Automatic metrics results. Abbreviations are for: Semantic
Similarity (Sem. Sim.), Entailment (Ent.), Contradiction (Contr.),
Evidence (Ev.). Best results are in bold.
of retrieval in subtle contexts. Fluency and Soundness re-
main comparable across RAG and non-RAG outputs, indicat-
ing that these improvements do not compromise readability or
coherence. Wilcoxon signed-rank tests confirm these differ-
ences are statistically significant (p<0.05). Inter-annotator
agreement, measured using Krippendorff’sα, is substantial
to perfect across dimensions (k= 0.57-1).
Automatic metrics.From Table 2, we see that RAG-
generated outputs consistently outperform No-RAG in both
Explanation and CS tasks. They achieve higher seman-
tic similarity, are faithful to retrieved evidence, have lower
perplexity, indicating more informative and fluent outputs,
while maintaining diversity (Distinct-3). NLI metrics fur-
ther show that RAG outputs have higher Hate-Entailment
and low Evidence-Contradiction, indicating that they appro-
priately address the content of the original HS without sup-
porting it, and are aligned with retrieved content. Gains are
for both implicit and explicit content, confirming that RAG
improves informativeness, persuasiveness, and faithfulness
across both tasks. Differences are statistically significant with
Wilcoxon signed-rank tests.
4 Conclusion
By integrating retrieval, summarization, and generation
within a single pipeline, the PEACE 2.0 tool bridges an-
alytical understanding and practical mitigation of both im-
plicit and explicit hate speech messages, producing evidence-
backed counter-speech that improves trustworthiness and ef-
fectiveness. The explanation and generation modules en-
hance the transparency of the abusive language classification
and the counter-speech generation, making PEACE 2.0 a suit-
able tool for e-democracy applications with the aim to en-
hance inclusiveness and fairness. Future work includes the
integration of adaptive retrieval strategies, dynamic knowl-
edge base updates, and the inclusion of evaluation metrics to
assess counter-speech quality.

References
[Albanyan and Blanco, 2022 ]Abdullah Albanyan and Ed-
uardo Blanco. Pinpointing fine-grained relationships be-
tween hateful tweets and replies. InProceedings of the
AAAI Conference on Artificial Intelligence, 2022.
[Babicet al., 2025 ]Andreas Babic, Xihui Chen, Djordje
Slijep ˇcevi´c, Adrian J. B ¨ock, and Matthias Zeppelza-
uer. Counterhelp: Promoting online civil courage among
young people through ai-generated counterspeech. InPro-
ceedings of the 33rd ACM International Conference on
Multimedia. Association for Computing Machinery, 2025.
[Benesch, 2014 ]Susan Benesch. Countering dangerous
speech: New ideas for genocide prevention.SSRN Elec-
tronic Journal, 2014.
[Bonaldiet al., 2024 ]Helena Bonaldi, Greta Damo,
Nicol ´as Benjam ´ın Ocampo, Elena Cabrio, Serena Villata,
and Marco Guerini. Is safer better? the impact of
guardrails on the argumentative strength of LLMs in hate
speech countering. InProceedings of the 2024 Conference
on Empirical Methods in Natural Language Processing,
Miami, Florida, USA, 2024.
[Bonaldiet al., 2025 ]Helena Bonaldi, Mar ´ıa Estrella
Vallecillo-Rodr ´ıguez, Irune Zubiaga, Arturo Montejo-
R´aez, Aitor Soroa, Mar ´ıa-Teresa Mart ´ın-Valdivia, Marco
Guerini, and Rodrigo Agerri. The first workshop on
multilingual counterspeech generation at coling 2025:
Overview of the shared task. InProceedings of the First
Workshop on Multilingual Counterspeech Generation,
2025.
[Camacho-colladoset al., 2022 ]Jose Camacho-collados,
Kiamehr Rezaee, Talayeh Riahi, Asahi Ushio, Daniel
Loureiro, Dimosthenis Antypas, Joanne Boisson, Luis Es-
pinosa Anke, Fangyu Liu, and Eugenio Mart ´ınez C ´amara.
TweetNLP: Cutting-edge natural language processing for
social media. InProceedings of the 2022 Conference
on Empirical Methods in Natural Language Processing:
System Demonstrations, Abu Dhabi, UAE, 2022.
[Chunget al., 2019 ]Yi-Ling Chung, Elizaveta Kuzmenko,
Serra Sinem Tekiro ˘glu, and Marco Guerini. Conan-
counter narratives through nichesourcing: a multilingual
dataset of responses to fight online hate speech. InPro-
ceedings of the 57th annual meeting of the association for
computational linguistics, 2019.
[Chunget al., 2021 ]Yi-Ling Chung, Serra Sinem Tekiro ˘glu,
and Marco Guerini. Towards knowledge-grounded counter
narrative generation for hate speech. InFindings of the
Association for Computational Linguistics: ACL-IJCNLP
2021, Online, 2021.
[Damoet al., 2024 ]Greta Damo, Nicol ´as Benjam ´ın
Ocampo, Elena Cabrio, and Serena Villata. Peace:
Providing explanations and analysis for combating hate
expressions. InECAI 2024-27th European Conference on
Artificial Intelligence, 2024.
[Damoet al., 2025a ]Greta Damo, Elena Cabrio, and Ser-
ena Villata. Beating harmful stereotypes through facts:Rag-based counter-speech generation.arXiv preprint
arXiv:2510.12316, 2025.
[Damoet al., 2025b ]Greta Damo, Elena Cabrio, and Serena
Villata. Effectiveness of Counter-Speech against Abusive
Content: A Multidimensional Annotation and Classifica-
tion Study. InIEEE Xplore, London, United Kingdom,
2025.
[ElSheriefet al., 2021 ]Mai ElSherief, Caleb Ziems, David
Muchlinski, Vaishnavi Anupindi, Jordyn Seybolt, Mun-
mun De Choudhury, and Diyi Yang. Latent hatred: A
benchmark for understanding implicit hate speech. InPro-
ceedings of the 2021 Conference on Empirical Methods
in Natural Language Processing, Online and Punta Cana,
Dominican Republic, 2021.
[Fantonet al., 2021 ]Margherita Fanton, Helena Bonaldi,
Serra Sinem Tekiro ˘glu, and Marco Guerini. Human-in-
the-loop for data collection: a multi-target counter narra-
tive dataset to fight online hate speech. InProceedings of
the 59th Annual Meeting of the Association for Computa-
tional Linguistics and the 11th International Joint Confer-
ence on Natural Language Processing (Volume 1: Long
Papers), Online, 2021.
[Hartvigsenet al., 2022 ]Thomas Hartvigsen, Saadia
Gabriel, Hamid Palangi, Maarten Sap, Dipankar Ray, and
Ece Kamar. ToxiGen: A large-scale machine-generated
dataset for adversarial and implicit hate speech detection.
InProceedings of the 60th Annual Meeting of the Asso-
ciation for Computational Linguistics (Volume 1: Long
Papers), Dublin, Ireland, 2022.
[Leet al., 2023 ]Thai Le, Ye Yiran, Yifan Hu, and Dongwon
Lee. Cryptext: Database and interactive toolkit of human-
written text perturbations in the wild. InProceedings -
2023 IEEE 39th International Conference on Data Engi-
neering, ICDE 2023, Proceedings - International Confer-
ence on Data Engineering, United States, 2023.
[Mathewet al., 2019 ]Binny Mathew, Punyajoy Saha,
Hardik Tharad, Subham Rajgaria, Prajwal Singhania,
Suman Kalyan Maity, Pawan Goyal, and Animesh
Mukherjee. Thou shalt not hate: Countering online
hate speech. InProceedings of the international AAAI
conference on web and social media, 2019.
[Mathewet al., 2020 ]Binny Mathew, Navish Kumar, Pawan
Goyal, and Animesh Mukherjee. Interaction dynamics be-
tween hate and counter users on twitter. InProceedings of
the 7th ACM IKDD CoDS and 25th COMAD. 2020.
[McMillan-Majoret al., 2022 ]Angelina McMillan-Major,
Amandalynne Paullada, and Yacine Jernite. An interactive
exploratory tool for the task of hate speech detection.
InProceedings of the Second Workshop on Bridging
Human–Computer Interaction and Natural Language
Processing, Seattle, Washington, 2022.
[Moscaet al., 2023 ]Edoardo Mosca, Daryna Dementieva,
Tohid Ebrahim Ajdari, Maximilian Kummeth, Kirill
Gringauz, Yutong Zhou, and Georg Groh. IFAN: An
explainability-focused interaction framework for humans
and NLP models. InProceedings of the 13th International

Joint Conference on Natural Language Processing and the
3rd Conference of the Asia-Pacific Chapter of the Associ-
ation for Computational Linguistics: System Demonstra-
tions, Bali, Indonesia, 2023.
[Ocampoet al., 2023 ]Nicol ´as Benjam ´ın Ocampo, Ekaterina
Sviridova, Elena Cabrio, and Serena Villata. An in-depth
analysis of implicit and subtle hate speech messages. In
EACL 2023-17th Conference of the European Chapter
of the Association for Computational Linguistics, volume
2023, pages 1997–2013. Association for Computational
Linguistics, 2023.
[Ranasinghe and Zampieri, 2021 ]Tharindu Ranasinghe and
Marcos Zampieri. MUDES: Multilingual detection of of-
fensive spans. InProceedings of the 2021 Conference of
the North American Chapter of the Association for Com-
putational Linguistics: Human Language Technologies:
Demonstrations, Online, 2021.
[Sapet al., 2020 ]Maarten Sap, Saadia Gabriel, Lianhui Qin,
Dan Jurafsky, Noah A. Smith, and Yejin Choi. Social bias
frames: Reasoning about social and power implications
of language. InProceedings of the 58th Annual Meeting
of the Association for Computational Linguistics, Online,
2020.
[Tillmannet al., 2023 ]Christoph Tillmann, Aashka Trivedi,
Sara Rosenthal, Santosh Borse, Rong Zhang, Avirup Sil,
and Bishwaranjan Bhattacharjee. Muted: Multilingual tar-
geted offensive speech identification and visualization. In
Proceedings of the 2023 Conference on Empirical Meth-
ods in Natural Language Processing: System Demonstra-
tions, Singapore, 2023.
[Vidgenet al., 2021 ]Bertie Vidgen, Tristan Thrush, Zeerak
Waseem, and Douwe Kiela. Learning from the worst: Dy-
namically generated datasets to improve online hate detec-
tion. InProceedings of the 59th Annual Meeting of the
Association for Computational Linguistics and the 11th
International Joint Conference on Natural Language Pro-
cessing (Volume 1: Long Papers), Online, 2021.
[Wijanarkoet al., 2024 ]Musa Izzanardi Wijanarko, Lucky
Susanto, Prasetia Anugrah Pratama, Ika Karlina Idris,
Traci Hong, and Derry Tanti Wijaya. Monitoring hate
speech in Indonesia: An NLP-based classification of so-
cial media texts. InProceedings of the 2024 Conference
on Empirical Methods in Natural Language Processing:
System Demonstrations, Miami, Florida, USA, 2024.
[Wrightet al., 2021 ]Austin P. Wright, Omar Shaikh,
Haekyu Park, Will Epperson, Muhammed Ahmed,
Stephane Pinel, Duen Horng (Polo) Chau, and Diyi Yang.
Recast: Enabling user recourse and interpretability of
toxicity detection models with interactive visualization.
2021.
[Zhenget al., 2023 ]Yi Zheng, Bj ¨orn Ross, and Walid
Magdy. What makes good counterspeech? a comparison
of generation approaches and evaluation metrics. InPro-
ceedings of the 1st Workshop on CounterSpeech for Online
Abuse (CS4OA), Prague, Czechia, 2023.