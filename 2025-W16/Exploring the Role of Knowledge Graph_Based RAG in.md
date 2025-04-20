# Exploring the Role of Knowledge Graph-Based RAG in Japanese Medical Question Answering with Small-Scale LLMs

**Authors**: Yingjian Chen, Feiyang Li, Xingyu Song, Tianxiao Li, Issey Sukeda, Irene Li

**Published**: 2025-04-15 08:46:39

**PDF URL**: [http://arxiv.org/pdf/2504.10982v3](http://arxiv.org/pdf/2504.10982v3)

## Abstract
Large language models (LLMs) perform well in medical QA, but their
effectiveness in Japanese contexts is limited due to privacy constraints that
prevent the use of commercial models like GPT-4 in clinical settings. As a
result, recent efforts focus on instruction-tuning open-source LLMs, though the
potential of combining them with retrieval-augmented generation (RAG) remains
underexplored. To bridge this gap, we are the first to explore a knowledge
graph-based (KG) RAG framework for Japanese medical QA small-scale open-source
LLMs. Experimental results show that KG-based RAG has only a limited impact on
Japanese medical QA using small-scale open-source LLMs. Further case studies
reveal that the effectiveness of the RAG is sensitive to the quality and
relevance of the external retrieved content. These findings offer valuable
insights into the challenges and potential of applying RAG in Japanese medical
QA, while also serving as a reference for other low-resource languages.

## Full Text


<!-- PDF content starts -->

Exploring the Role of Knowledge Graph-Based
RAG in Japanese Medical Question Answering
with Small-Scale LLMs
Yingjian Chen1,∗[0009 −0004−1359−6998], Feiyang Li1,∗[0009 −0007−2982−8206],
Xingyu Song1[0009 −0007−1640−0809], Tianxiao Li2[0000 −0002−9147−7511],
Issey Sukeda1[0000 −0002−9147−7511], and Irene Li1[0000 −0002−1851−5390]
1The University of Tokyo, 7-ch¯ ome-3-1 Hong¯ o, Bunkyo City, Tokyo 113-8654, Japan
2NEC Laboratories America, 490 California Ave., 105, Palo Alto, CA 94306, USA
∗Equal Contribution
irene.li@weblab.t.u-tokyo.ac.jp
Abstract. Large language models (LLMs) perform well in medical QA,
but their effectiveness in Japanese contexts is limited due to privacy con-
straints that prevent the use of commercial models like GPT-4 in clinical
settings. As a result, recent efforts focus on instruction-tuning open-
source LLMs, though the potential of combining them with retrieval-
augmented generation (RAG) remains underexplored. To bridge this
gap, we are the first to explore a knowledge graph-based (KG) RAG
framework for Japanese medical QA small-scale open-source LLMs. Ex-
perimental results show that KG-based RAG has only a limited impact
on Japanese medical QA using small-scale open-source LLMs. Further
case studies reveal that the effectiveness of the RAG is sensitive to the
quality and relevance of the external retrieved content. These findings
offer valuable insights into the challenges and potential of applying RAG
in Japanese medical QA, while also serving as a reference for other low-
resource languages.
Keywords: Japanese Medical Question Answering ·RAG ·Small-Scale
LLMs ·Knowledge Graph.
1 Introduction
Largelanguagemodels(LLMs)haveachievedremarkableperformanceinmedical
question answering (QA), even demonstrating the ability to pass medical licens-
ingexams(e.g.,theUnitedStatesMedicalLicensingExamination,USMLE)[19],
which highlights their potential to understand complex medical knowledge. In
particular, recent research [23,25,28] has explored the use of retrieval-augmented
generation (RAG) [6] to incorporate external medical knowledge into LLMs, ef-
fectively mitigating “hallucination” issues [17,29,30] and further enhancing their
applicability in medical QA tasks.
While these advances are promising, most existing efforts [11,27] focus on
English-onlyscenarios,withlimitedexplorationofmultilingualmedicalQA,par-
ticularly in Japanese. Since mainstream LLMs [9,24] are predominantly trainedarXiv:2504.10982v3  [cs.CL]  17 Apr 2025

2 Yingjian Chen, Feiyang Li, et al.
複視において正しいのはどれか？
 A:  複視は⽚⽅の眼を覆っている場合には決して起こらない  
 B:  外側の像は常に虚像である  
 C:  患者が上を向いているときに 第4神経⿇痺 が起こる  
D:  第6神経⿇痺 があると 、発散性の スクインツ が起こる  
第4神経⿇痺
LLM複視
第6神経⿇痺
スクインツDiplopia
Fourth Nerve  
Palsy
Sixth Nerve  
Palsy
SquintsTranslation
Step1: Entity Extraction And TranslationLocal Web UMLS
Step2: Knowledge Graph Retrievalnon-inevtious meningitis
Fourth cranial nerve dysfunctionFourth cranial nerve palsy
ocular movement disordersSixth cranial nerve palsy
non-infectious encephalitis
Guillain-Barré syndrome 
Abducens nerve palsyQ
ABCD21 345Similarity
0.84 0.71 0.57 0.66 0.41
Q
Step3: Multi-angle RankingMedCPT Cross Encoder
Step4: LLM Answering Ranking
D
Declarative  ConversionStatement1
Statement2Q & AInvalid Medical Knowledge Retrieval Self-Knowledge Mining Knowledge Chunk
LLM
Search
Update
複視において 正しいのはどれか？
 A:  複視は⽚⽅の眼を覆っている場合には決して起こらない  
 B:  外側の像は常に虚像である  
 C:  患者が上を向いているときに 第4神経⿇痺 が起こる  
D:  第6神経⿇痺 があると 、発散性の スクインツ が起こる  
第4神経⿇痺
LLM複視
第6神経⿇痺
スクインツDiplopia
Fourth Nerve  
Palsy
Sixth Nerve  
Palsy
SquintsTranslation
Step1: Entity Extraction And TranslationLocal Web UMLS
Step2: Knowledge Graph Retrievalnon-inevtious 
meningitis
Fourth cranial nerve
 dysfunctionFourth cranial nerve palsy
ocular movement 
disordersSixth cranial nerve palsy
non-infectious 
encephalitis
Guillain-Barré 
syndrome 
Abducens nerve palsyQ
21 345Similarity
0.84 0.71 0.57 0.66 0.41
Step3: Multi-angle RankingMedCPT Cross Encoder
Step4: LLM Answering Ranking
D
Declarative 
Conversion
Statement1
Statement2Q & AInvalid Medical Knowledge Retrieval Self-Knowledge Mining Knowledge Chunk
LLM
Search
UpdateABCD Q
............ ......
複視において 正しいのはどれか？
 A:  複視は⽚⽅の眼を覆っている場合には決して起こらない  
 B:  外側の像は常に虚像である  
 C:  患者が上を向いているときに 第4神経⿇痺 が起こる  
D:  第6神経⿇痺 があると 、発散性の スクインツ が起こる  
第4神経⿇痺
LLM複視
第6神経⿇痺
スクインツDiplopia
Fourth Nerve  
Palsy
Sixth Nerve  
Palsy
SquintsTranslation
Step1: Entity Extraction And TranslationLocal Web UMLS
Step2: Knowledge Graph Retrievalnon-inevtious 
meningitis
Fourth cranial nerve
 dysfunctionFourth cranial nerve palsy
ocular movement 
disordersSixth cranial nerve palsy
non-infectious 
encephalitis
Guillain-Barré 
syndrome 
Abducens nerve palsyQ
21 345Similarity
0.84 0.71 0.57 0.66 0.41
Step3: Multi-angle RankingMedCPT Cross Encoder
Step4: LLM Answering Ranking
D
Declarative 
Conversion
Statement1
Statement2Q & AInvalid Medical Knowledge Retrieval Self-Knowledge Mining Knowledge Chunk
LLM
Search
UpdateABCD Q
............ ......Q: Which of the following is correct 
regarding diplopia?
A: Diplopia never occurs when one eye is 
covered. 
B: The outer image is always a virtual image. 
C: The fourth cranial nerve palsy occurs 
when the patient looks upward. 
D: A sixth cranial nerve palsy causes a 
divergent squint.
今夜2回嘔吐し、胸部に非常に 強 い締め付け 感がありますが、 鼻 や 喉 は 通 っ
ています。 禁煙 を 助 けるためにパッチを 使 い 始 めたばかりで、 私 は 喘息持 ち
ですが、 現在 は 酸素 が 体内 に 留 まっていないような 感覚 があります。 脈拍 は
あり、また 不安症状 も経験しています。このパッチが 私 の喘息や呼吸に 影響
しているのでしょうか？
あなたの 症状 について 心配 されるのは 理解 できます。 胸部 の 強 い 締 め 付 け 感
や嘔吐は、 喘息 の 症状 や 他 の 健康問題 の 兆候 である 可能性 があります。 特
に、喘息 を 持 っている 場合、 ストレスや 不安 が 発作 を 引 き 起 こすことがあり
ます。また、 禁煙 のためのパッチが 体 に 与 える 影響 も 考慮 する 必要 がありま
す。 ニコチンパッチは、 禁煙 を 助 けるために 使用 されることが 多 いです
が、喘息 の 症状 を 悪化 させる 可能性 があるため、 注意 が 必要 です。 ..........Kownledge Graph based RAG
Step 1: Medical Entities Extraction
LLM胸部の締 め 付 け 感
不安症状 嘔吐喘息Step 2: Knowledge Graph Retrieval
asthma
anxiety symptoms
vomitingchest tightness
UMLS
Step 3: Ranking
Triple 1
Triple 2
Triple n...Top 1: 0.84
Top 2: 0.76
Top k: 0.64...QuestionSimilarity
Re-rankStep 4: Declarative Conversion
胸部絞扼感 は 喘息 の 症状 である
嘔吐は消化管 の 非特異的症状 である
喘息は好酸球性肺炎 の 一部 である
...Question
今夜2回嘔吐し、胸部に非常に 強 い締め付け 感があります
が、鼻や 喉 は 通 っています。 禁煙 を 助 けるためにパッチを 使
い始めたばかりで、 私 は 喘息持 ちですが、 現在 は 酸素 が 体内
に留まっていないような 感覚 があります。 脈拍 はあり、また
不安症状 も経験しています。このパッチが 私 の喘息や呼吸に
影響しているのでしょうか？
あなたの 症状 について 心配 されるのは 理解 できます。 胸部 の
強い締め 付 け 感 や 嘔吐 は、 喘息 の 症状 や 他 の 健康問題 の 兆候
である可能性 があります。 特 に、 喘息 を 持 っている 場合、 ス
トレスや 不安 が 発作 を 引 き 起 こすことがあります。また、 禁
煙のためのパッチが 体 に 与 える 影響 も 考慮 する 必要 がありま
す。 ニコチンパッチは、 ..........Kownledge Graph based RAG
Step 1: Entity Extraction
LLM胸部の締 め 付 け 感
不安症状
嘔吐喘息Step 2: Knowledge Graph Retrieval
asthma
anxiety symptoms
vomitingchest tightness
UMLS
Step 3: Ranking
Triple 1
Triple 2
Triple n...Top 1: 0.84
Top 2: 0.76
Top k: 0.64Question
Re-rankStep 4: Declarative Conversion
胸部絞扼感 は 喘息 の 症状 である
嘔吐は消化管 の 非特異的症状 である
喘息は好酸球性肺炎 の 一部 であるQuestion...
...Similarity
Fig. 1.The pipeline of our knowledge graph-based RAG mechanism. Given a Japanese
medical question, relevant medical knowledge is retrieved from the external knowledge
base and combined with the original Japanese question as input to the LLM for answer
generation. The English translation and full content are provided in Appendix C.
on English-centric medical data, the significant imbalance in resource distribu-
tion[5]limitstheireffectivenessinJapanesemedicalQAscenarios.WhileGPT-4
has demonstrated strong performance on the Japanese NMLE [10], strict pri-
vacy regulations prevent its use in clinical settings. As a result, research has
shifted toward open-source LLMs, but progress is hindered by the scarcity of
high-quality Japanese medical data—highlighting the broader challenges faced
by low-resource languages.
To address this limitation, in this paper, we are the first to explore the
use of Knowledge Graph-based RAG for Japanese medical question answering,
specifically targeting small-scale LLMs. Given the limited accessibility of ex-
ternal Japanese medical resources, we use the easily accessible external medical
knowledge base UMLS [4] to support knowledge retrieval, following the design of
MKG-Rank [12]. By applying word-level translation, LLMs can integrate non-
Japanese-centric medical knowledge, mitigating the limitations caused by the
relative scarcity of Japanese medical data and its access restrictions. Empirical
results show that the KG-based RAG has limited effect on Japanese medical QA
with small open-source LLMs, largely depending on the quality and relevance of
the retrieved KGs from external knowledge bases.
2 Method
Base Models. Our work focuses on evaluating the performance of small-scale
LLMsinJapanesemedicalQA.Specifically,weselect:(1)modelswithfewerthan
5B parameters, including Borea-Phi-3.5-Mini-Instruct-Common [3] and LLaMA-
3-ELYZA-JP-8B [7]; (2) models with 5B–10B parameters, including LLM-JP-3-
7.2B-Instruct3 [15], Mistral-7B-Instruct-v0.3 [2], Qwen2.5-7B-Instruct [22], and
LLaMA-3.1-8B-Instruct [14]; and (3) 10B–32B models, including Qwen2.5-14B-
Instruct [21], Phi-4-14B [18], and Gemma-3-12B-it [8]. In addition, we evaluate
the commercial LLM GPT-4o-mini [20] as a proprietary baseline.
Knowledge Graph-based RAG. Our pipeline, as illustrated in Fig. 1, con-
sists of four main steps: (1) Given a medical question Q, we first use an LLM

Exploring the Role of Knowledge Graph-Based RAG 3
ExpertQA-Bio ExpertQA-Med LiveQA
Model ROUGE-L BERTScore ROUGE-L BERTScore ROUGE-L BERTScore
Borea-Phi-3.5 [3] 4.33 61.20 4.92 51.54 4.05 59.98
Borea-Phi-3.5 + RAG 4.77 (+0.44%) 61.32 (+0.12%) 4.82 (-0.10%) 61.39 (-0.15%) 3.97 (-0.08%) 59.71 (-0.28%)
LLaMA-3-ELYZA-JP-8B [7] 22.29 76.51 23.12 76.81 16.23 71.51
LLaMA-3-ELYZA-JP-8B + RAG 22.53 (+0.24%) 76.22 (-0.29%) 23.13 (+0.01%) 76.75 (-0.06%) 16.37 (+0.14%) 71.56 (+0.05%)
Mistral-7B [2] 20.85 75.25 20.21 74.67 15.70 70.99
Mistral-7B + RAG 17.39 (-3.46%) 72.13 (-3.12%) 17.59 (-2.62%) 72.35 (-2.32%) 13.76 (-1.94%) 69.15 (-1.84%)
Qwen2.5-7B [22] 19.09 75.15 19.52 75.39 14.60 70.88
Qwen2.5-7B + RAG 20.15 (+1.06%) 75.18 (+0.03%) 20.56 (+1.04%) 74.95 (-0.44%) 15.97 (+1.37%) 71.68 (+0.80%)
LLaMA-3.1-8B [14] 18.84 74.09 18.49 74.42 14.78 70.50
LLaMA-3.1-8B + RAG 18.01 (-0.83%) 72.04 (-2.05%) 18.01 (-0.48%) 72.40 (-2.02%) 14.43 (-0.35%) 69.30 (-1.20%)
Qwen2.5-14B [21] 17.00 71.92 17.01 71.44 13.85 68.63
Qwen2.5-14B + RAG 17.19 (+0.19%) 71.25 (-0.67%) 17.47 (+0.46%) 71.74 (+0.30%) 13.89 (+0.04%) 68.62 (-0.01%)
Phi-4-14B [18] 9.72 65.89 9.15 65.12 8.78 64.97
Phi-4-14B + RAG 12.18 (+2.46%) 67.38 (+1.49%) 13.93 (+4.78%) 69.13 (+4.01%) 11.18 (+2.40%) 66.75 (+1.78%)
Gemma-3-12B-it [8] 19.79 74.65 21.25 74.89 14.61 69.71
Gemma-3-12B-it + RAG 20.44 (+0.65%) 75.13 (+0.48%) 21.99 (+0.74%) 75.71 (+0.82%) 15.34 (+0.73%) 70.42 (+0.71%)
GPT-4o-mini [20] 24.92 77.67 26.78 78.67 17.94 73.25
GPT-4o-mini + RAG 24.41 (-0.51%) 77.45 (-0.22%) 26.20 (-0.58%) 78.53 (-0.14%) 17.85 (-0.09%) 73.28 (+0.03%)
Table 1. EvaluationofROUGE-LandBERTScoreforbasemodelsandtheirKG-based
RAG-enhanced variants on three medical QA datasets: ExpertQA-Bio, ExpertQA-
Med, and LiveQA.
to extract relevant medical entities E; (2) The extracted entities Eare trans-
lated into English via word-level translation and used to query external medical
knowledge base UMLS to obtain relevant KGs G. (3) A ranking mechanism is
applied to select the most relevant triples G′based on their semantic relevance
to the question Q; (4) The selected triples are then converted into declarative
sentences by an LLM, along with the original question, are provided as input to
the LLM for answer generation.
3 Experiment
Datasets. We selected three long-form medical question answering datasets:
ExpertQA-Bio, ExpertQA-Med, and LiveQA. ExpertQA is a high-quality QA
dataset verified by domain experts, from which we used 96 biological questions
(ExpertQA-Bio) and 504 medical questions (ExpertQA-Med) [16]. LiveQA con-
sistsofconsumerhealthquestionssubmittedtotheNationalLibraryofMedicine,
comprising 627 training QA pairs and 104 test pairs [1]. All datasets were orig-
inally in English and were translated into Japanese using a multi-stage agentic
machine translation framework [26]. More details are provided in Appendix A
Results and Analysis. Table 1 presents the performance comparison between
KG-based RAG and the baseline LLMs on the medical QA task. In general, the
results suggest that the impact of our RAG mechanism is relatively limited, with
performance fluctuations observed across different base models. Specifically, for
Borea-Phi-3.5, LLaMA-3-ELYZA-JP-8B, Qwen2.5-14B, Gemma-3-12B-it, and
GPT-4o-mini, the RAG leads to marginal changes, with performance fluctuating
within 1% compared to the respective base models. In contrast, Qwen2.5-7B and
Phi-4-14B exhibit overall improvements when enhanced with RAG, particularly
Phi-4-14B, which achieves improvements of up to +4.78% in ROUGE-L and

4 Yingjian Chen, Feiyang Li, et al.
Question:ワルファリン （ワーファリン ）を服用している 人は避けるべき 野菜は何ですか？(Which
vegetables should people taking warfarin (Coumadin) avoid?)
Retrieved Medical Knowledge: 1.ワルファリンは 生理的に凝固因子濃度を低下させる効果があ
る。(Warfarin physiologically reduces the concentration of coagulation factors.) 2. ワルファリンは
クマリン 系の抗凝固薬である。(Warfarin is a coumarin-based anticoagulant.) ...
Ground Truth: ワーファリンを 服用している 人は、ビタミン Kを多く含む野菜の摂取を避けるか制限する
必要があります。なぜなら、それが 薬の血液凝固抑制効果（血液をサラサラにする 効果）を妨げる可能性があ
るからです。 ...(People taking warfarin should avoid or limit the intake of vegetables that are high
in vitamin K, as it may interfere with the medication’s blood clotting inhibition effect (the effect of
making the blood thinner).)
Generated Answer: ワルファリンを 服用している 人が避けるべき 野菜は、特にビタミン Kを多く含む
ものです。ビタミン Kは血液凝固に関与しており、ワルファリンの 効果を減少させる可能性があります。
...(People taking warfarin should avoid vegetables that are particularly high in vitamin K. Vitamin
K is involved in blood clotting and may reduce the effectiveness of warfarin.)
Question: 抗生薬は通常、細菌感染または真菌感染のある患者に投与されます。ウイルスが 原因であ
ることが 多い疾患については、ではどうでしょうか ？(Antibiotics are typically administered to patients
with bacterial or fungal infections. But what about diseases that are often caused by viruses?)
Retrieved Medical Knowledge: 1.バイ菌感染は、他のバイ菌性疾患に分類されます。 (Bacterial infec-
tions are classified under other bacterial diseases.) 2. バイ菌感染は、肝膿瘍に分類されます。 (Bacterial
infections are classified as liver abscesses.) ...
Ground Truth: 抗生物質は、感冒、インフルエンザ、 様々な咳などのウイルス 感染症に対しては効果
がありません。 ...これは、ウイルスが 細胞ではなく、 細菌細胞と比較して細胞壁を持たないためです。
...(Antibioticsareineffectiveagainstviralinfectionssuchasthecommoncold,influenza,andvarious
types of cough. ...This is because viruses do not have a cell wall, unlike bacterial cells. ...)
Generated Answer: 抗生物質は主に細菌感染に対して効果がありますが、ウイルス 感染に対しては
効果がありません。ウイルスが 原因の疾患、例えばインフルエンザや 風邪などには、 抗生物質は無効で
す。...(Antibiotics are mainly effective against bacterial infections but are ineffective against viral
infections. For illnesses caused by viruses, influenza or the common cold, antibiotics do not work.)
Table 2. Case Study. Two cases are presented, with full content and English transla-
tions provided in Appendix E.
+4.01% in BERTScore. On the other hand, the RAG leads to negative impacts
on Mistral-7B and LLaMA-3.1-8B, resulting in performance degradation.
Case Study. To further investigate the reasons behind the relatively limited
impact of RAG, we analyze the intermediate reasoning process, as shown in
Table 2. Specifically, in Case 1 (top), the retrieved medical knowledge includes
the information that is relevant and helpful for answering the question (e.g.
“ワルファリンは生理的に凝固因子濃度を低下させる効果がある”). However,
we observe that most LLMs already possess this knowledge internally, which
explainswhyRAGoffersonlymarginalimprovementsinsuchcases.Ontheother
hand, for Case 2 (bottom), the retrieved medical knowledge does not help answer
the question (e.g. “ バイ菌感染は、肝膿瘍に分類されます ”). This is primarily
because the specialized biomedical concepts in UMLS may not align well with
broader,moregeneralmedicalquestions.Consequently,theretrievedinformation
provideslimitedassistanceandmayevenintroducenoisethatnegativelyimpacts
the LLM’s reasoning.
4 Conclusion
This work presents the first exploration of a knowledge graph-based RAG frame-
work for Japanese medical QA using small-scale open-source LLMs. Empirical
findings show that its overall impact is limited, primarily constrained by the
quality and relevance of the retrieved content. These insights highlight the chal-
lenges and potential of applying RAG to Japanese and other low-resource lan-
guage medical QA tasks.

Exploring the Role of Knowledge Graph-Based RAG 5
Acknowledgment
This work was supported by JST ACT-X (Grant JPMJAX24CU) and JSPS
KAKENHI (Grant 24K20832).
References
1. Abacha, A.B., Agichtein, E., Pinter, Y., Demner-Fushman, D.: Overview of the
medical question answering task at trec 2017 liveqa. In: TREC. pp. 1–12 (2017)
2. AI, M.: Mistral-7b-instruct-v0.3. https://huggingface.co/mistralai/Mistral-7B-
Instruct-v0.3 (2024), retrieved April 08, 2025
3. AXCXEPT: Borea-phi-3.5-mini-instruct-common. https://huggingface.co/AXCX
EPT/Borea-Phi-3.5-mini-Instruct-Common (2024), retrieved April 08, 2025
4. Bodenreider, O.: The unified medical language system (umls): integrating biomed-
ical terminology. Nucleic acids research 32(suppl_1), D267–D270 (2004)
5. Chataigner, C., Taïk, A., Farnadi, G.: Multilingual hallucination gaps in large
language models. arXiv preprint arXiv:2410.18270 (2024)
6. Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S.,
Metropolitansky, D., Ness, R.O., Larson, J.: From local to global: A graph rag
approach to query-focused summarization. arXiv preprint arXiv:2404.16130 (2024)
7. ELYZA: Llama-3-elyza-jp-8b. https://huggingface.co/elyza/Llama-3-ELYZA-JP-
8B (2024), retrieved April 08, 2025
8. Google: gemma-3-12b-it. https://huggingface.co/google/gemma-3-12b-it (2025),
retrieved April 08, 2025
9. Hurst, A., Lerer, A., Goucher, A.P., Perelman, A., Ramesh, A., Clark, A., Os-
trow, A., Welihinda, A., Hayes, A., Radford, A., et al.: Gpt-4o system card. arXiv
preprint arXiv:2410.21276 (2024)
10. Kasai, J., Kasai, Y., Sakaguchi, K., Yamada, Y., Radev, D.: Evaluating gpt-
4 and chatgpt on japanese medical licensing examinations. arXiv preprint
arXiv:2303.18027 (2023)
11. Ke, Y., Yang, R., Lie, S.A., Lim, T.X.Y., Ning, Y., Li, I., Abdullah, H.R., Ting,
D.S.W., Liu, N.: Mitigating cognitive biases in clinical decision-making through
multi-agent conversations using large language models: simulation study. Journal
of Medical Internet Research 26, e59439 (2024)
12. Li, F., Chen, Y., Liu, H., Yang, R., Yuan, H., Jiang, Y., Li, T., Taylor, E.M.,
Rouhizadeh, H., Iwasawa, Y., et al.: Mkg-rank: Enhancing large language models
with knowledge graph for multilingual medical question answering. arXiv preprint
arXiv:2503.16131 (2025)
13. Lin, C.Y.: Rouge: A package for automatic evaluation of summaries. In: Text sum-
marization branches out. pp. 74–81 (2004)
14. Llama, M.: Llama-3.1-8b-instruct. https://huggingface.co/meta-llama/Llama-3.1-
8B-Instruct (2024), retrieved April 08, 2025
15. LLM-jp: llm-jp-3-7.2b-instruct3. https://huggingface.co/llm-jp/llm-jp-3-7.2b-
instruct3 (2024), retrieved April 08, 2025
16. Malaviya,C.,Lee,S.,Chen,S.,Sieber,E.,Yatskar,M.,Roth,D.:Expertqa:Expert-
curated questions and attributed answers. arXiv preprint arXiv:2309.07852 (2023)
17. McKenna, N., Li, T., Cheng, L., Hosseini, M.J., Johnson, M., Steedman, M.:
Sourcesofhallucinationbylargelanguagemodelsoninferencetasks.arXivpreprint
arXiv:2305.14552 (2023)

6 Yingjian Chen, Feiyang Li, et al.
18. Microsoft: phi-4. https://huggingface.co/microsoft/phi-4 (2024), retrieved April
08, 2025
19. Nori, H., King, N., McKinney, S.M., Carignan, D., Horvitz, E.: Capabilities of
gpt-4 on medical challenge problems. arXiv preprint arXiv:2303.13375 (2023)
20. OpenAI: Gpt-4o-mini. https://openai.com/index/
21. Qwen: Qwen2.5-14b-instruct. https://huggingface.co/Qwen/Qwen2.5-14B-
Instruct (2024), retrieved April 08, 2025
22. Qwen: Qwen2.5-7b-instruct. https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
(2024), retrieved April 08, 2025
23. Shi, Y., Xu, S., Yang, T., Liu, Z., Liu, T., Li, Q., Li, X., Liu, N.: Mkrag: Medical
knowledge retrieval augmented generation for medical question answering. arXiv
preprint arXiv:2309.16035 (2023)
24. Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.A., Lacroix, T.,
Rozière, B., Goyal, N., Hambro, E., Azhar, F., et al.: Llama: Open and efficient
foundation language models. arXiv preprint arXiv:2302.13971 (2023)
25. Xiong, G., Jin, Q., Lu, Z., Zhang, A.: Benchmarking retrieval-augmented gen-
eration for medicine. In: Ku, L.W., Martins, A., Srikumar, V. (eds.) Findings
of the Association for Computational Linguistics: ACL 2024. pp. 6233–6251. As-
sociation for Computational Linguistics, Bangkok, Thailand (Aug 2024). https:
//doi.org/10.18653/v1/2024.findings-acl.372 ,https://aclanthology.org/
2024.findings-acl.372/
26. Xuan, W., Yang, R., Qi, H., Zeng, Q., Xiao, Y., Xing, Y., Wang, J., Li, H., Li, X.,
Yu, K., et al.: Mmlu-prox: A multilingual benchmark for advanced large language
model evaluation. arXiv preprint arXiv:2503.10497 (2025)
27. Yang, H., Chen, H., Guo, H., Chen, Y., Lin, C.S., Hu, S., Hu, J., Wu, X., Wang, X.:
Llm-medqa: Enhancing medical question answering through case studies in large
language models. arXiv preprint arXiv:2501.05464 (2024)
28. Yang, R., Liu, H., Marrese-Taylor, E., Zeng, Q., Ke, Y.H., Li, W., Cheng, L.,
Chen, Q., Caverlee, J., Matsuo, Y., et al.: Kg-rank: Enhancing large language mod-
els for medical qa with knowledge graphs and ranking techniques. arXiv preprint
arXiv:2403.05881 (2024)
29. Yang, R., Ning, Y., Keppo, E., Liu, M., Hong, C., Bitterman, D.S., Ong, J.C.L.,
Ting, D.S.W., Liu, N.: Retrieval-augmented generation for generative artificial in-
telligence in health care. npj Health Systems 2(1), 2 (2025)
30. Zhang, M., Press, O., Merrill, W., Liu, A., Smith, N.A.: How language model
hallucinations can snowball. arXiv preprint arXiv:2305.13534 (2023)
31. Zhang, T., Kishore, V., Wu, F., Weinberger, K.Q., Artzi, Y.: Bertscore: Evaluating
text generation with bert. arXiv preprint arXiv:1904.09675 (2019)
A Dataset Details
We provide detailed statistics of the evaluation datasets, including the number
of samples and the average word count of questions and answers, as shown in
Table 3.
B Metrics
We adopt ROUGE-L [13] and BERTScore [31] to evaluate the quality of answers
generated by LLMs for Japanese medical QA tasks. These metrics provide a
comprehensive assessment of both lexical and semantic accuracy.

Exploring the Role of Knowledge Graph-Based RAG 7
Dataset Size Question Length Answer Length
ExpertQA-Bio 96 56.7 410.7
ExpertQA-Med 504 56.0 378.1
LiveQA 627 118.9 438.3
Table 3. Statistics of the evaluation datasets.
C English Translation of the Content in Fig. 1
For clearer presentation, we provide the English translation and the full content
of the question and answer in the Fig. 1, as illustrated in Fig. 2.
D prompts
In this section, we provide the prompts used in our knowledge graph-based RAG
mechanism, including Medical Entity Extraction (Table 4), Declarative Conver-
sion (Table 5) and Answer Generation (Table 6).
Medical Entity Extraction
text: { question }
Pleaseextractatmost4termsrelatedtomedicalthatyouthinkarethemostimportant
from the provided text.
Returns the result in the following json form. All the results are merged into one json.
–Examples of results:
{"medical terminologies" : ["term1", "term2", ...]}
result:
Table 4. Prompt Used for Medical Entity Extraction.
Declarative Conversion
あなたは 医学分野の知能助手です。
すべての 背景知識をそれぞれ 日本語の平叙文に変換する。医学に関係ないと思うもの
は何でも削除できます。
- Background Knowledge: { triple}
Converted Background Knowledge:
Table 5. Prompt Used for Declarative Conversion.

8 Yingjian Chen, Feiyang Li, et al.
4 orders of magnitude
speed improvement
Scenario 1
Q: 複視において 正しいのはどれか？
A:  複視は 片方 の 眼 を 覆 っている場
合には決 して 起 こらない   
B:  外側の 像 は 常 に 虚像 である   
C:  患者が 上 を 向 いているとき に第4神
経麻痺が起こる   
D:  第6神経麻痺 があると、 発散性の
スクインツ が起こるQ & A
diplopia fourth nerve palsy
squints sixth nerve palsy
1. ('MEDICAL', 'isa', 'UNKNOWN/MISC') 
2. ('médico', 'isa', 'modificador relacionado con una ocupación Y/O especialidad
clínica') 
3. ('第４脳神経麻痺 ', 'member of', ' 眼球運動障害（ＳＭＱ） ’) 
4. ('複視 ', 'translation of', 'V ision double’) 
5. ('外転神経麻痺 ', 'member of', ' 眼球運動障害（ＳＭＱ） ') 
6. ('第６脳神経麻痺 ', 'member of', ' 非感染性髄膜炎（ＳＭＱ） ') 
7. ('Dviguba rega', 'classified as', 'Diplopija')Retrieved Medical KGs (triplets )
1. Fourth cranial nerve palsy is a member of non-infectious encephalitis (SMQ). 
2. Sixth cranial nerve palsy is a member of ocular movement disorders (SMQ). 
3. Trochlear nerve palsy is a member of non-infectious encephalitis (SMQ). 
4. Fourth cranial nerve palsy is a member of non-infectious encephalitis (SMQ). 
5. Sixth cranial nerve palsy is a member of Guillain-Barré syndrome (SMQ). 
6. Abducens nerve palsy is a member of non-infectious meningitis (SMQ).Declarative Form of Medical Knowledge
Explanation :
The original QA  and refined medical
knowledge are jointly fed into the LLM,
enhancing its medical reasoning.
Answer : DQ: 複視において 正しいのはどれか？
A:  複視は 片方 の 眼 を 覆 っている場
合には決 して 起 こらない   
B:  外側の 像 は 常 に 虚像 である   
C:  患者が 上 を 向 いているとき に第4神
経麻痺が起こる   
D:  第6神経麻痺 があると、 発散性の
スクインツ が起こるdiplopia fourth nerve palsy
squints sixth nerve palsyExtracted and English-translated entities
1. ('MEDICAL', 'isa', 'UNKNOWN/MISC') 
2. ('第４脳神経麻痺 ', 'member of', ' 眼球運動 障害
（ＳＭＱ） ’) 
3. ('複視 ', 'translation of', 'V ision double’) 
......Retrieved Medical KGs (triplets )
1. Fourth cranial nerve palsy is a member of 
non-infectious encephalitis (SMQ). 
2. Sixth cranial nerve palsy is a member of ocular
 movement disorders (SMQ). 
3. Trochlear nerve palsy is a member of 
non-infectious encephalitis (SMQ). 
......Declarative Form of Medical Knowledge
Explanation :
The original QA  and refined medical
knowledge are jointly fed into the LLM,
enhancing its medical reasoning.
Answer : DExtracted and English-translated entities
English Version
Q: Which of the following is correct regarding diplopia?
A: Diplopia never occurs when one eye is covered. 
B: The outer image is always a virtual image. 
C: The fourth cranial nerve palsy occurs when the 
patient looks upward. 
D: A sixth cranial nerve palsy causes a divergent squint. Q: マラソンを 成功 させる 鍵 となる 属
性は何か？
A:  ストレングス  
B:  パワー  
C:  ストライドの 長 さ  
D:  スタミナStrengths Power
Length of stride StaminaExtracted and English-translated entities
1. ('Power', 'component of', 'Power level^peak during
 exercise:EngRat:Pt:^Patient:Qn')  
2. ('Power', 'component of', 'Cylindrical power:
InvLen:Pt:Eye.left:Qn')  
...... Retrieved Medical KGs (triplets )
ちますが、マラソンでは 持続的 なエネ
ルギー供給が 求 められるため、スタミ
ナが鍵となります。また、ストライド
の長さも影響を与 える 要素 ですが、ス
タミナが 最も重要な 属性 とされています。
マラソンを 成功させるためには、 持久力
や体力が 重要です。特 に「スタミナ」は
長時間の 運動を持続するために 必要 な 能
力であり ......Self-Knowledge Mining
Explanation :
when retrieving invalid medical KGs, we perform
self-knowledge mining to acquire relevant know-
ledge chunks , enhancing the LLM’ s reasoning 
in medical QA.
Answer : DInvalid low-score medical KG retrieval
Scenario 2
Q: マラソンを 成功 させる 鍵 となる 属
性は何か？
A:  ストレングス  
B:  パワー  
C:  ストライドの 長 さ  
D:  スタミナQ & A
Strengths Power
Length of stride StaminaExtracted and English-translated entities
1. ('Power', 'component of', 'Power level^peak during exercise:EngRat:Pt:
^Patient:Qn')  
2. ('Power', 'component of', 'Cylindrical power:InvLen:Pt:Eye.left:Qn')  
3. ('Power', 'component of', 'Heater power:EngRat:Pt:T ranscutaneous
 monitor:Qn') Retrieved Medical KGs (triplets )
ちますが、マラソンでは 持続的 なエネルギ ー供給が 求 められるため
、スタミナ が鍵となります。また、ストライドの 長 さも影響を与 え
る要素ですが、スタミ ナが最も重要な 属性 とされています。  
マラソンを 成功 させるためには、 持久力や体力が 重要 です。 特 に
「スタミナ」 は長時間の 運動 を 持続 するために 必要 な 能力であり、
マラソンのような 長距離走においては 特 に 重要 です。ストレングス
やパワーも 役立 ちますが、マラソン では持続的なエネルギー 供給 が
求められるためExplanation :
when retrieving invalid medical KGs, 
we perform self-knowledge mining to 
acquire relevant know- ledge chunks , 
enhancing the LLM’ s reasoning 
in medical QA.
Answer : DSelf-Knowledge MiningEnglish Version
Q: What are the key attributes for successfully 
completing a marathon?
A: Strength 
B: Power 
C: Stride length 
D: StaminaScenario 1
Q & A
Scenario 2
Q & A
今夜2回嘔吐し、胸部に非常に 強 い締め付け 感があります
が、鼻や 喉 は 通 っています。 禁煙 を 助 けるためにパッチを 使
い始めたばかりで、 私 は 喘息持 ちですが、 現在 は 酸素 が 体内
に留まっていないような 感覚 があります。 脈拍 はあり、また
不安症状 も経験しています。このパッチが 私 の喘息や呼吸に
影響しているのでしょうか？
あなたの 症状 について 心配 されるのは 理解 できます。 胸部 の
強い締め 付 け 感 や 嘔吐 は、 喘息 の 症状 や 他 の 健康問題 の 兆候
である可能性 があります。 特 に、 喘息 を 持 っている 場合、 ス
トレスや 不安 が 発作 を 引 き 起 こすことがあります。また、 禁
煙のためのパッチが 体 に 与 える 影響 も 考慮 する 必要 がありま
す。 ニコチンパッチは、 ..........Kownledge Graph based RAG
Step 1: Entity Extraction
LLM胸部の締 め 付 け 感
不安症状
嘔吐喘息Step 2: Knowledge Graph Retrieval
asthma
anxiety symptoms
vomitingchest tightness
UMLS
Step 3: Ranking
Triple 1
Triple 2
Triple n...Top 1: 0.84
Top 2: 0.76
Top k: 0.64Question
Re-rankStep 4: Declarative Conversion
胸部絞扼感 は 喘息 の 症状 である
嘔吐は消化管 の 非特異的症状 である
喘息は好酸球性肺炎 の 一部 であるQuestion...
...Similarity
今夜2回嘔吐し、胸部に非常に 強 い締め付け 感があります
が、鼻や 喉 は 通 っています。 禁煙 を 助 けるためにパッチを 使
い始めたばかりで、 私 は 喘息持 ちですが、 現在 は 酸素 が 体内
に留まっていないような 感覚 があります。 脈拍 はあり、また
不安症状 も経験しています。このパッチが 私 の喘息や呼吸に
影響しているのでしょうか？I vomited twice tonight and have a very strong tightness in
my chest, although my nose and throat are clear. I just
started using a patch to help me quit smoking. I have asthma,
but right now it feels like oxygen isn’t staying in my body. I
have a pulse, and I’m also experiencing anxiety symptoms.
Could this patch be affecting my asthma or breathing?
あなたの 症状 について 心配 されるのは 理解 できます。 胸部 の
強い締め 付 け 感 や 嘔吐 は、 喘息 の 症状 や 他 の 健康問題 の 兆候
である可能性 があります。 特 に、 喘息 を 持 っている 場合、 ス
トレスや 不安 が 発作 を 引 き 起 こすことがあります。また、 禁
煙のためのパッチが 体 に 与 える 影響 も 考慮 する 必要 がありま
す。
ニコチンパッチは、 禁煙 を 助 けるために 使用 されることが 多
いですが、 喘息 の 症状 を 悪化 させる 可能性 があるため、 注意
が必要です。 特 に、 胸部 の 締 め 付 け 感 や 呼吸困難 を 感 じてい
る場合は、パッチの 使用 を 中止 し、 医師 に 相談 することをお
勧めします。
また、嘔吐 や 不安症状 も、 身体 のストレス 反応 の 一部 である
可能性があります。これらの 症状 が 続 く 場合 や 悪化 する 場合
は、早急 に 医療機関 を 受診 することが 重要 です。あなたの 健
康が最優先 ですので、 専門家 の 意見 を 仰 ぐことを 強 くお 勧 め
します。It is understandable that you are concerned about your
symptoms. Severe chest tightness and vomiting could be
signs of asthma or other health issues. Especially for
individuals with asthma, stress and anxiety can sometimes
trigger symptoms. It is also important to consider the
potential effects of the nicotine patch on your body.
Nicotine patches are commonly used to support smoking
cessation, but they may worsen asthma symptoms in some
cases, so caution is necessary. If you are experiencing
chest tightness or difficulty breathing, it is advisable to
stop using the patch and consult a doctor.
Additionally, vomiting and anxiety symptoms may be part
of your body’s stress response. If these symptoms
persist or worsen, it is crucial to seek medical attention
promptly. Your health is the top priority, and we strongly
recommend consulting a medical professional.
Retrieved Medical Knowledge
胸部絞扼感 は 喘息 の 症状 である
胸部絞扼感 はアナフィラキシー 反応 の 一部 である
喘息は好酸球性肺炎 の 一部 である嘔吐は消化管 の 非特異的症状 であるRetrieved Medical Knowledge
Chest tightness is a symptom of asthma
Chest tightness can be part of an anaphylactic reaction
Asthma is a component of eosinophilic pneumoniaVomiting is a nonspecific symptom of the gastrointestinal tract
Fig. 2.English translation and the full content of the question and answer in the Fig. 1
E Case Study Details
For clarity and ease of understanding, we provide the full content and English
translations of the cases from Table 2 here, as shown in Fig. 3 and Fig. 4.

Exploring the Role of Knowledge Graph-Based RAG 9
Answer Generation with Medical Knowledge
あなたは 医学分野の知能助手です。質問をよく分析し、提供された背景知識とあなた
自身の知識に基づいて以下の質問に答えてください。できるだけ 512のtoken内で完全
に回答します。
日本語で質問に答える。
-問題: {question }
-背景知識: {background_knowledge }
-答える:
Table 6. Prompt Used for Answer Generation with Medical Knowledge.
Case 1
ワルファリン （ワーファリン）を 服用している
人は避けるべき野菜は何ですか？
Which vegetables should people taking
warfarin (Coumadin) avoid?Question:
Ground Truth:
ワーファリンを服用している人は、ビタミンKを多く含
む野菜の摂取を避けるか制限する必要があります。なぜ
なら、それが薬の血液凝固抑制効果（血液をサラサラに
する効果）を妨げる可能性があるからです。避けるか適
度に摂取すべき野菜には、ケール、ほうれん草、コラー
ドグリーン（ケールに似た葉野菜）、フダンソウ（スイ
スチャード）、カラシナ（からし菜）、ビートの葉、か
ぶの葉、ブロッコリーなどの葉物野菜が含まれます。さ
らに、パセリやコリアンダー（パクチー）などのハーブ
もビタミンKを多く含むため、ワーファリン服用中は適
度に摂取すべきです。
People taking warfarin should avoid or limit the
intake of vegetables rich in vitamin K, as they
may interfere with the blood-thinning effect of
the medication. Vegetables that should be avoided
or consumed in moderation include kale, spinach,
collard greens (leafy greens similar to kale),
Swiss chard, mustard greens, beet greens, turnip
greens, and broccoli. Additionally, herbs such as
parsley and coriander (cilantro), which are also
high in vitamin K, should be consumed in
moderation while taking warfarin.Retrieved Medical Knowledge:
1.ワルファリンは生理的に凝固因子濃度を低下させる効果がある。
Warfarin physiologically reduces the concentration of
coagulation factors.  
2.ワルファリンは生理的に凝固因子活性を低下させる効果がある。
Warfarin has the effect of lowering the activity of coagulation
factors physiologically.
3.ワルファリンはクマリン系の抗凝固薬である。
Warfarin is a coumarin-based anticoagulant.
4。ワルファリンはWARFARIN SODIUM 3 mg経口錠剤の有効成分である。
Warfarin is the active ingredient in WARFARIN SODIUM 3 mg oral
tablets. 
Generated Answer:
ワルファリンを服用している人が避けるべき野菜は、特に
ビタミンKを多く含むものです。ビタミンKは血液凝固に関
与しており、ワルファリンの効果を減少させる可能性があ
ります。具体的には、以下のような野菜が挙げられます。
- ケール  - ほうれん草  - ブロッコリー  - キャベツ  -
パセリ
これらの野菜はビタミンKの含有量が高いため、ワルファリ
ンの効果を妨げる可能性があるため、摂取を控えるか、医
師と相談の上で適切な量を守ることが重要です。
People taking warfarin should avoid vegetables that
are particularly high in vitamin K, as vitamin K is
involved in blood clotting and may reduce the
effectiveness of warfarin. Specifically, the
following vegetables should be considered:
- Kale -Spinach -Broccoli -Cabbage -Parsley
These vegetables contain high levels of vitamin K and
may interfere with the effect of warfarin. It is
important to either avoid them or consult with a
doctor to ensure an appropriate amount is consumed.
Fig. 3.Full content and English translation of the case 1 in Table 2.

10 Yingjian Chen, Feiyang Li, et al.
Case 2
抗生薬は通常、 細菌感染 または真菌感染 のある
患者に投与されます。 ウイルス が原因であるこ
とが多い疾患については、ではどうでしょう
か？
Antibiotics are typically administered to
patients with bacterial or fungal
infections. But what about diseases that
are often caused by viruses?Question:
Ground Truth:
抗生物質は、感冒、インフルエンザ、様々な咳などのウ
イルス感染症に対しては効果がありません。ウイルス感
染症の患者に対して、医師は症状管理のための一般用医
薬品を含む治療を推奨することがあり、抗生物質を使用
する場合は必要な場合にのみ使用することが常に推奨さ
れます。場合によっては、HIVに対する抗ウイルス療
法、インフルエンザ、COVID-19、サル痘(mpox)、または
インフルエンザに対するオセルタミビルリン酸塩（タミ
フル）など、特定のウイルスにのみ効果のある特定の薬
剤が処方されることがあります。抗生物質は細菌感染症
に特化して処方されるものであり、ウイルス感染症には
効果がないことに注意することが重要です。これは、ウ
イルスが細胞ではなく、細菌細胞と比較して細胞壁を持
たないためです。したがって、抗生物質はウイルス感染
症に対して効果を示しません。
Antibiotics are not effective against viral
infections such as the common cold, influenza, and
various types of coughs. For patients with viral
infections, doctors may recommend treatments that
include over-the-counter medications for symptom
management, and antibiotics should only be used
when necessary. In some cases, specific antiviral
medications may be prescribed for certain viruses,
such as antiretroviral therapy for HIV,
oseltamivir phosphate (Tamiflu) for influenza, or
treatments for COVID-19 or monkeypox (mpox). It is
important to note that antibiotics are prescribed
specifically for bacterial infections and are not
effective for viral infections. This is because
viruses do not have a cell wall, unlike bacterial
cells. Therefore, antibiotics are ineffective
against viral infections.Retrieved Medical Knowledge:
1.バイ菌感染は、ニトロフラン抗菌薬によって予防されることがあり
ます。
Bacterial infections may be prevented by nitrofuran
antibiotics.
2.バイ菌感染は、他のバイ菌性疾患に分類されます。
Bacterial infections are classified under other bacterial
diseases.
3.バイ菌感染は、肝膿瘍に分類されます。
Bacterial infections are classified as liver abscesses.
4.バイ菌感染は、他のバイ菌感染に分類されます。
Bacterial infections are classified as other bacterial
infections.
5.バイ菌感染は、ニトロフラン抗菌薬によって治療されることがあり
ます。
Bacterial infections may be treated with nitrofuran
antibiotics.
Generated Answer:
抗生物質は主に細菌感染に対して効果がありますが、ウイ
ルス感染に対しては効果がありません。ウイルスが原因の
疾患、例えばインフルエンザや風邪などには、抗生物質は
無効です。ウイルス感染に対しては、抗ウイルス薬が使用
されることがあります。 したがって、ウイルスが原因であ
ることが多い疾患に対しては、抗生物質を投与することは
適切ではありません。ウイルス感染に対しては、症状を緩
和するための対症療法や、必要に応じて抗ウイルス薬が用
いられます。
Antibiotics are primarily effective against bacterial
infections but are not effective against viral
infections. For diseases caused by viruses, such as
influenza and the common cold, antibiotics are
ineffective. Antiviral medications may be used for
viral infections. Therefore, it is not appropriate to
administer antibiotics for diseases that are often
caused by viruses. For viral infections, symptomatic
treatments are used to relieve symptoms, and
antiviral medications may be used when necessary.
Fig. 4.Full content and English translation of the case 2 in Table 2.