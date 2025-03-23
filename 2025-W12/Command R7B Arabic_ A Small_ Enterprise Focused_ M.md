# Command R7B Arabic: A Small, Enterprise Focused, Multilingual, and Culturally Aware Arabic LLM

**Authors**: Yazeed Alnumay, Alexandre Barbet, Anna Bialas, William Darling, Shaan Desai, Joan Devassy, Kyle Duffy, Stephanie Howe, Olivia Lasche, Justin Lee, Anirudh Shrinivason, Jennifer Tracey

**Published**: 2025-03-18 18:03:49

**PDF URL**: [http://arxiv.org/pdf/2503.14603v1](http://arxiv.org/pdf/2503.14603v1)

## Abstract
Building high-quality large language models (LLMs) for enterprise Arabic
applications remains challenging due to the limited availability of digitized
Arabic data. In this work, we present a data synthesis and refinement strategy
to help address this problem, namely, by leveraging synthetic data generation
and human-in-the-loop annotation to expand our Arabic training corpus. We
further present our iterative post training recipe that is essential to
achieving state-of-the-art performance in aligning the model with human
preferences, a critical aspect to enterprise use cases. The culmination of this
effort is the release of a small, 7B, open-weight model that outperforms
similarly sized peers in head-to-head comparisons and on Arabic-focused
benchmarks covering cultural knowledge, instruction following, RAG, and
contextual faithfulness.

## Full Text


<!-- PDF content starts -->

Command R7B Arabic: A Small, Enterprise Focused, Multilingual, and
Culturally Aware Arabic LLM
Yazeed Alnumay*, Alexandre Barbet*, Anna Bialas*, William Darling*,
Shaan Desai*,Joan Devassy*,Kyle Duffy*,Stephanie Howe*,
Olivia Lasche*,Justin Lee*,Anirudh Shrinivason*,Jennifer Tracey*
Cohere
Abstract
Building high-quality large language models
(LLMs) for enterprise Arabic applications re-
mains challenging due to the limited availabil-
ity of digitized Arabic data. In this work, we
present a data synthesis and refinement strategy
to help address this problem, namely, by lever-
aging synthetic data generation and human-in-
the-loop annotation to expand our Arabic train-
ing corpus. We further present our iterative
post training recipe that is essential to achiev-
ing state-of-the-art performance in aligning the
model with human preferences, a critical aspect
to enterprise use cases. The culmination of this
effort is the release of a small, 7B, open-weight
model that outperforms similarly sized peers
in head-to-head comparisons and on Arabic-
focused benchmarks covering cultural knowl-
edge, instruction following, RAG, and contex-
tual faithfulness.
1 Introduction
Multilingual language models are evolving rapidly
(Huang et al., 2024b), yet specific languages and
capabilities remain underdeveloped, particularly
in enterprise applications. While state-of-the-art
models continue to improve, they often struggle to
adapt to linguistic and professional needs in lan-
guages like Arabic (Gabriel Nicholas, 2023). This
challenge becomes even more pronounced when
additional constraints are introduced: the need to
keep the model small to ensure accessibility even
with limited resources, overcoming data scarcity,
and accounting for linguistic nuances that do not
translate well from English, all the while priori-
tizing rapid iteration to stay aligned with the fast-
moving market. To address these issues, we de-
veloped a post-training approach that efficiently
tailors cutting-edge models to specialized capabil-
ities. This report outlines our methodology and
*Equal contribution. Authors appear in alphabetical order
by second name.findings, offering insights into adapting LLMs for
language-specific and professional domains.
2 Related Work
With the recent rapid development in LLMs (Zhao
et al., 2024), some focus was placed on improving
model multilingualism through second language ac-
quisition techniques (Huang et al., 2024b). These
techniques aim to circumvent data scarcity in lan-
guages other than English by adding other language
capabilities to English models, which is more data
efficient. For instance, the Llama 3 family of
models adds a final pretraining stage by adding
multilingual pretraining data mixed with English
(Grattafiori et al., 2024). These techniques have
been applied to Arabic-centric models, such as AL-
LaM (Bari et al., 2025), Jais (Sengupta et al., 2023;
Inception, 2024), AceGPT (Huang et al., 2024a;
Zhu et al., 2024; Liang et al., 2024), and Fanar
(Fanar Team et al., 2025). These projects primar-
ily focused on pretraining data mixture, staging,
and tokenizer innovations, including vocabulary
expansion (ALLaM), iterative vocabulary expan-
sion (AceGPT), and morphology-based tokeniza-
tion (Fanar). While they contribute strong founda-
tional models for the community, they do not offer
computationally efficient post-training methods.
Post-training has become essential for building
robust models (Wei et al., 2022; Kumar et al., 2025;
Ouyang et al., 2022). Many research labs have con-
tributed to the open-source community by docu-
menting modern post-training techniques. Notable
examples include Tülu 3 (Lambert et al., 2025),
which provides a comprehensive overview of gen-
eral post-training methods, and Aya Expanse (Dang
et al., 2024), which focuses on multilingual adapta-
tion.
Our work builds on these efforts by developing a
systematic, iterative, and comprehensive approach
to efficiently adapt LLMs for languages. Specifi-
1arXiv:2503.14603v1  [cs.CL]  18 Mar 2025

Command R7B Gemma 2 9B Llama 3.1 8B Mistral 8B Jais 30B Qwen 2 5.7B
Competitor model0255075100Percentage (%) 58.741.3
56.943.1
84.815.2
66.933.1
84.215.8
61.238.8Command R7B Arabic win Competitor winFigure 1: Evaluations on enterprise usability factors (mArenaHard, described in Section 4). Auto win-rates on
Arabic version of LMSYS Arena "Hard" human preference tasks (Dang et al., 2024). Command R7B Arabic
outperforms all listed similarly-sized models.
cally, we leverage iterative tuning (Grattafiori et al.,
2024) methods that rely on best-of-N sampling to
generate instruction and preference data via auto-
mated reward models or human preference (Yuan
et al., 2024). We also further reduce compute re-
quirements by incorporating model merging tech-
niques (Goddard et al., 2024; Yang et al., 2024).
3 Methods
Our training procedure is illustrated in Figure 2.
We start by selecting a strong starting model (Sec-
tion 3.1), on which we perform three distinct train-
ing phases: (i)supervised fine-tuning (SFT) (Wei
et al., 2022), for which we employ iterative dataset
refinement techniques (Sections 3.2 and 3.3), (ii)
off-policy (offline) preference tuning, and (iii)itera-
tive preference tuning. The latter two are described
in Section 3.4. After each training phase, we merge
expert models into a single general model (Sec-
tion 3.5).
3.1 Base Model Selection
As a starting checkpoint, we chose Command R7B
(Cohere, 2024) - a strong, general purpose open-
weight model already trained on a large corpus of
multilingual data, including Arabic. Our primary
objective was to reach state-of-the-art performance
in Arabic enterprise use cases while preserving
the model’s performance on other core capabilities.
Starting from an already polished checkpoint meant
we could spend more effort on our data and training
efforts that refined Arabic-specific tasks.3.2 Multilingual Arbitrage for Capability
Enhancement
Previous work by Aya (Odumakinde et al., 2024)
has demonstrated that synthetic data generation is
crucial for achieving state-of-the-art performance,
and this is especially true for domains with lim-
ited data availability such as Arabic. However, a
key challenge when training Arabic LLMs is the
distinctive difference between Arabic and English.
Not only do these languages differ in syntax and
morphology, but there are also variations in cultural
and contextual nuances that make literal translation
challenging. For example, lexical control tasks
such as length adherence and structured genera-
tion are awkward or nonsensical when translated
to Arabic.
To address this, we implemented a human-in-
the-loop approach:
•We collaborated with expert annotators to
translate IFEval (Zhou et al., 2023) instruc-
tions into Arabic. Additionally, we augmented
the set with two instructions specific to the
Arabic language: “add Ndiacritics to the re-
sponse” and “use a specific grammatical verb
to start sentences”. This ensured better align-
ment with Arabic linguistic and cultural nu-
ances.
•These instructions were used as seeds to
synthetically generate instruction following
prompts in Arabic and subsequently the corre-
sponding completions.
2

Base ModelSFT Expert (1)
Math
SFT Expert (N)
Inst. FollowingSFT Mer geOff-policy Pref.
Expert (1)
Off-policy Pref.
Expert (N)Off-policy Pref.
MergeOn-policy Pref.
Expert (1)
On-policy Pref.
Expert (N)On-policy Pref.
MergeFigure 2: Outline of Command R7B Arabic’s training processes with three training stages, each training multiple
experts that are merged into a single general model. For instance, in the SFT stage, multiple SFT expert models are
trained to excel in specific domains, such as mathematics or instruction following. These experts are subsequently
merged to create a generalist SFT model via parameter-wise linear interpolation of the experts’ weights.
•In accordance with the work done in Aya’s
Multilingual Arbitrage (Odumakinde et al.,
2024), we scored and filtered completions us-
ing a reward model, a panel of LLM judges
for Arabic natural language quality, and max
reward difference for preference pair dataset
creation.
This targeted approach ensured that the model
learned to follow instructions naturally in Arabic,
which is apparent in arena style win-rates where our
model is consistently favored over other competitor
models, as shown in Figure 1.
3.3 Dataset Curation and Iterative Supervised
Refinement
Supervised Fine-
TuningEvaluationBase Data
Mixture
(Fixed)
Active Dataset
(Mutable)Yes NoImproves
Model?Multilingual
ArbitrageUse Expert
Figure 3: Flowchart for our iterative supervised refine-
ment approach. It ensures that all datasets used improve
targeted model performance by mixing a base data mix-
ture with a targeted dataset that is iteratively improved
via multilingual arbitrage.
The availability of high-quality Arabic datasets
is a well-documented challenge (Gabriel Nicholas,
2023). We aimed to incorporate both publicly avail-
able datasets, including ArMATH (Alghamdi et al.,
2022), ArabicaQA (Abdallah et al., 2024), and
synthetically generated datasets, while enforcing a
high-quality data standard. With this in mind, we
defined the Iterative Supervised Refinement during
Supervised Fine-Tuning (SFT) training phase as a
process to optimize our dataset composition. The
steps are illustrated in Figure 3 and are as follows:
1.Define a base data mix consisting of high-
quality instruction-tuning data.2.For each new dataset in consideration, add
it to the base data mixture and fine-tune the
model.
3.Evaluate the resulting model using a bench-
mark evaluation harness to measure the im-
pact of the new dataset.
4.If the dataset improves performance in any
critical capability, retain it for the next itera-
tion.
5.If no improvement was observed, apply Multi-
lingual Arbitrage, refining the prompts before
re-running the process.
This approach enabled us to design an opti-
mal dataset mixture that maximized the model’s
instruction-following capabilities while maintain-
ing a high standard for data quality.
3.4 Preference Tuning for Final Model
Optimization
Since we initialized from a strong Command R7B
model, it was essential to ensure that enhancements
in Arabic did not degrade performance on other
benchmarks. Similar to the methodology described
by Aya (Üstün et al., 2024), we used two stages
of preference tuning as final polishing steps to im-
prove model performance and align it with human
preferences. In the first phase, we performed offline
preference training on general preference datasets
to refine the model’s conversational fluency. In
the second phase, we ran iterative preference train-
ing, incorporating an Arabic-translated reasoning
and math-focused dataset (Alghamdi et al., 2022),
which proved particularly beneficial for maintain-
ing high performance across diverse enterprise use
cases. Both preference tuning stages utilize the di-
rect preference optimization (DPO) (Rafailov et al.,
2024) algorithm.
3

Benchmark R7B ArabicR7B Gemma 9B Llama 3.1 8B Qwen 2.5 7B Ministral 8B
(Cohere, 2024) (Gemma Team et al., 2024) (Grattafiori et al., 2024) (Yang et al., 2025) (Mistral, 2024)
AlGhafa-Native 82.2 81.5 81.3 80.1 80.2 76.6
ArabicMMLU 60.9 59.7 62.4 56.6 61.2 53.6
IFEval AR 69.0 57.8 67.8 48.4 62.4 49.3
TyDIQA-GoldP Arabic 83.0 79.9 76.4 65.9 60.9 57.7
FaithEval Arabic 51.6 49.9 47.0 40.9 49.9 25.5
Average 69.3 65.8 67.0 58.4 62.9 52.5
Table 1: Full performance comparison against competitor models on Arabic-specific benchmarks. The highest score
in each row is in bold . Command R7B Arabic is best-in-class compared to similarly sized models on all Arabic
benchmarks, with the exception of ArabicMMLU.
3.5 Expert Model Merging
After completing the iterative supervised refine-
ment procedure described in Section 3.3 to create
multiple expert models from various datasets, one
path forward is to retrain a new generalist model
by combining appropriate datasets based on the
insights obtained from these experiments. How-
ever, we can eliminate computational redundancy
by merging various expert models. This is a com-
mon practice with mature frameworks (Goddard
et al., 2024). The literature lacks conclusive theo-
retical foundations for the effectiveness of model
merging, but extensive experimentation has shown
it is a successful strategy in practice (Yang et al.,
2024).
To reduce the expert merge search space, we
only considered linear merges (Utans, 1996) of
the expert models. We tested several weighting
schemes based on the importance of each capability
and the size of each expert’s training data. In the
end, our best model was obtained by assigning
equal weight to each expert.
In practice, model merging reduces computa-
tional cost. However, it complicates replication
and adds an additional source of potential errors.
4 Results
4.1 Arabic Language
To measure the performance of various models in
Arabic language generation and understanding, we
utilized the following evaluation suite:
•IFEval AR : An internal Arabic translation
of the original English dataset (Zhou et al.,
2023) with 541 test samples. It measures a
model’s precise instruction following ability,
with instructions such as “use at least 300
words” or “do not use commas.”•AlGhafa-Native : The subset1of AlGhafa (Al-
mazrouei et al., 2023) tasks which were cu-
rated by native Arabic speakers, which encap-
sulates the following:
–MCQ Exams AR (562 samples)
(Hardalov et al., 2020).
–Belebele AR Dialects (5,400 samples)
and Belebele AR MSA (900 samples)
(Bandarkar et al., 2024).
–AraFacts balanced (80 samples)
(Sheikh Ali et al., 2021).
–SOQAL (155 samples) (Mozannar et al.,
2019).
–XGLUE (155 samples) (Liang et al.,
2020).
–Rating sentiment no neutral (8,000 sam-
ples) and rating sentiment (6,000 sam-
ples) from the HARD-Arabic-Dataset
(Elnagar et al., 2018).
–Sentiment (1,725 samples) (Abu Farha
et al., 2021).
We report the unweighted average percentage
performance across all tasks.
•TyDiQA-GoldP Arabic : The 921 samples in
Arabic from the original TyDiQA (Clark et al.,
2020) golden passage (GoldP) secondary task,
in which models are provided with a question
and a single passage that contains the ques-
tion’s answer. Models are prompted to deter-
mine the substring in the passage that answers
the question.
•ArabicMMLU (Koto et al., 2024): Inspired
by the original MMLU (Hendrycks et al.,
1https://huggingface.co/datasets/OALL/AlGhafa-Arabic-
LLM-Benchmark-Native
4

Benchmark R7B ArabicR7B Gemma 9B Llama 3.1 8B Qwen 2.5 7B Ministral 8B
(Cohere, 2024) (Gemma Team et al., 2024) (Grattafiori et al., 2024) (Yang et al., 2025) (Mistral, 2024)
BBH (Suzgun et al., 2022) 36.2 36.0 42.1 29.9 34.9 25.8
MuSR (Sprague et al., 2024) 11.9 10.2 9.7 8.4 8.5 8.4
GPQA (Rein et al., 2023) 7.9 7.8 14.8 2.4 5.5 4.5
MMLU Pro (Wang et al., 2024) 29.4 28.6 32.0 30.7 36.5 30.7
IfEval (Zhou et al., 2023) 83.3 77.1 74.4 78.6 75.9 59.0
MATH* (Hendrycks et al., 2021b) 19.6 29.9 19.1 19.3 50.0 19.6
Average 31.4 31.6 32.1 28.2 35.2 22.0
* The MATH benchmark used in this leaderboard changed in early January due to a DMCA takedown notice for the original benchmark.
Table 2: Performance comparison of R7B Arabic against similarly sized models on multiple benchmarks. The
highest score in each row is in bold . Command R7B Arabic retains most of the general and English capabilities of
its base model, Command R7B, as indicated by the similar average scores.
2021a) in English, ArabicMMLU is a collec-
tion of 14,575 native Arabic multiple choice
questions focusing on knowledge and reason-
ing. It covers 40 tasks at various education lev-
els (elementary to college) and regions (North
Africa, Levant, and Gulf).
•FaithEval Arabic : An internal Arabic trans-
lation of a 500 sample subset of the original
English dataset (Ming et al., 2024). It mea-
sures the model’s RAG performance when
provided with unanswerable, inconsistent, or
counterfactual contexts.
•Multilingual ArenaHard (Dang et al., 2024):
A machine translation of 500 questions from
the original English LMArena (formerly LM-
SYS) Arena-Hard-Auto (Li et al., 2024)
prompts into various other languages. We
limit our evaluation to the Arabic subset. The
evaluation uses GPT-4o as a judge to compare
completions from two different models.
Table 1 shows results compared to other models
in the same size category. The Command R7B Ara-
bic model outperforms all baselines across key Ara-
bic benchmarks, achieving an average score of 69.3,
surpassing Command R7B (65.8) and Gemma 9B
(67.0). It performs at the top of its size class in
the following benchmarks: Cultural Knowledge
(AlGhafa-Native), Instruction Following (IFEval
AR) validating our human-in-the-loop data strat-
egy, RAG Question Answering (TyDiQA-GoldP
Arabic), and RAG Faithfulness (FaithEval Arabic).
In General Knowledge (ArabicMMLU), Command
R7B Arabic scores third, while staying competitive
with Gemma 9B and Qwen 2.7.
4.2 General Capabilities
Retaining general capabilities is essential for the
model to be helpful in enterprise settings. We thor-oughly measured our model’s performance and
present the results of the standardized Hugging
Face Open LLM Leaderboard benchmarks (Four-
rier et al., 2024; Gao et al., 2021). Table 2 shows
that our model excels in IfEval and MuSR, achiev-
ing the highest scores among similarly sized mod-
els. Notably, it outperforms the initial checkpoint
on all benchmarks except for MATH, possibly due
to the change in methodology.
These benchmark results (Table 1 and Table 2),
coupled with auto win-rate data (Figure 1), vali-
date that our approach effectively enhances Arabic
language capabilities while maintaining robust per-
formance in enterprise applications.
5 Conclusion
In this work, we rapidly iterated to develop Com-
mand R7B Arabic, a small, yet competent Ara-
bic LLM optimized for enterprise applications.
By leveraging synthetic data generation, multilin-
gual arbitrage, and human-in-the-loop interven-
tions, we significantly improved instruction follow-
ing, retrieval-augmented generation (RAG), and
question answering capabilities in Arabic. How-
ever, transferring knowledge from English-centric
datasets to Arabic remains an open challenge. Fu-
ture work should explore more effective adaptation
strategies, ensuring higher linguistic and factual
alignment across languages.
6 Limitations
Our work focuses on Modern Standard Arabic
(MSA), which is widely used in formal and profes-
sional settings but differs significantly from spoken
dialects across the Arabic-speaking world. While
MSA provides a strong foundation for enterprise
applications, real world use cases often involve
dialectal Arabic, which varies by region and con-
5

text. Future work should explore dialect adaptation
strategies to improve robustness across diverse Ara-
bic varieties.
We adapted Faithfulness (FaithEval Arabic),
Question Answering (TyDi QA Arabic), and
Instruction Following (IFEval AR) to measure
enterprise-relevant capabilities. Still, these bench-
marks remain proxies rather than direct tests of
real-world deployment challenges. The effective-
ness of our model in enterprise workflows can only
be fully validated through real-world deployment
and user feedback.
7 Acknowledgments
This work was a collaboration between many teams
in Cohere. We would like to particularly ac-
knowledge the following people who supported
the project through advice and maintenance of our
core infrastructure:
Modeling Team: Théo Dehaze, Jesse Willman,
Lewis Stott, Florian Strub, Jay Alammar, Matthias
Gallé, Samuel Cahyawijaya, Alexandre Bérard,
Wei-Yin Ko, Kocmi Tom, Dennis Aumiller, Nathan
Grinsztajn, Phil Blunsom, Jon Ander Campos, Yi
Chern Tan, Sander Land, Nithya Govindarajan,
Nick Jakobi, Adrien Morisot, Olivia Markham;
C4AI: Sungjin Hong, Alejandro Salamanca,
Marzieh Fadaee, Ahmet Üstün, Sara Hooker;
Infrastructure: Cécile Robert-Michon, Jessica
Xie, Adi Bongale, Ace Eldeib, Sudip Roy, Manoj
Govindassamy, Maxime Brunet, Jeremy Pekmez,
Terrence Zhao, Renjie Huang;
Applied ML Team: Neeral Beladia, Gokce Ke-
skin, Utsav Garg, Jason Jung, Hemangani Nagara-
jan, Sanal Shivaprasad, Sam Passaglia, Edmond
Wen, Trushant Kalyanpur, Vivek Muppalla, Evren
Tumer, Harri Bell-Thomas;
Annotators: Arwa Alaya, Noha Shehata , Eyas
Shanaah , Abdullah Omran, Nermeen Isaac, Izzat
Homsi, Mahmoud Mansour, Mayar Soliman, Israr
Wahid, Vanessa Choueiry, Mona Knobloch, Fatima
Zahra Zyad;
Annotator Operations: Claire Cheng, Trisha
Starostina, Brenda Malacara Lopez;
Leadership: Aidan Gomez, Martin Kon, Saurabh
Baji, Phil Blunsom;
External partners: Neha Sengupta, Ali El Filali.References
Abdelrahman Abdallah, Mahmoud Kasem, Mahmoud
Abdalla, Mohamed Mahmoud, Mohamed Elkasaby,
Yasser Elbendary, and Adam Jatowt. 2024. Arabi-
caqa: A comprehensive dataset for arabic question
answering. Preprint , arXiv:2403.17848.
Ibrahim Abu Farha, Wajdi Zaghouani, and Walid Magdy.
2021. Overview of the WANLP 2021 shared task
on sarcasm and sentiment detection in Arabic. In
Proceedings of the Sixth Arabic Natural Language
Processing Workshop , pages 296–305, Kyiv, Ukraine
(Virtual). Association for Computational Linguistics.
Reem Alghamdi, Zhenwen Liang, and Xiangliang
Zhang. 2022. ArMATH: a dataset for solving Arabic
math word problems. In Proceedings of the Thir-
teenth Language Resources and Evaluation Confer-
ence, pages 351–362, Marseille, France. European
Language Resources Association.
Ebtesam Almazrouei, Ruxandra Cojocaru, Michele
Baldo, Quentin Malartic, Hamza Alobeidli, Daniele
Mazzotta, Guilherme Penedo, Giulia Campesan, Mu-
gariya Farooq, Maitha Alhammadi, Julien Launay,
and Badreddine Noune. 2023. AlGhafa evaluation
benchmark for Arabic language models. In Proceed-
ings of ArabicNLP 2023 , pages 244–275, Singapore
(Hybrid). Association for Computational Linguistics.
Lucas Bandarkar, Davis Liang, Benjamin Muller, Mikel
Artetxe, Satya Narayan Shukla, Donald Husa, Naman
Goyal, Abhinandan Krishnan, Luke Zettlemoyer, and
Madian Khabsa. 2024. The belebele benchmark: a
parallel reading comprehension dataset in 122 lan-
guage variants. In Proceedings of the 62nd Annual
Meeting of the Association for Computational Lin-
guistics (Volume 1: Long Papers) , page 749–775.
Association for Computational Linguistics.
M Saiful Bari, Yazeed Alnumay, Norah A. Alzahrani,
Nouf M. Alotaibi, Hisham Abdullah Alyahya, Sultan
AlRashed, Faisal Abdulrahman Mirza, Shaykhah Z.
Alsubaie, Hassan A. Alahmed, Ghadah Alabdul-
jabbar, Raghad Alkhathran, Yousef Almushayqih,
Raneem Alnajim, Salman Alsubaihi, Maryam Al
Mansour, Saad Amin Hassan, Dr. Majed Alruba-
ian, Ali Alammari, Zaki Alawami, Abdulmohsen Al-
Thubaity, Ahmed Abdelali, Jeril Kuriakose, Abdal-
ghani Abujabal, Nora Al-Twairesh, Areeb Alowisheq,
and Haidar Khan. 2025. ALLam: Large language
models for arabic and english. In The Thirteenth In-
ternational Conference on Learning Representations .
Jonathan H. Clark, Eunsol Choi, Michael Collins,
Dan Garrette, Tom Kwiatkowski, Vitaly Nikolaev,
and Jennimaria Palomaki. 2020. Tydi qa: A
benchmark for information-seeking question answer-
ing in typologically diverse languages. Preprint ,
arXiv:2003.05002.
Cohere. 2024. Introducing command r7b: Fast and
efficient generative ai.
6

John Dang, Shivalika Singh, Daniel D’souza, Arash
Ahmadian, Alejandro Salamanca, Madeline Smith,
Aidan Peppin, Sungjin Hong, Manoj Govindassamy,
Terrence Zhao, Sandra Kublik, Meor Amer, Viraat
Aryabumi, Jon Ander Campos, Yi-Chern Tan, Tom
Kocmi, Florian Strub, Nathan Grinsztajn, Yannis
Flet-Berliac, Acyr Locatelli, Hangyu Lin, Dwarak
Talupuru, Bharat Venkitesh, David Cairuz, Bowen
Yang, Tim Chung, Wei-Yin Ko, Sylvie Shang Shi,
Amir Shukayev, Sammie Bae, Aleksandra Piktus, Ro-
man Castagné, Felipe Cruz-Salinas, Eddie Kim, Lu-
cas Crawhall-Stein, Adrien Morisot, Sudip Roy, Phil
Blunsom, Ivan Zhang, Aidan Gomez, Nick Frosst,
Marzieh Fadaee, Beyza Ermis, Ahmet Üstün, and
Sara Hooker. 2024. Aya expanse: Combining re-
search breakthroughs for a new multilingual frontier.
Preprint , arXiv:2412.04261.
Ashraf Elnagar, Yasmin S Khalifa, and Anas Einea.
2018. Hotel arabic-reviews dataset construction for
sentiment analysis applications. Intelligent natural
language processing: Trends and applications , pages
35–52.
Fanar Team, Ummar Abbas, Mohammad Shahmeer Ah-
mad, Firoj Alam, Enes Altinisik, Ehsannedin Asgari,
Yazan Boshmaf, Sabri Boughorbel, Sanjay Chawla,
Shammur Chowdhury, Fahim Dalvi, Kareem Dar-
wish, Nadir Durrani, Mohamed Elfeky, Ahmed El-
magarmid, Mohamed Eltabakh, Masoomali Fatehkia,
Anastasios Fragkopoulos, Maram Hasanain, Majd
Hawasly, Mus’ab Husaini, Soon-Gyo Jung, Ji Kim
Lucas, Walid Magdy, Safa Messaoud, Abubakr Mo-
hamed, Tasnim Mohiuddin, Basel Mousi, Hamdy
Mubarak, Ahmad Musleh, Zan Naeem, Mourad Ouz-
zani, Dorde Popovic, Amin Sadeghi, Husrev Taha
Sencar, Mohammed Shinoy, Omar Sinan, Yifan
Zhang, Ahmed Ali, Yassine El Kheir, Xiaosong
Ma, and Chaoyi Ruan. 2025. Fanar: An arabic-
centric multimodal generative ai platform. Preprint ,
arXiv:2501.13944.
Clémentine Fourrier, Nathan Habib, Alina Lozovskaya,
Konrad Szafer, and Thomas Wolf. 2024. Open
llm leaderboard v2. https://huggingface.
co/spaces/open-llm-leaderboard/open_llm_
leaderboard .
Aliya Bhatia Gabriel Nicholas. 2023. Lost in transla-
tion: Large language models in non-english content
analysis. Center for Democracy & Technology .
Leo Gao, Jonathan Tow, Stella Biderman, Sid Black,
Anthony DiPofi, Charles Foster, Laurence Golding,
Jeffrey Hsu, Kyle McDonell, Niklas Muennighoff,
Jason Phang, Laria Reynolds, Eric Tang, Anish Thite,
Ben Wang, Kevin Wang, and Andy Zou. 2021. A
framework for few-shot language model evaluation.
Gemma Team, Morgane Riviere, Shreya Pathak,
Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupati-
raju, Léonard Hussenot, Thomas Mesnard, Bobak
Shahriari, Alexandre Ramé, Johan Ferret, Peter
Liu, Pouya Tafti, Abe Friesen, Michelle Casbon,
Sabela Ramos, Ravin Kumar, Charline Le Lan,Sammy Jerome, Anton Tsitsulin, Nino Vieillard,
Piotr Stanczyk, Sertan Girgin, Nikola Momchev,
Matt Hoffman, Shantanu Thakoor, Jean-Bastien Grill,
Behnam Neyshabur, Olivier Bachem, Alanna Wal-
ton, Aliaksei Severyn, Alicia Parrish, Aliya Ah-
mad, Allen Hutchison, Alvin Abdagic, Amanda
Carl, Amy Shen, Andy Brock, Andy Coenen, An-
thony Laforge, Antonia Paterson, Ben Bastian, Bilal
Piot, Bo Wu, Brandon Royal, Charlie Chen, Chintu
Kumar, Chris Perry, Chris Welty, Christopher A.
Choquette-Choo, Danila Sinopalnikov, David Wein-
berger, Dimple Vijaykumar, Dominika Rogozi ´nska,
Dustin Herbison, Elisa Bandy, Emma Wang, Eric
Noland, Erica Moreira, Evan Senter, Evgenii Elty-
shev, Francesco Visin, Gabriel Rasskin, Gary Wei,
Glenn Cameron, Gus Martins, Hadi Hashemi, Hanna
Klimczak-Pluci ´nska, Harleen Batra, Harsh Dhand,
Ivan Nardini, Jacinda Mein, Jack Zhou, James Svens-
son, Jeff Stanway, Jetha Chan, Jin Peng Zhou, Joana
Carrasqueira, Joana Iljazi, Jocelyn Becker, Joe Fer-
nandez, Joost van Amersfoort, Josh Gordon, Josh
Lipschultz, Josh Newlan, Ju yeong Ji, Kareem Mo-
hamed, Kartikeya Badola, Kat Black, Katie Mil-
lican, Keelin McDonell, Kelvin Nguyen, Kiranbir
Sodhia, Kish Greene, Lars Lowe Sjoesund, Lau-
ren Usui, Laurent Sifre, Lena Heuermann, Leti-
cia Lago, Lilly McNealus, Livio Baldini Soares,
Logan Kilpatrick, Lucas Dixon, Luciano Martins,
Machel Reid, Manvinder Singh, Mark Iverson, Mar-
tin Görner, Mat Velloso, Mateo Wirth, Matt Davi-
dow, Matt Miller, Matthew Rahtz, Matthew Watson,
Meg Risdal, Mehran Kazemi, Michael Moynihan,
Ming Zhang, Minsuk Kahng, Minwoo Park, Mofi
Rahman, Mohit Khatwani, Natalie Dao, Nenshad
Bardoliwalla, Nesh Devanathan, Neta Dumai, Nilay
Chauhan, Oscar Wahltinez, Pankil Botarda, Parker
Barnes, Paul Barham, Paul Michel, Pengchong
Jin, Petko Georgiev, Phil Culliton, Pradeep Kup-
pala, Ramona Comanescu, Ramona Merhej, Reena
Jana, Reza Ardeshir Rokni, Rishabh Agarwal, Ryan
Mullins, Samaneh Saadat, Sara Mc Carthy, Sarah
Cogan, Sarah Perrin, Sébastien M. R. Arnold, Se-
bastian Krause, Shengyang Dai, Shruti Garg, Shruti
Sheth, Sue Ronstrom, Susan Chan, Timothy Jor-
dan, Ting Yu, Tom Eccles, Tom Hennigan, Tomas
Kocisky, Tulsee Doshi, Vihan Jain, Vikas Yadav,
Vilobh Meshram, Vishal Dharmadhikari, Warren
Barkley, Wei Wei, Wenming Ye, Woohyun Han,
Woosuk Kwon, Xiang Xu, Zhe Shen, Zhitao Gong,
Zichuan Wei, Victor Cotruta, Phoebe Kirk, Anand
Rao, Minh Giang, Ludovic Peran, Tris Warkentin,
Eli Collins, Joelle Barral, Zoubin Ghahramani, Raia
Hadsell, D. Sculley, Jeanine Banks, Anca Dragan,
Slav Petrov, Oriol Vinyals, Jeff Dean, Demis Hass-
abis, Koray Kavukcuoglu, Clement Farabet, Elena
Buchatskaya, Sebastian Borgeaud, Noah Fiedel, Ar-
mand Joulin, Kathleen Kenealy, Robert Dadashi,
and Alek Andreev. 2024. Gemma 2: Improving
open language models at a practical size. Preprint ,
arXiv:2408.00118.
Charles Goddard, Shamane Siriwardhana, Malikeh
Ehghaghi, Luke Meyers, Vladimir Karpukhin, Brian
Benedict, Mark McQuade, and Jacob Solawetz. 2024.
7

Arcee’s mergekit: A toolkit for merging large lan-
guage models. In Proceedings of the 2024 Confer-
ence on Empirical Methods in Natural Language
Processing: Industry Track , pages 477–485.
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad Al-
Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten,
Alex Vaughan, et al. 2024. The llama 3 herd of mod-
els.arXiv preprint arXiv:2407.21783 .
Momchil Hardalov, Todor Mihaylov, Dimitrina
Zlatkova, Yoan Dinkov, Ivan Koychev, and Preslav
Nakov. 2020. Exams: A multi-subject high
school examinations dataset for cross-lingual and
multilingual question answering. arXiv preprint
arXiv:2011.03080 .
Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou,
Mantas Mazeika, Dawn Song, and Jacob Steinhardt.
2021a. Measuring massive multitask language under-
standing. Preprint , arXiv:2009.03300.
Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul
Arora, Steven Basart, Eric Tang, Dawn Song, and
Jacob Steinhardt. 2021b. Measuring mathematical
problem solving with the math dataset. Preprint ,
arXiv:2103.03874.
Huang Huang, Fei Yu, Jianqing Zhu, Xuening Sun, Hao
Cheng, Dingjie Song, Zhihong Chen, Abdulmohsen
Alharthi, Bang An, Juncai He, Ziche Liu, Zhiyi
Zhang, Junying Chen, Jianquan Li, Benyou Wang,
Lian Zhang, Ruoyu Sun, Xiang Wan, Haizhou Li,
and Jinchao Xu. 2024a. Acegpt, localizing large lan-
guage models in arabic. Preprint , arXiv:2309.12053.
Kaiyu Huang, Fengran Mo, Xinyu Zhang, Hongliang
Li, You Li, Yuanchi Zhang, Weijian Yi, Yulong Mao,
Jinchen Liu, Yuzhuang Xu, et al. 2024b. A sur-
vey on large language models with multilingualism:
Recent advances and new frontiers. arXiv preprint
arXiv:2405.10936 .
Inception. 2024. Jais family model card. Hugging Face .
Fajri Koto, Haonan Li, Sara Shatanawi, Jad Doughman,
Abdelrahman Boda Sadallah, Aisha Alraeesi, Khalid
Almubarak, Zaid Alyafeai, Neha Sengupta, Shady
Shehata, Nizar Habash, Preslav Nakov, and Timothy
Baldwin. 2024. ArabicMMLU: Assessing massive
multitask language understanding in arabic. In Find-
ings of the Association for Computational Linguistics:
ACL 2024 .
Komal Kumar, Tajamul Ashraf, Omkar Thawakar,
Rao Muhammad Anwer, Hisham Cholakkal,
Mubarak Shah, Ming-Hsuan Yang, Phillip H. S. Torr,
Salman Khan, and Fahad Shahbaz Khan. 2025. Llm
post-training: A deep dive into reasoning large lan-
guage models. Preprint , arXiv:2502.21321.
Nathan Lambert, Jacob Morrison, Valentina Pyatkin,
Shengyi Huang, Hamish Ivison, Faeze Brahman,
Lester James V . Miranda, Alisa Liu, Nouha Dziri,
Shane Lyu, Yuling Gu, Saumya Malik, VictoriaGraf, Jena D. Hwang, Jiangjiang Yang, Ronan Le
Bras, Oyvind Tafjord, Chris Wilhelm, Luca Soldaini,
Noah A. Smith, Yizhong Wang, Pradeep Dasigi, and
Hannaneh Hajishirzi. 2025. Tulu 3: Pushing fron-
tiers in open language model post-training. Preprint ,
arXiv:2411.15124.
Tianle Li, Wei-Lin Chiang, Evan Frick, Lisa Dunlap,
Tianhao Wu, Banghua Zhu, Joseph E. Gonzalez, and
Ion Stoica. 2024. From crowdsourced data to high-
quality benchmarks: Arena-hard and benchbuilder
pipeline. Preprint , arXiv:2406.11939.
Juhao Liang, Zhenyang Cai, Jianqing Zhu, Huang
Huang, Kewei Zong, Bang An, Mosen Alharthi, Jun-
cai He, Lian Zhang, Haizhou Li, Benyou Wang, and
Jinchao Xu. 2024. Alignment at pre-training! to-
wards native alignment for arabic LLMs. In The
Thirty-eighth Annual Conference on Neural Informa-
tion Processing Systems .
Yaobo Liang, Nan Duan, Yeyun Gong, Ning Wu, Fenfei
Guo, Weizhen Qi, Ming Gong, Linjun Shou, Daxin
Jiang, Guihong Cao, Xiaodong Fan, Ruofei Zhang,
Rahul Agrawal, Edward Cui, Sining Wei, Taroon
Bharti, Ying Qiao, Jiun-Hung Chen, Winnie Wu,
Shuguang Liu, Fan Yang, Daniel Campos, Rangan
Majumder, and Ming Zhou. 2020. XGLUE: A new
benchmark dataset for cross-lingual pre-training, un-
derstanding and generation. In Proceedings of the
2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP) , pages 6008–6018,
Online. Association for Computational Linguistics.
Yifei Ming, Senthil Purushwalkam, Shrey Pandit,
Zixuan Ke, Xuan-Phi Nguyen, Caiming Xiong,
and Shafiq Joty. 2024. Faitheval: Can your
language model stay faithful to context, even if
"the moon is made of marshmallows". Preprint ,
arXiv:2410.03727.
Mistral. 2024. Ministral 8b instruct model card. Hug-
ging Face .
Hussein Mozannar, Karl El Hajal, Elie Maamary, and
Hazem Hajj. 2019. Neural arabic question answering.
Preprint , arXiv:1906.05394.
Ayomide Odumakinde, Daniel D’souza, Pat Verga,
Beyza Ermis, and Sara Hooker. 2024. Multilingual
arbitrage: Optimizing data pools to accelerate multi-
lingual progress. Preprint , arXiv:2408.14960.
Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Car-
roll L. Wainwright, Pamela Mishkin, Chong Zhang,
Sandhini Agarwal, Katarina Slama, Alex Ray, John
Schulman, Jacob Hilton, Fraser Kelton, Luke Miller,
Maddie Simens, Amanda Askell, Peter Welinder,
Paul Christiano, Jan Leike, and Ryan Lowe. 2022.
Training language models to follow instructions with
human feedback. Preprint , arXiv:2203.02155.
Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano
Ermon, Christopher D. Manning, and Chelsea Finn.
2024. Direct preference optimization: Your lan-
guage model is secretly a reward model. Preprint ,
arXiv:2305.18290.
8

David Rein, Betty Li Hou, Asa Cooper Stickland,
Jackson Petty, Richard Yuanzhe Pang, Julien Di-
rani, Julian Michael, and Samuel R. Bowman. 2023.
Gpqa: A graduate-level google-proof q&a bench-
mark. Preprint , arXiv:2311.12022.
Neha Sengupta, Sunil Kumar Sahu, Bokang Jia,
Satheesh Katipomu, Haonan Li, Fajri Koto, William
Marshall, Gurpreet Gosal, Cynthia Liu, Zhiming
Chen, Osama Mohammed Afzal, Samta Kamboj,
Onkar Pandit, Rahul Pal, Lalit Pradhan, Zain Muham-
mad Mujahid, Massa Baali, Xudong Han, Son-
dos Mahmoud Bsharat, Alham Fikri Aji, Zhiqiang
Shen, Zhengzhong Liu, Natalia Vassilieva, Joel Hes-
tness, Andy Hock, Andrew Feldman, Jonathan Lee,
Andrew Jackson, Hector Xuguang Ren, Preslav
Nakov, Timothy Baldwin, and Eric Xing. 2023.
Jais and jais-chat: Arabic-centric foundation and
instruction-tuned open generative large language
models. Preprint , arXiv:2308.16149.
Zien Sheikh Ali, Watheq Mansour, Tamer Elsayed, and
Abdulaziz Al-Ali. 2021. AraFacts: The first large
Arabic dataset of naturally occurring claims. In Pro-
ceedings of the Sixth Arabic Natural Language Pro-
cessing Workshop , pages 231–236, Kyiv, Ukraine
(Virtual). Association for Computational Linguistics.
Zayne Sprague, Xi Ye, Kaj Bostrom, Swarat Chaudhuri,
and Greg Durrett. 2024. Musr: Testing the limits
of chain-of-thought with multistep soft reasoning.
Preprint , arXiv:2310.16049.
Mirac Suzgun, Nathan Scales, Nathanael Schärli, Se-
bastian Gehrmann, Yi Tay, Hyung Won Chung,
Aakanksha Chowdhery, Quoc V . Le, Ed H. Chi,
Denny Zhou, and Jason Wei. 2022. Challenging
big-bench tasks and whether chain-of-thought can
solve them. Preprint , arXiv:2210.09261.
Ahmet Üstün, Viraat Aryabumi, Zheng Yong, Wei-Yin
Ko, Daniel D’souza, Gbemileke Onilude, Neel Bhan-
dari, Shivalika Singh, Hui-Lee Ooi, Amr Kayid, Fred-
die Vargus, Phil Blunsom, Shayne Longpre, Niklas
Muennighoff, Marzieh Fadaee, Julia Kreutzer, and
Sara Hooker. 2024. Aya model: An instruction fine-
tuned open-access multilingual language model. In
Proceedings of the 62nd Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers) , pages 15894–15939, Bangkok, Thai-
land. Association for Computational Linguistics.
Joachim Utans. 1996. Weight averaging for neural
networks and local resampling schemes. In Proc.
AAAI-96 Workshop on Integrating Multiple Learned
Models. AAAI Press , pages 133–138. Citeseer.
Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni,
Abhranil Chandra, Shiguang Guo, Weiming Ren,
Aaran Arulraj, Xuan He, Ziyan Jiang, Tianle Li, Max
Ku, Kai Wang, Alex Zhuang, Rongqi Fan, Xiang Yue,
and Wenhu Chen. 2024. Mmlu-pro: A more robust
and challenging multi-task language understanding
benchmark. Preprint , arXiv:2406.01574.Jason Wei, Maarten Bosma, Vincent Y . Zhao, Kelvin
Guu, Adams Wei Yu, Brian Lester, Nan Du, An-
drew M. Dai, and Quoc V . Le. 2022. Finetuned
language models are zero-shot learners. Preprint ,
arXiv:2109.01652.
An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui,
Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu,
Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jian-
hong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang,
Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu,
Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng
Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tian-
hao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren,
Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang,
Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and
Zihan Qiu. 2025. Qwen2.5 technical report. Preprint ,
arXiv:2412.15115.
Enneng Yang, Li Shen, Guibing Guo, Xingwei Wang,
Xiaochun Cao, Jie Zhang, and Dacheng Tao. 2024.
Model merging in llms, mllms, and beyond: Methods,
theories, applications and opportunities. Preprint ,
arXiv:2408.07666.
Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho,
Xian Li, Sainbayar Sukhbaatar, Jing Xu, and Ja-
son Weston. 2024. Self-rewarding language models.
Preprint , arXiv:2401.10020.
Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang,
Xiaolei Wang, Yupeng Hou, Yingqian Min, Be-
ichen Zhang, Junjie Zhang, Zican Dong, Yifan Du,
Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao
Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang
Liu, Peiyu Liu, Jian-Yun Nie, and Ji-Rong Wen.
2024. A survey of large language models. Preprint ,
arXiv:2303.18223.
Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha
Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and
Le Hou. 2023. Instruction-following evaluation for
large language models. Preprint , arXiv:2311.07911.
Jianqing Zhu, Huang Huang, Zhihang Lin, Juhao
Liang, Zhengyang Tang, Khalid Almubarak, Abdul-
mohsen Alharthik, Bang An, Juncai He, Xiangbo
Wu, Fei Yu, Junying Chen, Zhuoheng Ma, Yuhao
Du, He Zhang, Emad A. Alghamdi, Lian Zhang,
Ruoyu Sun, Haizhou Li, Benyou Wang, and Jinchao
Xu. 2024. Second language (arabic) acquisition of
llms via progressive vocabulary expansion. Preprint ,
arXiv:2412.12310.
9