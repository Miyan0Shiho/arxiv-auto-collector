# HEAD-QA v2: Expanding a Healthcare Benchmark for Reasoning

**Authors**: Alexis Correa-Guillén, Carlos Gómez-Rodríguez, David Vilares

**Published**: 2025-11-19 11:31:32

**PDF URL**: [https://arxiv.org/pdf/2511.15355v1](https://arxiv.org/pdf/2511.15355v1)

## Abstract
We introduce HEAD-QA v2, an expanded and updated version of a Spanish/English healthcare multiple-choice reasoning dataset originally released by Vilares and Gómez-Rodríguez (2019). The update responds to the growing need for high-quality datasets that capture the linguistic and conceptual complexity of healthcare reasoning. We extend the dataset to over 12,000 questions from ten years of Spanish professional exams, benchmark several open-source LLMs using prompting, RAG, and probability-based answer selection, and provide additional multilingual versions to support future work. Results indicate that performance is mainly driven by model scale and intrinsic reasoning ability, with complex inference strategies obtaining limited gains. Together, these results establish HEAD-QA v2 as a reliable resource for advancing research on biomedical reasoning and model improvement.

## Full Text


<!-- PDF content starts -->

HEAD-QA v2: Expanding a Healthcare Benchmark for Reasoning
Alexis Correa-Guillén, Carlos Gómez-Rodríguez, David Vilares
Universidade da Coruña, CITIC
Departamento de Ciencias de la Computación y Tecnologías de la Información
Campus de Elviña s/n 15071, A Coruña, Spain
{alexis.cguillen@udc.es, carlos.gomez, david.vilares}@udc.es
Abstract
We introduce HEAD-QA v2, an expanded and updated version of a Spanish/English healthcare multiple-choice
reasoning dataset originally released by Vilares and Gómez-Rodríguez (2019). The update responds to the
growing need for high-quality datasets that capture the linguistic and conceptual complexity of healthcare reasoning.
We extend the dataset to over 12,000 questions from ten years of Spanish professional exams, benchmark
several open-source LLMs using prompting, RAG, and probability-based answer selection, and provide additional
multilingual versions to support future work. Results indicate that performance is mainly driven by model scale
and intrinsic reasoning ability, with complex inference strategies obtaining limited gains. Together, these results
establishHEAD-QAv2asareliableresourceforadvancingresearchonbiomedicalreasoningandmodelimprovement.
Keywords:Multi-choice question answering, LLMs, Healthcare
1. Introduction
HEAD-QA (v1) (Vilares and Gómez-Rodríguez,
2019) is a Spanish/English multiple-choice health-
caredatasetdesignedtoevaluatemodelreasoning
abilities. It comprises 6,765 questions from official
exams issued between 2013 and 2017. It was con-
ceived as a step toward more demanding bench-
marks, following the rise of early reading compre-
hension datasets such as SQuAD (Rajpurkar et al.,
2016), SNLI (Bowman et al., 2015), and the AI2
Reasoning Challenge (Clark et al., 2018), among
others, as well as the neural architectures devel-
oped for them (Kumar et al., 2016; Chen et al.,
2017). Notably, experimental results revealed that
these architectures lacked the capacity to reason
effectively about diagnostic knowledge and failed
tocapturedefinitionsanddomain-specificconcepts
essential for accurate inference, often performing
worse than simple information retrieval baselines.
More specifically, HEAD-QA consists of multiple-
choice questions modeled after Spain’s competi-
tive specialization exams (Ministerio de Sanidad
de España, 2023), which are used to evaluate and
rank graduates in fields such as medicine (MIR),
nursing (EIR), biology (BIR), chemistry (QIR), psy-
chology (PIR), and pharmacy (FIR). These highly
demanding exams require months or even years
of preparation, as their results determine both the
specialization and the training location where can-
didates complete the final 3–5 years of residency
before becoming fully qualified professionals. The
dataset has since gained notable adoption, hav-
ing been used to evaluate influential architectures
and models such as RMKV (Peng et al., 2023),
Falcon (Penedo et al., 2023) and OLMo (Groen-eveld et al., 2024), to investigate data reliability in
both open-source and proprietary systems (Elazar
et al., 2023), and to develop and assess special-
ized solutions in the medical domain (Zhang et al.,
2023; Wang et al., 2024). It has also served as a
precursor to similar medical QA datasets in other
languages, including Chinese (Li et al., 2021) and
French(Labraketal.,2022),extendingitsinfluence
on healthcare question answering research.
In the current context, the landscape of ques-
tion answering and reasoning has changed pro-
foundly with the rise of large language models
(LLMs) (OpenAI, 2023; Jiang et al., 2024a; Dubey
et al., 2024; Liu et al., 2024a; Gemma, 2025;
Yang et al., 2025). These models have advanced
substantially in reasoning, knowledge integration,
and domain adaptation through instruction tuning
and retrieval-augmented generation (RAG). This
shift has redefined what constitutes a challenging
benchmark—spanning domains such as coding
(Zheng et al., 2025), Ph.D.-level knowledge (Phan
et al., 2025), machine translation (Andrews et al.,
2025) and multimodal reasoning (Padlewski et al.,
2024)—and has led to an explosion of datasets
(Rogers et al., 2023; Liu et al., 2024b).
ContributionWe present HEAD-QA v2, an ex-
panded and updated version designed to better re-
flect the era of large-scale reasoning models. The
new release addresses the limited size and tem-
poral coverage of its predecessor by incorporat-
ing 12,751 multiple-choice questions from Spanish
professional medical qualification exams—more
than doubling the dataset and extending its time
span. We expect this expansion to enable future
researchonmodelgeneralization,knowledgereten-arXiv:2511.15355v1  [cs.CL]  19 Nov 2025

tion,andtemporaleffectstoagreaterextentthanits
predecessor. We further establish new baselines
through a systematic evaluation of open-source
LLMs, exploring multiple inference strategies, in-
cluding prompting, retrieval-augmented generation,
andaprobability-basedapproach. Together,weex-
pectthatthesecontributionsofferapracticalbench-
markforstudyinghowLLMsadapttodomainevolu-
tion, balance accuracy with efficiency, and perform
complex reasoning in specialized contexts. The
dataset is available at https://huggingface.
co/datasets/alesi12/head_qa_v2.
2. Dataset Construction
This section outlines the construction of HEAD-QA
v2, which, like its predecessor, is based on offi-
cial, publicly available exams from the Ministerio
de Sanidad de España. Each exam includes: (i)
a two-column PDF containing the text, (ii) a CSV
file listing the correct answers, and (iii) when appli-
cable, a folder with referenced images indexed nu-
merically (e.g., 1, 2, 3, 4, ...), enabling text–image
alignment.1
2.1. Preprocessing
The preprocessing pipeline involves converting,
cleaning, and standardizing the exam data.
1.PDF to text conversion. Exams were con-
verted from PDF to plain text using pdfto-
text, preserving the two-column layout.
2.Image mapping. Images were automatically
linked, as related questions begin with “Ques-
tion linked to image no. X,” where X is the
image identifier.
3.Questionfiltering. Questionswithoutanofficial
answer from the Spanish Ministry of Health
were removed, as they correspond to disputed
or withdrawn items.
4.Manual corrections. Minor edits to fix errors
and standardize content. Chemical formulas
were converted to SMILES notation (see Fig-
ure1)usingMathpix,facilitatingprocessingby
text-based ML models (Schwaller et al., 2017;
Chithrananda et al., 2020). The few affected
questions were processed manually.
Figure 1: Example of chemical formula converted
to SMILES notation for text processing.
1Questions containing images are relatively rare, and
visual processing is therefore excluded from this work.5.Storage. Files are stored in Parquet format
(The Apache Software Foundation, 2024) for
efficient compression and fast download.
2.2. Format
Each question includes eight fields (Figure 2). A
unique identifier requires both name(exam name)
andqid(question ID).
•qid(int): Question number within the exam.
•qtext(str): Question text.
•ra(int): Correct answer identifier.
•answers(list): Answer options, each with:
–aid(int): Option ID.
–atext(str): Option text.
•image (Image): Associated image in PILfor-
mat (Clark and Contributors, 2023), or nullif
none.
•year(int): Exam year.
•category (str): Discipline (e.g., Medicine,
Nursing).
•name (str): Exam identifier combin-
ing year, discipline, and version (e.g.,
Cuaderno_2013_0_B).
{’qid ’: 1,
’qtext ’: ’Excitatory postsynaptic
potentials :’,
’ra ’: 3,
’answers ’: [{’aid ’: 1, ’atext ’: ’Are all -
or - none responses .’},
{’aid ’: 2, ’atext ’: ’Are
hyperpolarizing .’},
{’aid ’: 3, ’atext ’: ’Can be summed .’},
{’aid ’: 4, ’atext ’: ’Propagate over
long distances .’},
{’aid ’: 5, ’atext ’: ’Exhibit a
refractory period . ’}] ,
’image ’: None ,
’year ’: 2013 ,
’category ’: ’biology ’,
’name ’: ’Cuaderno_2013_1_B ’}
Figure2: AHEAD-QAv2questioninJSONformat.
2.3. Dataset statistics
The dataset contains a total of 12,751 questions
distributedacrosssixdisciplinesandtenyears(see
Table 1). Among them, 334 questions include im-
ages. Of these, 36 correspond to the four most
recent nursing exams (2019–2022), while the rest
belong to the medical exams—with over 30 image-
based questions per test until 2018, and around 25
per test in subsequent years.

’13 ’14 ’15 ’16 ’17 ’18 ’19 ’20 ’21 ’22 Total
BIR227 225 226 228 226 221 177 177 203 209 2119
QIR228 228 228 231 227 229 179 179 205 206 2140
MIR227 228 231 232 231 230 181 183 207 206 2156
EIR181 203 230 223 232 228 181 180 206 205 2069
FIR229 228 225 228 229 228 180 182 207 210 2146
PIR226 227 226 230 225 228 180 173 202 204 2121
Total131813391366137213701364107810741230124012751
Table 1: Number of questions per discipline/year.
Each question has one and only one correct an-
swer. In the 2013 and 2014 exams, questions
include five possible answers (2,657 items, rep-
resenting 21% of the total), while the remaining
exams feature four options per question. The cor-
rect answer is approximately uniformly distributed
across the available options, although it is slightly
less likely to appear in the first and last positions.
This is not specific to this dataset but rather a well-
documented bias in test design, as examiners tend
to avoid placing the correct answer at the extremes
(Attali and Bar-Hillel, 2003). This minor imbalance
is not directly relevant to the purposes of this work,
asnomodelistrainedorconditionedonanswerpo-
sitions. Yet, recent studies have showed that LLMs
exhibit positional biases in multiple-choice tasks,
slightly favoring middle options (Pezeshkpour and
Hruschka, 2023; Zheng et al., 2024).
In terms of question length (Figure 3), it remains
stable over time, with the trend observed in HEAD-
QA v1 persisting in recent years. Differences are
more evident across disciplines (Figure 4): ques-
tions in biology, chemistry, pharmacology, and psy-
chology tend to be shorter, while those in medicine
andnursingaregenerallylongeranddetailed,often
involvingdiagnosticreasoningthatrequiresprecise,
context-rich information.
2.4. Machine Translation and Variants
Toassesstheimpactoflanguagevariation,wecon-
sider the original Spanish dataset and its machine-
translated English version, based on the approach
of Vilares and Gómez-Rodríguez, who addressed
the same objective in HEAD-QA v1 using Google’s
seq2seq model. For v2, we follow the recent
trend of using LLMs for translation, leveraging their
strong contextual reasoning and ability to process
longer inputs while maintaining high translation
qualityacrossdomains(Vilaretal.,2023;Zhuetal.,
2024). In particular, we adopt LLaMA-3.1-8B and
its instruction-tuned variant.
Translation prompt.We explored three prompt-
ing configurations: (i) zero-shot, providing a min-
imal translation instruction; (ii) one-shot, adding
a single manually translated example mirroring
the HEAD-QA format; and (iii) an instruction-
tuned setup where the system message defines
Figure 3: Question length distribution by year.
Figure 4: Question length distribution by discipline.
the model as an “expert translator,” sets the
Spanish →Englishdirection,andenforcestworules:
(a)preservethemultiple-choiceformat,and(b)out-
put only the translation. The usermessage is the
Spanish question and options verbatim.
Format integrity.To maintain structural parity
with the source, we apply light post-processing:
earlystoppingwhenthemodelemitsthelastoption
or begins a new prompt keyword (e.g., SPANISH );
removal of trailing, non-requested text; normaliza-
tion of option identifiers (replacing variants like “A)”,
“a.”, etc., with 1.,2., ...); and validation checks
to ensure that the output is a proper translation
rather than an attempted answer or commentary
(i.e., non-empty question and options, and consis-
tent number of options). We check output valid-
ity through automatic checks for (i) empty text in
the question or options, (ii) mismatched number
of options, and (iii) incorrect numbering (e.g., re-
peated or misordered identifiers). For English, the
instruction-tunedconfigurationproducedthefewest
errors (28), followed by the zero-shot (44) and one-
shot (186) setups. This pattern shows that these
models adhere to structured translation prompts,
obtaining stable, well-aligned outputs.

Selection of final translations.Each question
has three translated versions per target language,
correspondingtodifferentpromptingconfigurations.
The final dataset is compiled by selecting the most
reliable translation according to the following rules:
(i) if only one version is valid, it is kept; (ii) if several
are valid, the one from the configuration with fewer
errors is chosen; and (iii) if all are valid, the two
most similar are compared, selecting the one from
the lower-error setup. Questions without a valid
translation are discarded. A manual evaluation of
a random sample of questions was conducted to
verifythereliabilityandconsistencyofthisselection
procedure.
Other language variants.Using the same trans-
lation and selection pipeline, we additionally gener-
ated Italian, Galician, and Russian versions of the
dataset. Automatic format checks confirmed good
structural consistency across these languages.
Whilemodelevaluationwasnotconductedonthese
versions—since, unlike for English, no human vali-
dationcouldbeperformedinthefinalselectionstep
due to resource constraints—they will be released
alongsidethemaindatasettoserveasafoundation
forfutureresearchoncross-lingualandmultilingual
evaluation within the HEAD-QA framework.
Qualitative evaluation of translations.Still, to
automatically assess the quality of the full trans-
lated datasets (English, Italian, Russian, and Gali-
cian) and enable comparison with future versions,
we apply a back-translation (BT) approach. Each
target-language version is translated back into
Spanish and compared with the original text, ob-
taining round-trip translation (RTT) scores as a
reference-freequalityproxy(Zhuoetal.,2023). We
compute both surface-level (BLEU) and semantic
(BERTScore) similarity metrics. Results show that
GalicianandItalianachievethehighestBLEU(0.66
and 0.57) and BERTScore-F1 (0.80 and 0.77), fol-
lowed by English (0.41 / 0.69) and Russian (0.33
/ 0.65). These values are strong overall and con-
sistent with linguistic distance, languages closer to
Spanish yield higher lexical and semantic similarity,
confirming that the translation pipeline maintains
robust and semantically reliable outputs across all
languages.
3.Baselines and Inference Strategies
Next, we present the models and inference strate-
gies adopted, following standard practices.
3.1. Models
We evaluate four open-access, instruction-tuned
LLMs:Llama 3.1 (8B, 70B)(Dubey et al., 2024)
Decoder-only models with 8B and 70B param-
eters, trained on multilingual data and officially
supporting several languages beyond English, in-
cluding Spanish. Both are optimized for long-
context processing and use grouped-query atten-
tion (GQA) (Ainslie et al., 2023) to improve in-
ference efficiency over standard multi-head atten-
tion (Vaswani et al., 2017).
Mistral v0.3 (7B)(Jiang et al., 2023) A 7B
decoder-only model that combines grouped-query
and sliding-window attention for efficient process-
ing of sequences.
Mixtral v0.1 (8×7B)(Jiang et al., 2024b) Archi-
tecturally similar to Mistral 7B, Mixtral introduces a
Mixture-of-Experts (MoE) design, activating two of
eight experts per token to enhance efficiency by
limiting active computation at each step.
The two model families, Llama 3.1 and Mistral,
were selected for their broad adoption and good
performance across diverse NLP tasks. Choos-
ing one smaller and one larger model from each
family enables a controlled examination of scaling
effectsinHEAD-QAv2,clarifyinghowmodelcapac-
ity influences biomedical reasoning and multiple-
choice performance. While an exhaustive com-
parison across all available LLMs is beyond this
study’s scope, these models span both dense and
mixture-of-experts architectures, offering a repre-
sentative and methodologically sound basis for
analysis. Sincetheprimaryobjectiveofthisworkis
thedatasetitself,modelevaluationservesmainlyto
characterize its difficulty and illustrate how different
architectures respond to its challenges.2
3.2. Answer Selection Strategies
Each model answers multiple-choice questions us-
ing a consistent input–output scheme.
Model Input.By default, each question is format-
tedasasingletextsequencethatincludestheques-
tion stem and its possible answers, each preceded
by a numerical index, as illustrated in Figure 5.
Model Output.For all inference strategies, the
model is queried to produce a short JSON struc-
ture indicating the index of the selected answer.
2All experiments were conducted under consistent
hardware conditions using NVIDIA A100 GPUs (40 GB)
with 16-bit precision. Smaller models (Mistral-7Band
Llama-3.1-8B) were run on single-GPU nodes, whereas
larger ones (Llama-3.1-70BandMixtral-8x7B) required
four GPUs, distributing the computational load evenly
across devices.

Excitatory postsynaptic potentials :
1. Are all -or - none .
2. Are hyperpolarizing .
3. Can be summed .
4. Propagate over long distances .
5. Have a refractory period .
Figure 5: HEAD-QA v2 question encoded as a
single input sequence.
For example, if the chosen option is the third, the
expected output is {answer: 3} . Enforcing a
fixed output format simplifies both extraction and
post-processing of predictions, regardless of minor
variations in spacing, casing, or punctuation.
3.2.1. Prompting Strategies
Zero-shotpromptingFigure6showstheprompt
used in the zero-shot setting. It defines the ex-
pected output format and provides minimal condi-
tioning, instructing the model to act as an expert in
scientific and healthcare domains.
<| begin_of_text |><| start_header_id |> system
<| end_header_id |>
<| eot_id |><| start_header_id |>user <|
end_header_id |>
You are an expert in specialized scientific
and health disciplines . Respond to the
following multiple - choice question :
Provide the answer in the following JSON
format : { Answer : [ number ]}
For example , if the answer is 1, write : {
Answer : 1} <| eot_id |><| start_header_id |>
user <| end_header_id |>
< PLACEHOLDER FOR THE QUESTION AND OPTIONS
><| eot_id |><| start_header_id |> assistant
<| end_header_id |>
Figure 6: Zero-shot prompt. The example, for
Llama-3.1, shows the use of headers and special
tokens that delimit user–assistant interactions and
metadata as specified by the model architecture.
In-context learningLLMs often perform better
when given examples within the prompt, as these
help condition their responses. In this work, Fig-
ure 7 shows the few-shot prompt for Spanish ques-
tions, which includes three fixed examples from
diversedisciplines. Theseexamples, adaptedfrom
the United States Medical Licensing Examination
(USMLE) questions, were selected to match the
nature of HEAD-QA.3While a detailed analysis
is beyond the scope of this study, prior work has
shown that the choice and quality of in-context ex-
amples can strongly influence performance (Bon-
isoli et al., 2025). This phenomenon has also been
3Parallel Spanish and English versions were created
to ensure linguistic and domain consistency.interpretedasaformofimplicitlearningduringinfer-
ence (Dherin et al., 2025), suggesting that models
may adapt dynamically—an ability that would be
particularly relevant for sensitive (and very person-
alized) domains such as healthcare.
<| begin_of_text |><| start_header_id |> system <| end_header_id |>
You are an expert in specialized scientific and health
disciplines . Respond to the following multiple - choice
question by indicating only the number of the correct
option . No explanations are needed .<| eot_id |><|
start_header_id |>user <| end_header_id |>
Which neurotransmitter is primarily involved in mood
regulation ?
1. Dopamine
2. Serotonin
3. GABA
4. Acetylcholine <| eot_id |><| start_header_id |> assistant <|
end_header_id |>
{ Answer : 2} <| eot_id |><| start_header_id |>user <| end_header_id
|>
Which of the following is an example of a neutralization
reaction in chemistry ?
1. CH4 + 2O2 -> CO2 + 2 H2O
2. Na + Cl2 -> 2 NaCl
3. 2H2 + O2 -> 2 H2O
4. HCl + NaOH -> NaCl + H2O <| eot_id |><| start_header_id |>
assistant <| end_header_id |>
{ Answer : 4} <| eot_id |><| start_header_id |>user <| end_header_id
|>
...
< PLACEHOLDER FOR THE QUESTION AND OPTIONS ><| eot_id |><|
start_header_id |> assistant <| end_header_id |>
Figure 7: Example of a few-shot prompt with sam-
ples. Case shown for the Llama-3.1-8B model.
Chain-of-Thought promptingIn this setting, the
modelisinstructedtoproducebriefreasoningsteps
before providing an answer. As shown in Figure 8,
the prompt asks the model to evaluate each op-
tion before selecting the most plausible one. This
design encourages reasoning while keeping gener-
ations concise and inference efficient.
<| begin_of_text |><| start_header_id |> system <| end_header_id |>
You are an expert in scientific and health disciplines .
Carefully analyze the following multiple - choice
question and provide the correct answer . There is one
and only one correct answer . Think through each option
briefly before responding in the JSON format : { Answer :
[ number ]}. <| eot_id |><| start_header_id |>user <|
end_header_id |>
...
Figure 8: Example of a CoT prompt with brief rea-
soning before the final answer, using the Llama-
3.1-8B model.
3.2.2. Retrieval-Augmented Generation
Following an approach shown to improve biomedi-
cal question answering (Xiong et al., 2024), in this
workwealsoaimtomitigatepotentialhallucinations
by retrieving relevant passages from an external,
reliable corpus and appending them to the model’s
prompt to better guide its responses.

Our RAG implementation consists of three com-
ponents: (i) an LLM, (ii) a biomedical corpus, and
(iii) a retrieval system. For (i), we use the mod-
els introduced in §3.1. For (ii), we use the cor-
pus proposed by Jin et al. (2021), which con-
tains 18 medical textbooks commonly used for
USMLEpreparation.4For(iii),weuseMedCPT(Jin
et al., 2023), a dual-encoder model based on
BERT (Devlin et al., 2019). It includes two spe-
cialized encoders—ncbi/MedCPT-Article-Encoder
and ncbi/MedCPT-Query-Encoder—that map cor-
pus fragments and queries (here, HEAD-QA v2
questions) into 768-dimensional vectors.5,6
Since the corpus is in English, retrieval was per-
formed using English-translated versions of the
questions,andtheretrievedpassageswerereused
for both the English and Spanish versions of the
benchmark. Each question was paired with the
two most relevant fragments, balancing contextual
coverage with efficiency during LLM inference (to-
gether with the zero-shot prompt).
Assessing corpus alignmentTo evaluate the
suitability of this corpus for our benchmark, Fig-
ures 9, 10, and 11 show a two-dimensional
UMAP (McInnes et al., 2020) projection of all 126k
corpus fragments and the 12k HEAD-QA v2 ques-
tions. Distinct clusters correspond to individual
textbooks, with minimal overlap. Importantly, most
HEAD-QA v2 questions project into high-density
corpus regions, indicating strong topical alignment.
For example, psychology questions cluster around
Psychiatry_DSM-5 and Neurology_Adams, while
biology and pharmacology items align with related
sources. These observations suggest that the cor-
pus and retrieval setup may supply relevant con-
textual evidence, motivating their inclusion as a
baseline for our benchmark.
3.2.3. Selection via log-probabilities
Unlike the previous methods, which require au-
toregressive text generation, this approach directly
4This dataset, publicly available on the Hugging
Face Hub ( https://huggingface.co/datasets/
MedRAG/textbooks ), consists of approximately
126,000 short text fragments, each under 1,000
characters.
5Semantic similarity is computed via dot product.
These models were trained on 255 million PubMed
query–article pairs, making them highly effective for
biomedical retrieval.
6For similarity search, we use FAISS (Facebook AI
Similarity Search) (Douze et al., 2024), leveraging its na-
tive integration with the Hugging Face datasets library
for low-memory data handling. We employ a flat index
type, which performs exhaustive comparison across all
vectors with 32-bit precision, ensuring maximal retrieval
accuracy.compares the probabilities that a language model
assigns to each candidate answer sequence.
Formally, let C= (c1, c2, . . . , c n)represent
the token sequence of a question and Ai=
(a1, a2, . . . , a m)the sequence corresponding to the
i-th answer option. For each token aj, the model
computes a conditional probability qj=P(Xn+j|
X1=c1, . . . , X n=cn, Xn+1=a1, . . . , X n+j−1 =
aj−1). The overall likelihood of an answer se-
quence is then defined as the geometric mean of
its token probabilities, P(Ai) = Qm
j=1qj(aj)1/m.
The model selects as correct the option that max-
imizes this probability, i.e.,= arg max iP(Ai). Be-
causemultiplyingmanysmallprobabilitiescanlead
to numerical instability, all computations are per-
formed in 32-bit precision. In addition, probabili-
ties are evaluated in log-space to improve stabil-
ity and efficiency, using the equivalent formulation
logP(A i) =1
mPm
j=1logqj(aj).
4. Experimental setup
Performance is evaluated using three metrics:(1)
accuracy, the proportion of correct answers; (2)
the normalized exam score, based on the official
Spanish medical exam scheme (three wrong an-
swers cancel one correct) and normalized by total
items; and (3) the unanswered ratio, the fraction of
questions with no valid response.
4.1. ‘Prompting Strategy’ Evaluation
Table 2 reports performance metrics for all prompt-
ing configurations (zero-shot, few-shot, and CoT).
Prompt ModelEnglish (en) Spanish (es)
Acc ScoreP naAcc ScoreP na
Zero-shotMixtral-8x7B 70.59 66.97 2.03 66.01 60.43 4.94
Mistral-7B 59.55 52.61 4.82 52.79 43.56 3.25
Llama-3.1-8B 70.43 67.86 0.39 61.93 56.61 0.38
Llama-3.1-70B 83.1584.16 0.43 83.2784.14 0.40
Few-shotMixtral-8x7B 69.78 66.23 4.64 66.85 62.05 3.83
Mistral-7B 60.63 54.59 4.42 54.06 45.90 3.02
Llama-3.1-8B 70.49 68.240.36 62.58 57.440.24
Llama-3.1-70B 82.90 84.14 0.51 83.2484.410.38
CoTMixtral-8x7B 67.08 62.53 6.71 64.27 58.66 7.22
Mistral-7B 56.19 47.89 8.54 48.07 36.55 10.55
Llama-3.1-8B 69.13 66.50 5.55 61.05 55.30 6.11
Llama-3.1-70B 82.5484.202.11 82.10 83.21 4.07
Table 2: Performance metrics (accuracy, exam
score,andproportionofunansweredquestions)for
all prompting configurations (zero-shot, few-shot,
and CoT) in English and Spanish. Best values per
column are highlighted in bold.
Overall, performance is consistently higher in
English than in Spanish across all configurations,
except for Llama-3.1-70B, where results are equiv-
alent. This confirms that models handle En-
glish—either natively or through translation—more
effectively. The gap is particularly pronounced

Figure 9: Kernel density estima-
tion of corpus fragments by text-
book source.
Figure10: Globalkerneldensity
estimation of the corpus (with-
out separating by textbook).
Figure11: ScatterplotofHEAD-
QA v2 questions by discipline
overlaid on the corpus density
map.
Figure 12: Performance evolution over time for
eachmodelontheEnglishsubsetundertheprompt-
ing setup. Colors indicate model families: Mistral-
7B (green), Mixtral-8x7B (blue), Llama-3.1-70B (or-
ange), and Llama-3.1-8B (pink). Markers denote
prompting strategies: squares for zero-shot, trian-
gles for few-shot, and diamonds for CoT.
in smaller models, suggesting that limited capac-
ity (at least to represent specialized healthcare
knowledge) amplifies cross-lingual variability. In
contrast, larger models show stronger generaliza-
tion, narrowing the difference between languages.
Modelscalehasaclearimpact: accuracyandexam
scores increase steadily with model size, while the
proportion of unanswered questions decreases.
Regarding prompting strategies, zero-shot and
few-shot approaches achieve comparable results,
suggesting that providing a single example offers
limited additional benefit given the models’ instruc-
tion tuning. Exploring the impact of example selec-
tioncouldbeaninterestingdirectionforfuturework.
In contrast—perhaps unexpectedly—CoT prompt-
ing consistently reduces accuracy and increases
non-response rates except for the Llama-3.1-70B,
indicatingthatexplicitreasoningstepsmayactually
reduce performance in this healthcare domain.
Figure 12 shows that English performance re-
mains stable across exam years, with larger mod-
elsoutperformingsmallerones. Englishresultsare
slightly higher than Spanish (not shown for space),and simpler prompting strategies get the most reli-
able outcomes.
4.2. ‘RAG Strategy’ Evaluation
Prompt ModelEnglish (en) Spanish (es)
Acc ScoreP naAcc ScoreP na
RAGMixtral-8x7B 69.80 66.25 4.67 66.90 62.07 3.91
Mistral-7B 56.63 49.11 2.14 49.61 39.70 2.30
Llama-3.1-8B 66.45 62.83 0.69 59.13 52.86 0.57
Llama-3.1-70B 82.45 83.13 0.32 82.52 83.03 0.22
Table 3: Performance metrics (accuracy, exam
score, and proportion of unanswered questions)
for all RAG-based configurations in English and
Spanish.
Table 3 presents the performance metrics for the
models that used RAG to condition their prompt.
Overall, results show that incorporating retrieved
contextthroughRAGdoesnotleadtoconsistentim-
provementsoverstandardprompting. Performance
remains slightly higher in English than in Spanish.
LargermodelsbenefitthemostfromRAG,maintain-
ing competitive accuracy and lower non-response
rates, while smaller models tend to degrade when
exposed to noisy or weakly relevant evidence.
Compared to the zero-shot baseline, RAG gets
slightly lower scores in both languages. This sug-
gests that the retrieved passages are not always
effectively integrated into the generation process,
that the model can often rely on its internal knowl-
edgeinstead,orthattheretrievedinformationisnot
sufficiently relevant. The weak correlation between
retrievalrelevance(see§3.2.2)andanswercorrect-
ness ( r= 0.07) further supports this interpretation:
modelperformanceappearstodependprimarilyon
internal knowledge rather than external evidence.
Yearly trends remain stable across models and lan-
guages, closely mirroring those observed in the
prompting experiments, as shown in Figure 13.
4.3. ‘Log-probability’ Evaluation
Table 4 reports accuracy and normalized exam
scores for this setup. By design, the unanswered

Figure 13: Performance evolution over time for
each model in English under the RAG setup. Col-
ors indicate model families: Mixtral-8x7B (green),
Mistral-7B (orange), Llama-3.1-8B (blue), and
Llama-3.1-70B (pink).
ratio is0%, as models are required to select one
option per question. Despite this, scores drop no-
tably compared to prompting-based approaches,
indicating that direct likelihood evaluation is less
effective for multiple-choice reasoning. Perfor-
mance remains consistently higher in English than
in Spanish, with the gap being more pronounced
in smaller models. Larger models mitigate this dif-
ference, maintaining more stable accuracy across
languages. The observed performance gap can
also be attributed to the independent evaluation of
each option: without jointly considering all alterna-
tives, models lose the elimination-based reasoning
that benefits prompting approaches.
As shown in Figure 14, yearly trends remain
stable, with only minor fluctuations after 2018.
Although this method minimizes resource us-
age—since no text generation is involved—the effi-
ciency gain does not compensate for the accuracy
loss, limiting its practical value for HEAD-QA v2.
Strategy ModelEnglish (en) Spanish (es)
Acc Score Acc Score
Prob-basedMixtral-8x7B 52.84 44.06 47.87 37.60
Mistral-7B 47.86 37.49 39.42 27.69
Llama-3.1-8B 45.25 34.31 37.82 24.80
Llama-3.1-70B 54.15 46.04 51.42 42.12
Table 4: Performance metrics (accuracy and exam
score) for the probability-based selection strategy
(Section 3.2.3) in English and Spanish.
5. Discussion
The results reveal consistent trends across model
families, highlighting how architectural scale, lan-
guage, and methodological design shape perfor-
mance in HEAD-QA v2. Model size emerges as
the most decisive factor: Llama-3.1-70B consis-
Figure 14: Performance evolution over time for
each model in English under the probability-based
selection setup. Colors indicate model families:
Llama-3.1-8B (green), Llama-3.1-70B (orange),
Mistral-7B (blue), and Mixtral-8x7B (pink).
tently achieves the highest accuracy and normal-
ized exam scores, while smallest models performs
lowest across all metrics. These results align with
broader findings in LLM evaluation, where scaling
enhancesbothfactualrecallandreasoningstability.
Language effects are present but moderate, with
smaller models showing slightly reduced perfor-
mance in Spanish. This may stem from differences
in tokenization efficiency, knowlegde integration,
and from weaker multilingual representations in
smaller architectures, which maybe be less robust
tolexicalandsyntacticvariabilityacrosslanguages.
Methodologically,neithermoreelaborateprompt-
ing (few-shot or CoT) nor retrieval-augmented gen-
eration produces consistent improvements. In
some cases, these strategies even reduce perfor-
mance, suggesting that additional contextual input
can introduce noise or divert the model from lever-
aging its internal knowledge. Considering their
higher computational and developmental costs,
such methods offer limited benefit in this setting.
Finally, the probability-based answer selection
strategy performs notably worse than generation-
based approaches. Since each option is scored
independently, the model cannot perform the com-
parativereasoningandcontextualalignmenttypical
of human multiple-choice problem-solving, result-
ing in systematic accuracy drops.
6. Conclusion
This work introduced HEAD-QA v2, a new large-
scale, multilingual benchmark designed to eval-
uate complex reasoning in the biomedical do-
main. Through extensive experiments across mul-
tiple modern large language models and inference
strategies, we established empirical baselines and

analyzed the factors that most influence perfor-
mance. Ourfindingsindicatethat,forhighlyspecial-
ized biomedical question answering, the intrinsic
knowledge and reasoning capacity of the language
modelplayafargreaterrolethanthesophistication
of the inference strategy. Techniques such as RAG
and CoT prompting, while successful in other do-
mains,didnotobtainconsistentgainsinthissetting
and introduced additional computational and im-
plementation overhead. Overall, improvements on
HEAD-QAv2seemmorecloselytiedtoscalingand
refining the underlying models than to increasing
inference complexity, though alternative strategies
may still offer potential for future exploration.
Limitations
This study did not include evaluations with fron-
tier proprietary LLMs such as GPT-4, Claude, or
Gemini, primarily due to funding resources to ac-
cess APIs. Consequently, the results reflect trends
among open-access models up to 70B parameters.
Additionally, while the English translations were
automatically generated and reviewed for termino-
logical consistency, large-scale human validation
was not feasible. Minor translation inconsistencies
could therefore influence model performance, es-
pecially in domain-specific terminology.
Another limitation concerns the scope of the
benchmarkitself. HEAD-QAv2focusesonmultiple-
choice biomedical questions, which represent only
a subset of complex reasoning skills.
Ethical Considerations
HEAD-QA v2 is based on publicly available exami-
nationquestionsdesignedforhealthcareeducation,
containing no personal or patient data. Neverthe-
less, the dataset and experiments involve content
related to medical knowledge, and outputs from
large language models should not be interpreted
as clinical advice.
All experiments were conducted with open-
access models and publicly available data, ensur-
ing reproducibility and compliance with data use
terms. We acknowledge that automatic translation
andmodel-generatedtextmaypropagatebiasesor
inaccuracies, and encourage caution when using
the dataset or models in real-world or educational
healthcare contexts.
Acknowledgments
We acknowledge grants GAP (PID2022-
139308OA-I00) funded by MICI-
U/AEI/10.13039/501100011033/ and ERDF,
EU; LATCHING (PID2023-147129OB-C21) funded
by MICIU/AEI/10.13039/501100011033 and ERDF,EU; and TSI-100925-2023-1 funded by Ministry
for Digital Transformation and Civil Service and
“NextGenerationEU” PRTR; as well as funding by
Xunta de Galicia (ED431C 2024/02), and CITIC,
as a center accredited for excellence within the
Galician University System and a member of the
CIGUS Network, receives subsidies from the
Department of Education, Science, Universities,
and Vocational Training of the Xunta de Galicia.
Additionally, it is co-financed by the EU through the
FEDER Galicia 2021-27 operational program (Ref.
ED431G 2023/01).
7. Bibliographical References
Joshua Ainslie, James Lee-Thorp, Michiel de Jong,
Yury Zemlyanskiy, Federico Lebrón, and Sumit
Sanghai. 2023. Gqa: Training generalized multi-
querytransformermodelsfrommulti-headcheck-
points.
Pierre Andrews, Mikel Artetxe, Mariano Coria
Meglioli, Marta R Costa-jussà, Joe Chuang,
David Dale, Cynthia Gao, Jean Maillard, Alex
Mourachko,ChristopheRopers,etal.2025. Bou-
quet: dataset, benchmark and open initiative for
universal quality evaluation in translation.arXiv
preprint arXiv:2502.04314.
Yigal Attali and Maya Bar-Hillel. 2003. Guess
where: The position of correct answers in
multiple-choice test items as a psychometric
variable.Journal of Educational Measurement,
40(2):109–128.
Giovanni Bonisoli, David Vilares, Federica Rollo,
and Laura Po. 2025. Document-level event ex-
traction from italian crime news using minimal
data.Knowledge-Based Systems, 317:113386.
Samuel R. Bowman, Gabor Angeli, Christopher
Potts,andChristopherD.Manning.2015. Alarge
annotated corpus for learning natural language
inference. InProceedings of the 2015 Confer-
ence on Empirical Methods in Natural Language
Processing, pages 632–642, Lisbon, Portugal.
Association for Computational Linguistics.
Danqi Chen, Adam Fisch, Jason Weston, and An-
toine Bordes. 2017. Reading Wikipedia to an-
swer open-domain questions. InProceedings
of the 55th Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long
Papers), pages 1870–1879, Vancouver, Canada.
Association for Computational Linguistics.
Seyone Chithrananda, Gabriel Grand, and Bharath
Ramsundar. 2020. Chemberta: Large-scale self-

supervisedpretrainingformolecularpropertypre-
diction.
Alex Clark and Contributors. 2023. Pillow (pil
fork). https://python-pillow.org/ . Ver-
sion 10.0.0, Accessed: October 13, 2025.
Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar
Khot,AshishSabharwal,CarissaSchoenick,and
Oyvind Tafjord. 2018. Think you have solved
question answering? try arc, the ai2 reasoning
challenge.arXiv preprint arXiv:1803.05457.
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and
Kristina Toutanova. 2019. BERT: Pre-training
of deep bidirectional transformers for language
understanding. InProceedings of the 2019 Con-
ference of the North American Chapter of the As-
sociation for Computational Linguistics: Human
Language Technologies, Volume 1 (Long and
Short Papers), pages 4171–4186, Minneapolis,
Minnesota. Association for Computational Lin-
guistics.
Benoit Dherin, Michael Munn, Hanna Mazzawi,
Michael Wunder, and Javier Gonzalvo. 2025.
Learning without training: The implicit dynam-
ics of in-context learning.
Matthijs Douze, Alexandr Guzhva, Chengqi
Deng, Jeff Johnson, Gergely Szilvasy, Pierre-
Emmanuel Mazaré, Maria Lomeli, Lucas Hos-
seini, and Hervé Jégou. 2024. The faiss library.
Abhimanyu Dubey, Abhinav Jauhri, Abhinav
Pandey, Abhishek Kadian, Ahmad Al-Dahle,
Aiesha Letman, Akhil Mathur, Alan Schelten,
Amy Yang, Angela Fan, et al. 2024. The llama
3 herd of models.arXiv e-prints, pages arXiv–
2407.
Yanai Elazar, Akshita Bhagia, Ian Magnusson, Ab-
hilasha Ravichander, Dustin Schwenk, Alane
Suhr, Pete Walsh, Dirk Groeneveld, Luca Sol-
daini, Sameer Singh, et al. 2023. What’s in my
big data?arXiv preprint arXiv:2310.20707.
Gemma. 2025. Gemma 3 technical report.arXiv
preprint arXiv:2503.19786.
Dirk Groeneveld, Iz Beltagy, Evan Walsh, Akshita
Bhagia, Rodney Kinney, Oyvind Tafjord, Ananya
Jha, Hamish Ivison, Ian Magnusson, Yizhong
Wang, Shane Arora, David Atkinson, Russell Au-
thur, Khyathi Chandu, Arman Cohan, Jennifer
Dumas, Yanai Elazar, Yuling Gu, Jack Hessel,
Tushar Khot, William Merrill, Jacob Morrison,
Niklas Muennighoff, Aakanksha Naik, Crystal
Nam, Matthew Peters, Valentina Pyatkin, Abhi-
lasha Ravichander, Dustin Schwenk, Saurabh
Shah, William Smith, Emma Strubell, Nishant
Subramani, Mitchell Wortsman, Pradeep Dasigi,Nathan Lambert, Kyle Richardson, Luke Zettle-
moyer, Jesse Dodge, Kyle Lo, Luca Soldaini,
Noah Smith, and Hannaneh Hajishirzi. 2024.
OLMo: Accelerating the science of language
models. InProceedings of the 62nd Annual Meet-
ing of the Association for Computational Linguis-
tics (Volume 1: Long Papers), pages 15789–
15809, Bangkok, Thailand. Association for Com-
putational Linguistics.
Albert Q. Jiang, Alexandre Sablayrolles, Arthur
Mensch, Chris Bamford, Devendra Singh Chap-
lot, Diego de las Casas, Florian Bressand,
Gianna Lengyel, Guillaume Lample, Lucile
Saulnier, Lélio Renard Lavaud, Marie-Anne
Lachaux, Pierre Stock, Teven Le Scao, Thibaut
Lavril, Thomas Wang, Timothée Lacroix, and
William El Sayed. 2023. Mistral 7b.
Albert Q Jiang, Alexandre Sablayrolles, Antoine
Roux, Arthur Mensch, Blanche Savary, Chris
Bamford, Devendra Singh Chaplot, Diego de las
Casas, Emma Bou Hanna, Florian Bressand,
et al. 2024a. Mixtral of experts.arXiv preprint
arXiv:2401.04088.
Albert Q. Jiang, Alexandre Sablayrolles, Antoine
Roux, Arthur Mensch, Blanche Savary, Chris
Bamford, Devendra Singh Chaplot, Diego de las
Casas, Emma Bou Hanna, Florian Bressand, Gi-
anna Lengyel, Guillaume Bour, Guillaume Lam-
ple, Lélio Renard Lavaud, Lucile Saulnier, Marie-
Anne Lachaux, Pierre Stock, Sandeep Subrama-
nian, Sophia Yang, Szymon Antoniak, Teven Le
Scao, Théophile Gervet, Thibaut Lavril, Thomas
Wang, Timothée Lacroix, and William El Sayed.
2024b. Mixtral of experts.
Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung
Weng, Hanyi Fang, and Peter Szolovits. 2021.
What disease does this patient have? a
large-scale open domain question answering
dataset from medical exams.Applied Sciences,
11(14):6421.
Qiao Jin, Won Kim, Qingyu Chen, Donald C
Comeau, Lana Yeganova, W John Wilbur, and
Zhiyong Lu. 2023. Medcpt: Contrastive pre-
trained transformers with large-scale pubmed
search logs for zero-shot biomedical information
retrieval.Bioinformatics, 39(11).
Ankit Kumar, Ozan Irsoy, Peter Ondruska, Mohit
Iyyer, James Bradbury, Ishaan Gulrajani, Vic-
tor Zhong, Romain Paulus, and Richard Socher.
2016. Ask me anything: Dynamic memory net-
works for natural language processing. InPro-
ceedings of The 33rd International Conference
on Machine Learning, volume 48 ofProceed-
ings of Machine Learning Research,pages1378–
1387, New York, New York, USA. PMLR.

YanisLabrak,AdrienBazoge,RichardDufour,Beat-
rice Daille, Pierre-Antoine Gourraud, Emmanuel
Morin, and Mickael Rouvier. 2022. FrenchMedM-
CQA: A French multiple-choice question answer-
ing dataset for medical domain. InProceedings
of the 13th International Workshop on Health Text
Mining and Information Analysis (LOUHI), pages
41–46, Abu Dhabi, United Arab Emirates (Hy-
brid). Association for Computational Linguistics.
Jing Li, Shangping Zhong, and Kaizhi Chen. 2021.
MLEC-QA: A Chinese Multi-Choice Biomedical
Question Answering Dataset. InProceedings
of the 2021 Conference on Empirical Methods
in Natural Language Processing, pages 8862–
8874, Online and Punta Cana, Dominican Re-
public.AssociationforComputationalLinguistics.
Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang,
Bochao Wu, Chengda Lu, Chenggang Zhao,
Chengqi Deng, Chenyu Zhang, Chong Ruan,
etal.2024a. Deepseek-v3technicalreport.arXiv
preprint arXiv:2412.19437.
YangLiu,JiahuanCao,ChongyuLiu,KaiDing,and
Lianwen Jin. 2024b. Datasets for large language
models: A comprehensive survey.arXiv preprint
arXiv:2402.18041.
Leland McInnes, John Healy, and James Melville.
2020. Umap: Uniform manifold approximation
and projection for dimension reduction.
Ministerio de Sanidad de España. 2023.
Cuadernos de examen - formación san-
itaria especializada. https://fse.
mscbs.gob.es/fseweb/view/public/
datosanteriores/cuadernosExamen/
busquedaConvocatoria.xhtml . Accedido
el 19 de octubre de 2023.
OpenAI. 2023. Chatgpt (3.5 version). https://
chat.openai.com/. Large language model.
Piotr Padlewski, Max Bain, Matthew Henderson,
Zhongkai Zhu, Nishant Relan, Hai Pham, Dono-
van Ong, Kaloyan Aleksiev, Aitor Ormazabal,
Samuel Phua, et al. 2024. Vibe-eval: A
hard evaluation suite for measuring progress
of multimodal language models.arXiv preprint
arXiv:2405.02287.
Guilherme Penedo, Quentin Malartic, Daniel
Hesslow, Ruxandra Cojocaru, Hamza Alobei-
dli, Alessandro Cappelli, Baptiste Pannier, Ebte-
sam Almazrouei, and Julien Launay. 2023. The
refinedweb dataset for falcon llm: Outperform-
ing curated corpora with web data only. InAd-
vances in Neural Information Processing Sys-
tems, volume 36, pages 79155–79172. Curran
Associates, Inc.Bo Peng, Eric Alcaide, Quentin Anthony, Alon
Albalak, Samuel Arcadinho, Stella Biderman,
Huanqi Cao, Xin Cheng, Michael Chung, Leon
Derczynski, Xingjian Du, Matteo Grella, Kran-
thi Gv, Xuzheng He, Haowen Hou, Przemyslaw
Kazienko, Jan Kocon, Jiaming Kong, Bartłomiej
Koptyra, Hayden Lau, Jiaju Lin, Krishna Sri Ipsit
Mantri, Ferdinand Mom, Atsushi Saito, Guangyu
Song, Xiangru Tang, Johan Wind, Stanisław
Woźniak, Zhenyuan Zhang, Qinghua Zhou, Jian
Zhu, and Rui-Jie Zhu. 2023. RWKV: Reinvent-
ing RNNs for the transformer era. InFindings
of the Association for Computational Linguistics:
EMNLP 2023, pages 14048–14077, Singapore.
Association for Computational Linguistics.
Pouya Pezeshkpour and Estevam Hruschka. 2023.
Large language models sensitivity to the order
of options in multiple-choice questions.
Long Phan, Alice Gatti, Ziwen Han, Nathaniel Li,
Josephina Hu, Hugh Zhang, Chen Bo Calvin
Zhang, Mohamed Shaaban, John Ling, Sean
Shi, et al. 2025. Humanity’s last exam.arXiv
preprint arXiv:2501.14249.
Pranav Rajpurkar, Jian Zhang, Konstantin Lopy-
rev, and Percy Liang. 2016. SQuAD: 100,000+
questions for machine comprehension of text. In
Proceedings of the 2016 Conference on Empir-
ical Methods in Natural Language Processing,
pages 2383–2392, Austin, Texas. Association
for Computational Linguistics.
Anna Rogers, Matt Gardner, and Isabelle Augen-
stein. 2023. Qa dataset explosion: A taxonomy
ofnlpresourcesforquestionansweringandread-
ing comprehension.ACM Computing Surveys,
55(10):1–45.
Philippe Schwaller, Theophile Gaudin, David Lanyi,
Costas Bekas, and Teodoro Laino. 2017. "found
in translation": Predicting outcomes of com-
plex organic chemistry reactions using neural
sequence-to-sequence models.
The Apache Software Foundation. 2024. Apache
parquet documentation. Accedido el 19 de oc-
tubre de 2024.
Ashish Vaswani, Noam Shazeer, Niki Parmar,
Jakob Uszkoreit, Llion Jones, Aidan N. Gomez,
Lukasz Kaiser, and Illia Polosukhin. 2017. Atten-
tion is all you need.
David Vilar, Markus Freitag, Colin Cherry, Ji-
aming Luo, Viresh Ratnakar, and George Foster.
2023. Prompting PaLM for translation: Assess-
ing strategies and performance. InProceedings
of the 61st Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long

Papers), pages 15406–15427, Toronto, Canada.
Association for Computational Linguistics.
David Vilares and Carlos Gómez-Rodríguez. 2019.
HEAD-QA:Ahealthcaredatasetforcomplexrea-
soning. InProceedings of the 57th Annual Meet-
ing of the Association for Computational Linguis-
tics, pages 960–966, Florence, Italy. Association
for Computational Linguistics.
Xidong Wang, Nuo Chen, Junyin Chen, Yidong
Wang, Guorui Zhen, Chunxian Zhang, Xiangbo
Wu, Yan Hu, Anningzhe Gao, Xiang Wan, et al.
2024. Apollo: A lightweight multilingual medi-
cal llm towards democratizing medical ai to 6b
people.arXiv preprint arXiv:2403.03640.
Guangzhi Xiong, Qiao Jin, Zhiyong Lu, and
Aidong Zhang. 2024. Benchmarking retrieval-
augmented generation for medicine.
An Yang, Anfeng Li, Baosong Yang, Beichen
Zhang, Binyuan Hui, Bo Zheng, Bowen Yu,
Chang Gao, Chengen Huang, Chenxu Lv, et al.
2025. Qwen3 technical report.arXiv preprint
arXiv:2505.09388.
Xinlu Zhang, Chenxin Tian, Xianjun Yang, Lichang
Chen, Zekun Li, and Linda Ruth Petzold. 2023.
Alpacare: Instruction-tuned large language mod-
els for medical application.arXiv preprint
arXiv:2310.14558.
ChujieZheng,HaoZhou,FandongMeng,JieZhou,
andMinlieHuang.2024. Largelanguagemodels
are not robust multiple choice selectors.
Zihan Zheng, Zerui Cheng, Zeyu Shen, Shang
Zhou, Kaiyuan Liu, Hansen He, Dongruixuan
Li, Stanley Wei, Hangyi Hao, Jianzhu Yao, et al.
2025. Livecodebench pro: How do olympiad
medalists judge llms in competitive program-
ming?arXiv preprint arXiv:2506.11928.
Wenhao Zhu, Hongyi Liu, Qingxiu Dong, Jingjing
Xu, ShujianHuang, LingpengKong, JiajunChen,
andLeiLi.2024. Multilingualmachinetranslation
with large language models: Empirical results
and analysis. InFindings of the Association for
Computational Linguistics: NAACL 2024, pages
2765–2781, Mexico City, Mexico. Association for
Computational Linguistics.
TerryYueZhuo,QiongkaiXu,XuanliHe,andTrevor
Cohn. 2023. Rethinking round-trip translation for
machine translation evaluation.