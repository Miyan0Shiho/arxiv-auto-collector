# Prompt Tuning for Natural Language to SQL with Embedding Fine-Tuning and RAG

**Authors**: Jisoo Jang, Tien-Cuong Bui, Yunjun Choi, Wen-Syan Li

**Published**: 2025-11-11 13:41:13

**PDF URL**: [https://arxiv.org/pdf/2511.08245v1](https://arxiv.org/pdf/2511.08245v1)

## Abstract
This paper introduces an Error Correction through Prompt Tuning for NL-to-SQL, leveraging the latest advancements in generative pre-training-based LLMs and RAG. Our work addresses the crucial need for efficient and accurate translation of natural language queries into SQL expressions in various settings with the growing use of natural language interfaces. We explore the evolution of NLIDBs from early rule-based systems to advanced neural network-driven approaches. Drawing inspiration from the medical diagnostic process, we propose a novel framework integrating an error correction mechanism that diagnoses error types, identifies their causes, provides fixing instructions, and applies these corrections to SQL queries. This approach is further enriched by embedding fine-tuning and RAG, which harnesses external knowledge bases for improved accuracy and transparency. Through comprehensive experiments, we demonstrate that our framework achieves a significant 12 percent accuracy improvement over existing baselines, highlighting its potential to revolutionize data access and handling in contemporary data-driven environments.

## Full Text


<!-- PDF content starts -->

Prompt Tuning for Natural Language to SQL
with Embedding Fine-Tuning and RAG
Jisoo Jang, Tien-Cuong Bui, Yunjun Choi, and Wen-Syan Li
Graduate School of Data Science, Seoul National University, Seoul, South Korea
{simonjisu,cuongbt91,datajun77,wensyanli}@snu.ac.kr
Abstract.This paper introduces an Error Correction through Prompt
Tuning for NL-to-SQL, leveraging the latest advancements in genera-
tive pre-training-based LLMs and RAG. Our work addresses the crucial
need for efficient and accurate translation of natural language queries
into SQL expressions in various settings with the growing use of natu-
ral language interfaces. We explore the evolution of NLIDBs from early
rule-basedsystemstoadvancedneuralnetwork-drivenapproaches.Draw-
ing inspiration from the medical diagnostic process, we propose a novel
framework integrating an error correction mechanism that diagnoses er-
ror types, identifies their causes, provides fixing instructions, and applies
these corrections to SQL queries. This approach is further enriched by
embedding fine-tuning and RAG, which harnesses external knowledge
bases for improved accuracy and transparency. Through comprehensive
experiments, we demonstrate that our framework achieves a significant
12 percent accuracy improvement over existing baselines, highlighting
its potential to revolutionize data access and handling in contemporary
data-driven environments.
Keywords:NL-to-SQL·Embedding Fine-Tuning·Large Language
Models·Retrieval-Augmented Generation
1 Introduction
Natural language interfaces (NLIs) offer a convenient means for querying and
interacting with various data types. Increasingly, individuals are using everyday
language to access online and offline information. For instance, people might
inquire about today’s weather or discuss tomorrow’s agenda on their personal
calendarswithNLIs.AnotableexampleisMoveworks,acompanythatprimarily
develops an AI copilot to boost employee productivity. This technology enables
employees to effortlessly access personal or company data using text, which em-
ploys natural language. As generative AI continues evolving, natural language
interfaces seamlessly integrate into our everyday lives.
Natural Language Interfaces to Databases (NLIDBs) [1] facilitate data access
in the Natural Language Processing (NLP) domain by allowing users to query
Presented at the Workshop on Robust ML in Open Environments (PAKDD 2024)arXiv:2511.08245v1  [cs.CL]  11 Nov 2025

2 Jisoo et al.
databases using natural language. These systems aim to translate a natural lan-
guagequestionq nlintoanequivalentSQLexpressionq sql(NL-to-SQL)inagiven
database systemD. Various approaches have been explored for NLIDBs. Early
methods like LUNAR [2] and Athena [3] employed rule-based systems to parse
or map NL queries to an ontology syntactically and then to SQL. With advance-
ments in NLP and neural networks, intermediate representation systems trans-
form NL queries into vector representations before generating SQL. For instance,
SQLova [4] and HydraNet [5] predict SQL keyword sections, while RATSQL [6]
and BRIDGE [7] utilize encoder-decoder frameworks for SQL generation.
Recent advancements in generative pre-training-based large language mod-
els (LLMs) [8][9] address NL-to-SQL challenges. These models, trained on vast
internet-sourced data, excel in understanding human language. They operate
through sequence-to-sequence tasks, where users provide a prompt to gener-
ate conditional answers. Research shows that LLMs’ effectiveness in NL-to-SQL
tasks is significantly influenced by the design of these prompts. Notably, in-
cluding schema and foreign critical information in prompts led to a 51.6% im-
provement in execution accuracy on the Spider development dataset [10][11], a
benchmark for large-scale NL-to-SQL tasks.
Two fundamental approaches for improving LLM-based NL-to-SQL are fine-
tuning [12][13], adapting LLMs to structured tabular data, and inference-only
[14] with either prompt engineering or few-shot in-context learning. For instance,
Pourreza et al. [15] improved the generation performance by decomposing prob-
lems and employing chain-of-thought [16] with few-shot prompting. Despite nu-
merous advancements, the few-shot approach depends on the quality and quan-
tity of examples [17]. The Retrieval-Augmented Generation (RAG) approach
[18] can mitigate the few-shot approach drawbacks, which enhances LLMs by
integrating facts from external knowledge bases. It ensures access to the most
current and accurate information and offers users transparency in the model’s
generative process. However, optimizing content augmentation within LLMs’
constrained input size remains a complex issue in the NL-to-SQL field.
ThispaperproposesanErrorCorrectionframeworkbasedonPromptTuning
with embedding fine-tuning and RAG for NL-to-SQL. The motivation for this
work stems from medical diagnosis processes. As depicted in Fig. 1, our frame-
workincorporatesanerrorcorrectionprocessthatdiagnoseserrortypes,findsthe
reasons, provides fixing instructions based on RAG, and applies the instructions
to fix SQL errors. To improve retrieval processes, we fine-tune a pre-trained Sen-
tenceTransformermodelwithacustomizederrorcorrectiondataset.Weevaluate
the efficiency and correctness of our framework through extensive experiments.
Experimental results demonstrate that our framework achieves a 12% accuracy
improvement compared to baselines.
Therestofthepaperisorganizedasfollows:Section2introducesbackground.
Section 3 presents error analysis. Section 4 explains our framework. Section 5
shows our experiment results. Finally, Section 6 summarizes the findings and
discusses the limitations and future works.

Prompt Tuning for NL2SQL with Embedding Fine-Tuning and RAG 3
Legend New Error Case 
classiﬁes 
Correction Cases Error Types 
Knowledge Base 
(Error id, Error name, 
and simple explanation) LLM 
Classiﬁed Error Types 
Case Correction INFO Case 
Vector DB 
request 
relevant cases Relevant Cases LLM 
 LLM 
generate generate 1. Diagnose 2. Write Prescription 3. Apply Treatment 
T ext type object 
Case object T able description 
NL question 
Generated SQL 
Execution result 
T able description 
NL question 
Generated SQL 
Execution result Error Type 
Correct SQL 
Reason 
Instruction Reason 
Instruction Correct SQL Diagnosis 
Prompt Prescription 
Prompt Treatment 
Prompt 
e0
e1
…
e13 No error 
Other:DISTINCT 
…
Group-by:Wrong Cols [            ,            ] to text to text 
e4 e9to text 
Fig. 1.Overview of Error Correction through Prompt Tuning (ECPT)
2 Background
2.1 Large language model (LLM)
LLMs like GPT-3.5 and GPT-4, PaLM [19], and LLaMa [20] achieve compre-
hensive language understanding and generation via pre-training with an enor-
mous amount of data and fine-tuning with RLHF [21]. Additionally, domain-
specific LLMs like BloombergGPT [22], trained with domain-specific and pro-
prietary data, are also prevalent. LLMs excel in numerous NLP tasks since they
can understand diverse input contexts through billions of trained parameters.
2.2 Fine-tuning Language Models
Foundationmodelscanbefine-tunedforspecifictasksthroughprimarymeth-
ods:model fine-tuning,prompt tuning, andprompt engineering. Model
fine-tuning involves adjusting the pre-trained model’s weights using supplemen-
tary labeled data, which can be both time-intensive and costly. Conversely,
prompt tuning or prompt engineering [14] is more cost-efficient. In prompt tun-
ing, AI-generated numerical sequences, known assoft prompts, alter a separate
tuneable model’s embedding space to guide LLM to perform specific tasks. Ad-
ditionally, human engineers can employhard promptsor engineered prompts,
which are instructions or examples integrated with the input prompt, to steer
the model toward specific tasks. These hard prompts, considered few-shot learn-
ing examples, are more interpretable but generally less effective than the AI-
generated soft prompts. While soft prompts offer better performance, they lack
interpretability,resemblinga‘blackbox’similartodeeplearningmodels,whereas
hard prompts, being engineer-generated, are easier to understand.

4 Jisoo et al.
2.3 Retrieval-Augmented Generation
Fine-tuning a model alone often fails to equip it with the comprehensive
knowledge necessary to answer particular, contextually evolving questions. Since
LLMs are trained with Internet data, they may suffer from inconsistencies and
outdated information.Retrieval-Augmented Generation (RAG)[18], en-
hancing LLMs by retrieving facts from external knowledge bases is a solution
for these issues. It reduces the need for continuous retraining and fine-tuning
of new data, resulting in better costs of maintaining high-quality Q&A perfor-
mance. Additionally, RAG can be used together with fine-tuning approaches to
maximize the benefits of both approaches.
3 Error Analysis
To gain a clearer insight into the limitations of LLMs in a zero-shot scenario,
we randomly selected a subset of 16 databases and 959 NL questions in the
Spider [11] training dataset. The basic prompt was utilized in conjunction with
foreign keys to assist LLMs in formulating accurate responses to NL questions,
as suggested in [23]. After executing the generated queries, we classified the
execution results as follows:
•Success:The SQL executed successfully compared to the ground-truth
queries.
•Execution Error:An SQL system error occurs when executing the gener-
ated SQL.
•Empty Table:The SQL result returns an empty result. Usually, it happens
when there are no matching values in the WHERE clause.
•Undesired Result:The SQL executed well but did not match the user’s
expectation compared to the ground-truth queries.
Bydouble-checkingtheexecutionresultsandthegeneratedqueriesmanually,
we excluded 14 questions that the natural language question did not match the
ground-truth query. For example, it is hard to consider selectingrIDfrom the
table in the question ‘Find the title and star rating of the movie that got the
least rating star for each reviewer.’ The SELECT clause in the ground-truth
query wasSELECT T2.title, T1.rID, T1.stars, min(T1.stars).
Tofurtherstudytheexecutionresults,ourteammanuallyclassifiedtheminto
six categories, similar to [15] but with some modification, as shown in Fig. 2. One
of the different categories is ‘Other:Not Enough Value Information’. It has been
discovered that many SQL queries lack value information, resulting in an empty
table. E.g., in the question ‘What are all the different food allergies?’ and the
ground-truth query of WHERE clause isWHERE allergytype = "food". LLM
generates"Food"instead of"food".
Since LLMs are trained on extensive data and code for general purposes
but not trained on tasks on structured tabular data, they might struggle with
knowledge of enterprise-scaled data lakes. They can craft queries from basic

Prompt Tuning for NL2SQL with Embedding Fine-Tuning and RAG 5
Group by 
7.8% 
Other 
21% Nested 
21% 
Wrong Cols 
4.8% Alias 4% Wrong Cols 
7.6% 
Wrong Keyword 
7.6% 
Wrong Tables/Cols 
9.6% Wrong Sub Query 
11.2% 
Set Operation 
9.8% 
Not Enough 
Value Information 
12.3% 
DESC 
3.8% 
DISTINCT 
4.9% Cond 
17% Wrong Cols 
10.5% 
Schema-Linking 
27.5% 
Invalid 
5.5% 
Join 
17.2% Not Detected 
3%
Fig. 2.Statistics of error classification
table descriptions containing only the table, column names, and foreign keys
but often miss crucial value information, leading to inaccuracies. This guides
us in developing an idea to construct a knowledge base for error correction and
utilizing RAG to solve the problem.
When categorizing, we note reasons for LLM-generated failures and suggest
instructions. Ambiguous SQL is labeled as a failure against the ground-truth
standard. For example, the question ‘Which major has the most number of
students?’ yields two similar but different SQL queries: ground-truth (SELECT
major) and generated query (SELECT Major, COUNT(*) AS StudentCount).
Although both are technically correct, we set the ground-truth query as the
desired answer to the NL question to resolve these ambiguities.
All the categories are described in Table 1. Detailed descriptions of each
category are in the supplementary material.
4 Methodology
Error Correction through Prompt Tuning (ECPT)is a three-step
approach for fixing SQL query errors. It starts with the LLM identifying error
types using a Diagnosis Prompt. Relevant cases are then fetched from a vector
database. The LLM explains the error and generates instructions on correcting
it via the Prescription Prompt. Finally, the Treatment Prompt generates the
corrected SQL with instructions. It is similar to how doctors make a diagnosis
and try to treat a patient. Given error information, we decompose the error
correction process into three steps as shown in Fig. 1: (1) Diagnose, (2) Write
Prescription, and (3) Apply Treatment.

6 Jisoo et al.
4.1 Terminology Definitions
The termcaserefers to the outcome of zero-shot SQL generation and ex-
ecution, encompassing a table description (including table, column names, and
foreign keys), a natural language question, generated SQL, and its execution
result. This information, formatted as structured text, aids in prompt creation
or vector conversion for similarity searches in a vector database.
Table 1.Error Types
Error ID Error Name Short Explanations
e1 Other:DISTINCT Didn’t use or use keyword DISTINCT properly.
e2 Other:DESC Didn’t use or use keyword DESC properly.
e3 Other:Not Enough Value Information Wrong value in the WHERE clause.
e4 Schema-Linking:Wrong Cols Unnecessary or wrong columns in SELECT clause refer to question.
e5 Schema-Linking:Cond Missing or used wrong logic in the conditions.
e6 Nested:Wrong Sub Query Unnecessary or wrong sub query.
e7 Nested:Set Operation Didn’t used set operation.
e8 Join:Wrong Tables/Cols Joined unnecessary or wrong tables or columns.
e9 Join:Wrong Keyword Didn’t use JOIN keyword where it should be used or misuse LEFT/RIGHT JOIN.
e10 Invalid:Wrong Cols Use columns that do not exist in the table.
e11 Invalid:Alias Used same column name in a single statement without any alias.
e12 Group-by:Not Detected Didn’t use GROUP BY keyword where it should be used.
e13 Group-by:Wrong Cols Group by wrong columns or unnecessary group by.
Inourerroranalysis(Section3),wemanuallyidentifiedvariousErrorTypes
andoutlinedtheminTable1,eachcomprisingmultiplecorrectioncases.These
cases provide details like the error type, correct SQL, reasons for failure, and
instructions. Additionally, we transform ’Case’ information from these cases into
embedding vectors using the Sentence Transformer model [24], storing them in
FAISS [25] for similarity searches with new error case query vectors.
Step 1: Diagnosing.LLM performs well in the classification problem with a
diagnostic reasoning process [26]. We developed aDiagnosis Promptfor LLMs
to classify error types. When an error arises, case details and an error types table
are input as text to the prompt (indicated by red and yellow lines in Fig. 3). The
LLM outputs one or more error types, ranked in descending order of severity,
highlighting the most critical error first.
Step 2: Writing Prescription.In our experiment (Section 5), using the self-
generic correction prompt from [15], we found that LLMs can’t correct SQL
errors due to limited knowledge. To address this, we introduced aPrescription
Promptfor LLMs. This prompt generates reasoning and instructions for the
new error case. It starts by converting the error case into structured text and
using the Sentence Transformer model to embed it as a vector. Then, it finds
the most relevant cases from the vector database via similarity searching and
transforms them into text for the Prescription Prompt, as shown in the blue
blocks of Fig. 3.
Step 3: Applying Treatment.Given the instructions on fixing the generated
SQL, the final step is to serve it as input to theTreatment Promptfor LLM

Prompt Tuning for NL2SQL with Embedding Fine-Tuning and RAG 7
Diagnosis Prompt 
…(omit the format instructions)… 
1. Table description (`table_desc`: str): the description of the table with format(Table [table 
name]: columns=[*, column names]\n…\nForeign Keys=[ ... ]). 
2. Question (`question`: str): a natural language query about the table. 
3. Generated SQL (`generated_sql`: str): the generated SQL by the model. 
4. Execution result (`exec_res`: str): the execution result of the generated SQL, there are 
four types of errors (1) Execution Error: SQLite Error when executing the SQL (2) Empty 
Table: The SQL result returns empty table(or empty list) (3) Undesired Result: Executed well 
but doesn't match user's expectation (4) Success: the SQL executed well. 
If the `exec_res` is not 'Success', you need to diagnose the error id(or multiple error ids) with 
the given information above as much as you can. If there exists multiple error ids, you MUST 
rank the most possible error id in descending order. 
Else, you need to output 'e0' for `error_ids`. 
Here is the new error case: 
```
"table_desc": """{table_desc}""" 
"question": "{question}" 
"generated_sql": """{generated_sql}""" 
"exec_res": "{exec_res}" 
"error_ids": #fillin 
```
Here is the error types refer to the new error case, empty in the column 'case_example' 
means there is no existing example avaliable in our database, however it doesn't mean it 
has the error. 
{error_table} 
Begin to diagnose with no explanation! 
…(omit the format instructions)… Prescription Prompt 
…(omit the format instructions)… 
…(omit the context: 1. Table description ~ 4. Execution result)… 
5. Error Name (`error_names`: list[str]): the error names of the generated SQL. 
6. Reason (`reason`: str): the reason why the generated SQL has the error. 
7. Instruction (`instruction`: list[str]): the instruction to generate a correct SQL. It consists of 
the smallest unit of modification. Each amendment shall describe the modification of the 
content of 'generated_sql'. 
We have several error cases that have the same error name and execution result as the 
current error case. Please refer to the following error cases and write down the reasons and 
instructions for the current error case. 
{prescription_examples} 
…(omit the format instructions)… 
Begin! Remind to keep format instructions. 
New error case: 
```
"table_desc": """{table_desc}""" 
"question": "{question}" 
"generated_sql": """{generated_sql}""" 
"exec_res": "{exec_res}" 
"error_names": {error_names} 
"reason": #fillin , 
"instruction": #fillin 
```
Fig. 3.Diagnosis Prompt and Prescription Prompt are designed to diagnose and write
prescriptions. Red blocks refer to the new error case information, yellow blocks are
error type information, blue blocks are relevant case examples, and green blocks are
answers that LLM should fill in.
to generate the correct SQL. The system will attempt this up to three times
until a ‘Success’ is achieved in the execution result.
4.2 Embedding Fine-Tuning
SinceLLMsaretrainedonvastanddiversedatasources,theycanunderstand
diversecontexts.Intuitively,wordswithsimilarmeaningsareclosertoeachother
in the embedding space, and those that are not are further apart. However, in
our case, embedding vectors for correction cases should be organized by case
types rather than token meanings. Thus, we fine-tuned a Sentence Transformer
model with MPNet [27] base architecture using a dataset of correction cases
labeled with 14 labels (13 error types and one success). We employed triplet loss
[28] for 20 epochs to enhance relevance in case retrieval.
5 Experiments and Results
5.1 Evaluations, Models, and Metrics
Our evaluation was conducted on the Spider [11] development set, where
ground-truthqueriescanbeeasilyaccessible.WeusedOpenAI’smodels(GPT3.5-
turbo, GPT4-turbo) with 0.01 temperature and a different number of max to-
kens(100,1,024,and600foreachstepinECPT).Theperformancewasmeasured
by execution accuracy in most of the experiments. To assess the effectiveness of

8 Jisoo et al.
the diagnosis, we utilize a straightforward hit rate metric: the number of trials
that succeed in fixing errors divided by the number of total trials.
5.2 Embedding Fine-Tuning
100
 50
 0 50 100
x1100
50
050100x2
w/o Fine-tuning
100
 50
 0 50 100
x1100
50
050100x2
w/ Fine-tuning (10% Cases)
100
 50
 0 50 100
x1100
50
050100x2
w/ Fine-tuning (50% Cases)
100
 50
 0 50 100
x1100
50
050100x2
w/ Fine-tuning (100% Cases)Correction Cases(n=945) Embedding Space
Fig. 4.Embedding vector spaces of correction cases visualized via T-SNE[29]. Each
point represents an embedded vector for each correction case. Success cases are colored
green, and other colors are failed cases.
OptimizingtaskswithRAG(Retriever-AugmentedGeneration)involvescom-
plex challenges, particularly in providing relevant examples and augmentations.
We aim to enhance the semantic understanding of success and error types by
refining the embedded vectors of correction cases, as illustrated in our results
(Fig. 4). We evaluated the impact of fine-tuning on RAG by testing various sizes
ofcorrectioncases,ensuringanequaldistributionofeacherrortype.Thisprocess
revealed that while additional correction cases help distinctly separate success
and error values in the embedding space, error cases remain closely clustered
due to their multiple types.
5.3 Results on Execution Accuracy
Fig. 5 shows that using different models and case numbers for RAG sig-
nificantly improves execution accuracy on the development set. Switching to
GPT4-turbo and incorporating an embedding fine-tuned model yielded about a
12% gain compared to the baseline, the GPT3.5-turbo model with a generic self-
correction prompt and no RAG. Both GPT3.5-turbo and GPT4-turbo models
saw improvements with RAG from 76.07% to 78.78% and from 83.04%(GPT4-
turbo in Table 2) to 84.88%, respectively. Also, further benefits were gained from
the embedding fine-tuned model (increasing by about 1% and 3%). Interestingly,
the size of the correction cases had minimal impact, suggesting that correcting
SQL errors might require fewer examples. This aspect needs more exploration
in the future.

Prompt Tuning for NL2SQL with Embedding Fine-Tuning and RAG 9
Generic
Self-Correction
(W/O RAG)GPT3.5t
(100%)GPT3.5t-EFT(10%)GPT3.5t-EFT(50%)GPT3.5t-EFT(100%)GPT4t
(100%)GPT4t-EFT(10%)GPT4t-EFT(50%)GPT4t-EFT
(100%)
Models(% of Correction Cases(n=264))65707580859095100Spider Dev-set Accuracy(%)76.0778.7879.8478.97 79.4684.8887.69 87.3188.08Spider Dev-set Accuracy with different Models
GPT3.5t
GPT4t
 W/O RAG
W/O Embedding Fine-Tuning
W/ Embedding Fine-Tuning
Fig. 5.Experimentsresults.Percentagevaluesinthex-axismeanaccuraciesofdifferent
configurations with LLM models.
Table 2.Spider Dev-Set Performance Summary. Correction accuracy is determined by
dividing successful fixes by 247 error cases. Execution accuracy is obtained by dividing
the total of fixed cases and zero-shot prompting successes by 1,032 cases.
Prompt Type Model Option A
Fine-Tuned
EmbeddingsOption B
Provide Example
in DiagnosisOption C
Resolve all
at onceCorrection
Accuracy
(247 cases)Spider Dev-set
Execution Accuracy
(1032 cases)
- GPT4 - - - - 77.33%
- GPT4-turbo - - - - 83.04%
Generic GPT3.5-turbo False False False 0.00% 76.07%
ECPT GPT3.5-turbo False False False 11.34% 78.78%
ECPT GPT3.5-turbo True False False 14.17% 79.46%
ECPT GPT3.5-turbo True True False 14.57% 79.55%
ECPT GPT4-turbo False False False 36.84% 84.88%
ECPT GPT4-turbo True False False 50.20% 88.08%
ECPT GPT4-turbo True True False 44.13% 86.63%
ECPT GPT4-turbo True False True 48.18% 87.60%
ECPT GPT4-turbo True True True 50.61% 88.18%
Table 2 presents the execution accuracy in an ablation study. Zero-shot NL-
to-SQL results are shown in the first two rows. Option A uses a fine-tuned
embedding model, while Option B employs a diagnostic prompt with a new er-
ror case alongside each error type. Option C selects only the top error types
post-diagnosis. Embedding fine-tuning improved error correction, but other op-
tions didn’t enhance performance. Compared with other works as a reference,
Din-SQL [15] achieved 74.2% execution accuracy on the dev set of Spider. Our
approach emphasizes error correction and needs feedback based on ground-truth
SQL queries, especially for execution result: “Undesired Result.”

10 Jisoo et al.
5.4 Hit Rate and Cost Usage
Table 3 reports the hit rate, token usage, and costs while using API, a novel
analysis not previously undertaken. We investigate the cost-effectiveness of accu-
racy improvements in three key experiments. The accuracy gain per dollar spent
compared to the baseline (76.07%) is 0.52%, 0.59%, and 0.38% for each exper-
iment. Notably, despite its higher hit rate, the third experiment incurs higher
costs due to excessive token usage in input prompts, approximately 1.6 times
more than the GPT4t-EFT experiment.
Table 3.Hit Rate and Cost Usage. Prompt tokens are the number of tokens used for
input prompts. Completion tokens are the number of tokens that LLM generates.
Experiments Prompt Tokens Completion Tokens Total Cost($) Hit Rate # of Trials Execution Accuracy
GPT3.5t-EFT 2,020,555 129,455 6.58 5.09% 687 79.46%
GPT4t-EFT 1,665,553 119,112 20.23 23.01% 539 88.08%
GPT4t-EFT w/ options B, C 2,808,228 120,913 31.71 23.81% 525 88.18%
6 Conclusion
ThispaperproposedanovelapproachtoNL-to-SQL,addressingacriticalgap
in current systems. By integrating error correction with prompt tuning, embed-
dingfine-tuning,andRAG,weaddressedthecrucialneedforaccuratetranslation
of natural language questions into SQL expressions. Our method drew inspira-
tion from medical diagnosis processes and went beyond simple query translation;
it intelligently diagnoses and corrects errors, leveraging external knowledge bases
to refine its outputs. The notable 12% accuracy improvement over existing base-
lines underscores the effectiveness of our framework, marking a substantial ad-
vancement in data access and management. This breakthrough has far-reaching
implications, offering a powerful tool for diverse users, particularly in decision-
making roles, and sets a new benchmark for future research and development in
NL-to-SQL.
We see much room for improvement in our framework. First, since LLM-
generated queries need to be verified with ground-truth ones, we can integrate
Human-in-the-loop approaches to address this challenge. Second, manual error
typeinitializationcanbeautomaticbyutilizingLLMagents.Finally,RAG-based
prompt tuning remains resource-intensive due to the nature of the decomposed
error correction process.

Prompt Tuning for NL2SQL with Embedding Fine-Tuning and RAG 11
References
1. I. Androutsopoulos, G. D. Ritchie, and P. Thanisch. Natural language interfaces
to databases - an introduction, 1995.
2. William Woods. The lunar sciences natural language information system.BBN
report, 1972.
3. Diptikalyan Saha, Avrilia Floratou, Karthik Sankaranarayanan, Umar Farooq Min-
has, Ashish R Mittal, and Fatma Özcan. Athena: an ontology-driven system for
natural language querying over relational data stores.Proceedings of the VLDB
Endowment, 9(12):1209–1220, 2016.
4. Wonseok Hwang, Jinyeong Yim, Seunghyun Park, and Minjoon Seo. A comprehen-
sive exploration on wikisql with table-aware word contextualization.arXiv preprint
arXiv:1902.01069, 2019.
5. Qin Lyu, Kaushik Chakrabarti, Shobhit Hathi, Souvik Kundu, Jianwen Zhang,
and Zheng Chen. Hybrid ranking network for text-to-sql.arXiv preprint
arXiv:2008.04759, 2020.
6. Bailin Wang, Richard Shin, Xiaodong Liu, Oleksandr Polozov, and Matthew
Richardson. Rat-sql: Relation-aware schema encoding and linking for text-to-sql
parsers.arXiv preprint arXiv:1911.04942, 2019.
7. Xi Victoria Lin, Richard Socher, and Caiming Xiong. Bridging textual and
tabular data for cross-domain text-to-sql semantic parsing.arXiv preprint
arXiv:2012.12627, 2020.
8. Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya
Sutskever, et al. Language models are unsupervised multitask learners.OpenAI
blog, 1(8):9, 2019.
9. Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Pra-
fulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell,
et al. Language models are few-shot learners.Advances in neural information pro-
cessing systems, 33:1877–1901, 2020.
10. Nitarshan Rajkumar, Raymond Li, and Dzmitry Bahdanau. Evaluating the text-
to-sqlcapabilitiesoflargelanguagemodels.arXiv preprint arXiv:2204.00498,2022.
11. Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li, James
Ma, Irene Li, Qingning Yao, Shanelle Roman, et al. Spider: A large-scale human-
labeled dataset for complex and cross-domain semantic parsing and text-to-sql
task.arXiv preprint arXiv:1809.08887, 2018.
12. Haoyang Li, Jing Zhang, Cuiping Li, and Hong Chen. Resdsql: Decoupling schema
linking and skeleton parsing for text-to-sql. InAAAI, 2023.
13. Torsten Scholak, Nathan Schucher, and Dzmitry Bahdanau. Picard: Parsing in-
crementally for constrained auto-regressive decoding from language models.arXiv
preprint arXiv:2109.05093, 2021.
14. Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Gra-
ham Neubig. Pre-train, prompt, and predict: A systematic survey of prompting
methods in natural language processing.arXiv preprint arXiv:2107.13586, 2021.
15. Mohammadreza Pourreza and Davood Rafiei. Din-sql: Decomposed in-context
learning of text-to-sql with self-correction.arXiv preprint arXiv:2304.11015, 2023.
16. JasonWei,XuezhiWang,DaleSchuurmans,MaartenBosma,BrianIchter,FeiXia,
Ed Chi, Quoc Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning
in large language models.arXiv preprint arXiv:2201.11903, 2022.
17. Yisheng Song, Ting Wang, SK Mondal, and JP Sahoo. A comprehensive survey of
few-shot learning: Evolution, applications, challenges, and opportunities (2022).

12 Jisoo et al.
18. Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks.
arXiv preprint arXiv:2005.11401, 2020.
19. Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav
Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebas-
tian Gehrmann, et al. Palm: Scaling language modeling with pathways.arXiv
preprint arXiv:2204.02311, 2022.
20. Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne
Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal
Azhar,etal. Llama:Openandefficientfoundationlanguagemodels.arXiv preprint
arXiv:2302.13971, 2023.
21. Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela
Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Train-
ing language models to follow instructions with human feedback.Advances in
Neural Information Processing Systems, 35:27730–27744, 2022.
22. Shijie Wu, Ozan Irsoy, Steven Lu, Vadim Dabravolski, Mark Dredze, Se-
bastian Gehrmann, Prabhanjan Kambadur, David Rosenberg, and Gideon
Mann. Bloomberggpt: A large language model for finance.arXiv preprint
arXiv:2303.17564, 2023.
23. Dawei Gao, Haibin Wang, Yaliang Li, Xiuyu Sun, Yichen Qian, Bolin Ding, and
Jingren Zhou. Text-to-sql empowered by large language models: A benchmark
evaluation.arXiv preprint arXiv:2308.15363, 2023.
24. Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using
siamese bert-networks.arXiv preprint arXiv:1908.10084, 2019.
25. Jeff Johnson, Matthijs Douze, and Hervé Jégou. Billion-scale similarity search with
gpus.IEEE Transactions on Big Data, 7(3):535–547, 2019.
26. Xiaofei Sun, Xiaoya Li, Jiwei Li, Fei Wu, Shangwei Guo, Tianwei Zhang, and
Guoyin Wang. Text classification via large language models.arXiv preprint
arXiv:2305.08377, 2023.
27. Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, and Tie-Yan Liu. Mpnet: Masked
and permuted pre-training for language understanding.Advances in Neural Infor-
mation Processing Systems, 33:16857–16867, 2020.
28. Elad Hoffer and Nir Ailon. Deep metric learning using triplet network.arXiv
preprint arXiv:1412.6622, 2014.
29. Laurens Van der Maaten and Geoffrey Hinton. Visualizing data using t-sne.Jour-
nal of machine learning research, 9(11), 2008.