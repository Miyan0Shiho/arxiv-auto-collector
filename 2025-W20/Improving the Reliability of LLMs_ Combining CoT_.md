# Improving the Reliability of LLMs: Combining CoT, RAG, Self-Consistency, and Self-Verification

**Authors**: Adarsh Kumar, Hwiyoon Kim, Jawahar Sai Nathani, Neil Roy

**Published**: 2025-05-13 23:57:02

**PDF URL**: [http://arxiv.org/pdf/2505.09031v1](http://arxiv.org/pdf/2505.09031v1)

## Abstract
Hallucination, where large language models (LLMs) generate confident but
incorrect or irrelevant information, remains a key limitation in their
application to complex, open-ended tasks. Chain-of-thought (CoT) prompting has
emerged as a promising method for improving multistep reasoning by guiding
models through intermediate steps. However, CoT alone does not fully address
the hallucination problem. In this work, we investigate how combining CoT with
retrieval-augmented generation (RAG), as well as applying self-consistency and
self-verification strategies, can reduce hallucinations and improve factual
accuracy. By incorporating external knowledge sources during reasoning and
enabling models to verify or revise their own outputs, we aim to generate more
accurate and coherent responses. We present a comparative evaluation of
baseline LLMs against CoT, CoT+RAG, self-consistency, and self-verification
techniques. Our results highlight the effectiveness of each method and identify
the most robust approach for minimizing hallucinations while preserving fluency
and reasoning depth.

## Full Text


<!-- PDF content starts -->

arXiv:2505.09031v1  [cs.AI]  13 May 2025Improving the Reliability of LLMs: Combining Chain-of-Thought
Reasoning and Retrieval-Augmented Generation
Adarsh Kumar
Computer Science and Engineering
Texas A&M University
adarsh0801@tamu.eduHwiyoon Kim
Computer Science and Engineering
Texas A&M University
hwiyoonkim@tamu.edu
Jawahar Sai Nathani
Computer Science and Engineering
Texas A&M University
jawaharsainathani@tamu.eduNeil Roy
Computer Science and Engineering
Texas A&M University
neilroy@tamu.edu
Abstract
Hallucination, where large language models
(LLMs) generate confident but incorrect or ir-
relevant information, remains a key limitation
in their application to complex, open-ended
tasks. Chain-of-thought (CoT) prompting has
emerged as a promising method for improving
multistep reasoning by guiding models through
intermediate steps. However, CoT alone does
not fully address the hallucination problem. In
this work, we investigate how combining CoT
with retrieval-augmented generation (RAG),
as well as applying self-consistency and self-
verification strategies, can reduce hallucina-
tions and improve factual accuracy. By incorpo-
rating external knowledge sources during rea-
soning and enabling models to verify or revise
their own outputs, we aim to generate more
accurate and coherent responses. We present
a comparative evaluation of baseline LLMs
against CoT, CoT+RAG, self-consistency, and
self-verification techniques. Our results high-
light the effectiveness of each method and iden-
tify the most robust approach for minimizing
hallucinations while preserving fluency and rea-
soning depth.
1 Introduction
Large Language Models (LLMs) have made signif-
icant strides in various natural language processing
tasks, but one of the persistent challenges they face
is the issue of hallucination, where models gener-
ate incorrect or fabricated information that appears
plausible. This problem can hinder the reliability
and trustworthiness of LLMs in real-world applica-
tions. (Naveed et al., 2024)
To address the issue of hallucination in Large
Language Models (LLMs), an effective approach
involves integrating Chain-of-Thought (CoT) rea-
soning with Retrieval-Augmented Generation
(CoT-RAG). In this method, the model generatesreasoning steps based on evidence retrieved from
an external knowledge base, rather than relying
on potentially inaccurate or fabricated informa-
tion.(Gao et al., 2024) In RAG, the model retrieves
relevant information from a knowledge base or doc-
ument corpus (such as Wikipedia) to support the
generation process. This allows the model to access
up-to-date, verifiable information that can help cor-
rect factual inaccuracies in the reasoning process.
Additionally, we examine the impact of Self Con-
sistency and Self Verification strategies, which fur-
ther enhance the reliability of model outputs. Self
Consistency is a technique where the model gener-
ates multiple candidate answers for a given query,
and the most consistent answer across different
runs is selected. This approach helps reduce ran-
dom errors and ensures that the model’s output
is not overly influenced by any single, potentially
flawed reasoning path. On the other hand, Self
Verification involves an iterative process where the
model checks and refines its own generated an-
swers against predefined correct answers and exter-
nal knowledge sources. This post-hoc validation
step ensures that the model’s outputs are factually
correct by enabling it to reflect on and correct its
own reasoning.
In this work, we are utilizing benchmark meth-
ods to compare the performance of various models
on multiple datasets.(Chen et al., 2024) Specifi-
cally, the models GPT-3.5-Turbo, DeepSeek, and
Llama 2 are evaluated across three major datasets:
HaluEval, TruthfulQA, and FEVER. Each model’s
performance is measured using several metrics, in-
cluding Retrieval-Augmented Generation (RAG),
Chain-of-Thought (CoT), and their combinations
with Self Consistency and Self Verification. (Li
et al., 2025) The results are presented as percent-
ages, allowing us to compare the effectiveness of
each model across these metrics.

2 Related Literature
Chain-of-thought (CoT) reasoning has been shown
to enhance LLM performance on complex tasks.
(Wei et al., 2022) introduced CoT prompting to
help models like GPT-3 generate intermediate rea-
soning steps, improving task accuracy. Similarly,
(Kojima et al., 2022) demonstrated CoT’s effective-
ness on benchmarks like MATH and StrategyQA.
To address hallucination, recent studies have
integrated retrieval-augmented generation (RAG)
with CoT. (Zhou et al., 2023) showed that com-
bining RAG with CoT helps reduce hallucinations
by ensuring the model references relevant external
knowledge. (Liu et al., 2023) focused on refining
retrieval methods to improve CoT’s accuracy and
mitigate hallucinations, while (Singh and Kapoor,
2023) explored how CoT can help track facts dur-
ing open-domain question answering to minimize
hallucinations.
In addition to CoT, recent advancements have
introduced Self Consistency and Self Verification
techniques as key components to reduce hallucina-
tions and improve the factual accuracy of LLMs.
Self Consistency, as explored by (Wang et al.,
2023), emphasizes generating multiple answers
and selecting the most consistent one to enhance
model reliability and accuracy in ambiguous tasks.
Similarly, Self Verification, as proposed by (Weng
et al., 2023), involves an iterative process where the
model verifies its own generated answers against
predefined correct answers and external knowledge
sources, further mitigating the risk of hallucination
and increasing trust in the generated outputs.
3 Novelty & Challenges
3.1 Novelty
This work introduces mainly three different meth-
ods to tackle LLM Hallucinations.
•We tested a combination of several Chain-
of-Thought (CoT) reasoning with Retrieval-
Augmented Generation (RAG), allowing
LLMs to ground their intermediate reasoning
steps in external knowledge. This integration
addresses the challenge of hallucination in
open-ended tasks by anchoring reasoning to
factual sources.
•This method generates multiple reasoning
paths by adjusting the temperature parameter
and aggregates consistent answers, reducing
the risk of unreliable or divergent outputs.•We explore self-verification, where the model
reflects on and critiques its response. This
addresses the challenge of unchecked halluci-
nations by introducing a post-hoc validation
step, improving trustworthiness and factual
alignment.
3.2 Key Challenges
Some of the Key challenges which we faced were
•Generating multiple reasoning paths and ag-
gregating them significantly increases infer-
ence time and resource usage. This makes
deployment of self-consistency techniques ex-
pensive.
•Manual evaluation is time-consuming, and au-
tomated metrics may not fully capture factual
inaccuracies.
•In RAG, irrelevant or low-quality retrieval re-
sults can introduce noise instead of improving
accuracy.
4 Dataset and Approaches
4.1 Dataset
We evaluated hallucination detection across three
datasets: HaluEval, FEVER, and TruthfulQA, each
with distinct structures and evaluation criteria:
4.1.1 HaluEval
HaluEval (qa) (Li et al., 2023) is a benchmark
dataset designed to evaluate hallucination in large
language model outputs. It includes questions or
prompts paired with correct response, hallucinated
response and knowledge to support the correct an-
swer. The dataset that we are using contains 10,000
samples in total, focusing on Open-domain QA.
4.1.2 FEVER
FEVER (v1.0) (Thorne et al., 2018) is a large-scale
benchmark dataset developed to evaluate a model’s
ability to verify factual claims using evidence from
a structured knowledge base (Wikipedia). It is
widely used in fact-checking and evidence-based
reasoning tasks. It consists of a label indicat-
ing whether the claim is Supported, Refuted, Not
Enough Info. It has approximately 145,000 claim-
evidence pairs
4.1.3 TruthfulQA
The TruthfulQA (generation) dataset (Lin et al.,
2021) is a benchmark designed to evaluate the abil-
ity of language models to generate factually correct

and non-misleading answers, particularly in the
presence of common misconceptions or false be-
liefs. It consists of 817 samples, each containing
a question, a list of correct answers, and a list of
incorrect (but often plausible-sounding) answers.
The questions cover a wide range of open-ended
topics, making the dataset suitable for assessing
truthfulness and robustness in language models.
4.2 Dataset Preprocessing
To ensure that the datasets are prepared for the
model evaluation, we performed a series of pre-
processing steps, which include text cleaning and
tokenization. With these steps we standardize the
data for our model. The textual data from the Truth-
fulQA, HaluEval, and FEVER datasets contains
various inconsistencies, such as punctuation varia-
tions, redundant spaces, and case mismatches. We
applied standard text normalization techniques, in-
cluding:
•Lowercasing all text to maintain consistency.
•Removing special characters and excessive
whitespace.
•Stripping leading and trailing spaces. For
compatibility with large language models
(LLMs), we tokenized the textual inputs using
a pre-trained tokenizer from HuggingFace.
•Due to limited hardware we evaluated on 500
samples per dataset.
4.3 Chain of Thought (CoT)
Chain of Thought (CoT) prompting is a technique
that guides language models to reason through a
problem step-by-step before generating a final an-
swer. By structuring the model’s reasoning process,
CoT helps reduce logical errors and hallucination
rates, particularly in tasks that require multi-step
inference or factual grounding.
In our implementation, we explored and tested
various CoT prompt templates such as " Let’s think
step-by-step ", "Think about it like a scientist ", and
"Explain your reasoning before giving the final an-
swer " through prompt engineering to determine the
most effective format for all dataset. Once identi-
fied, we integrated the selected Chain of Thought
prompt as a system-level instruction to the lan-
guage model. This ensured that the model engaged
in intermediate reasoning rather than providing di-
rect answers, leading to more accurate and inter-
pretable responses across different tasks.4.4 RAG
Retrieval-Augmented Generation (RAG) is a
method that enhances the ability of large language
models (LLMs) to produce accurate and context-
aware responses. Instead of relying solely on the
model’s internal knowledge, RAG supplements it
with relevant external documents retrieved based on
the input query. This retrieval step ensures that the
model is grounded in up-to-date or domain-specific
information, improving both factual accuracy and
consistency.
Figure 1: Illustration of Retrieval-Augmented Genera-
tion (RAG) used in our setup.
In our implementation as shown in Figure 1, we
adapted the knowledge gathering strategy based on
the structure and available metadata of each dataset:
1.For the HaluEval dataset, the process was di-
rect. It includes a knowledge field that con-
tains context information aligned with the
question. This field provided sufficient detail
to be passed directly to the LLM as supporting
context during response generation.
2.The FEVER dataset, although lacking a
direct knowledge field, includes an evi-
dence_wiki_url, which refers to Wikipedia
article titles related to the claim. We used
the Wikipedia API to fetch the full content of
these articles and used them as the knowledge
base for this dataset.

3.For TruthfulQA, the challenge was more com-
plex. Each sample contains a source_url,
but these URLs span over 140 different web-
sites with varied structures, making auto-
mated scraping unreliable. To address this,
we prompted an LLM to identify the do-
main or topic of each question (e.g., from
“What happens if someone eats watermelon
seeds?” we inferred the domain as “Water-
melon Seeds”). Using these domain terms, we
queried Wikipedia using its advanced search
features to retrieve the top five most relevant
article titles. The full content of these articles
was then retrieved via the Wikipedia API, pro-
viding consistent and high-quality knowledge
documents.
After gathering the knowledge content, we pro-
cessed it for indexing. Each document was split
into smaller text chunks to ensure compatibility
with the input limitations of our vector database,
Pinecone. We generated embeddings for each
chunk and stored them in the database along with
metadata such as a unique ID, article title, and
when applicable the domain inferred from the
query. The raw text chunks were stored separately
in local storage, indexed by their IDs for efficient
retrieval.
When a query is received, we first generate its
embedding using the same encoder used during
indexing. Based on the dataset the query belongs
to, we direct the embedding to the corresponding
collection in Pinecone. The database returns the
top five most similar document chunks, along with
their metadata. Using the document IDs from the
metadata, we retrieve the original text content from
our local storage.
Finally, we combine the query with the retrieved
document texts and metadata, and feed them into
the LLM. This enables the model to generate re-
sponses that are well-grounded in the context re-
trieved from the knowledge base, thus ensuring
more reliable and informative answers.
4.5 RAG + CoT
Since Retrieval-Augmented Generation (RAG) pro-
vides factual grounding by retrieving external con-
text relevant to a query, and Chain of Thought
(CoT) prompting improves reasoning by encourag-
ing the model to break down its thought process
into intermediate steps thereby reducing logical
errors and hallucinations, we decided to combinethese two strategies into a unified approach. The
goal was to leverage RAG’s strength in evidence-
based context retrieval alongside CoT’s structured
reasoning capability, enhancing the overall accu-
racy and consistency of the model across all three
datasets.
In our combined approach, we first retrieved
supporting knowledge documents using the RAG
pipeline, ensuring that each query was supple-
mented with relevant external context. We then
applied Chain of Thought prompting as a system-
level instruction, prompting the model to reason
through the retrieved context before generating a
response. This integration enabled the model to
not only access supporting information but also
process and interpret it methodically.
4.6 Self Consistency
Self-consistency is a decoding strategy where a
language model generates multiple responses for
the same input and selects the final output based on
the most frequent or consistent answer. This helps
reduce randomness and improves the reliability of
responses, especially in tasks requiring reasoning
or multi-step logic.
Algorithm 1 Self-Consistency Based Hallucination
Detection with Varying Temperature on HaluEval
Require: Input x, Ground Truth y, Language
Model M, Number of samples n= 9, Sim-
ilarity threshold τ= 0.5, Temperature range
Tmin, Tmax
Ensure: Hallucination decision for input x
1:Initialize counters: count factual = 0 ,
count hallucinated = 0
2:fori= 1tondo
3: Sample temperature Ti∼ U(Tmin, Tmax)
4: Generate response ˆyi=M(x;Ti)
5: Compute similarity score si=S(ˆyi, y)
6: ifsi> τthen
7: count factual←count factual + 1
8: else
9: count hallucinated ←count hallucinated + 1
10: end if
11:end for
12:ifcount factual > count hallucinated then
13: return "Non-hallucinated"
14:else
15: return "Hallucinated"
16:end if

Figure 2: CoT Outputs with different prompts
Figure 3: Self Verification Output vs.Base Output
For each input, we sample n= 9responses from
the model to reduce variance and avoid ambiguous
cases, such as a 50-50 split. To promote diverse
reasoning trajectories, we vary the temperature pa-
rameter Tduring decoding, where higher Tvalues
increase output randomness, and lower Tvalues
make the generation more deterministic.
Each output ˆyiis then evaluated against the
ground truth yusing a Cosine Similarity function
S(ˆyi, y)where,
S(ˆyi, y) =⃗ vˆyi·⃗ vy
∥⃗ vˆyi∥ · ∥⃗ vy∥
where ⃗ vˆyiand⃗ vyare the vector embeddings of the
generated and reference answers, respectively. We
came up with a threshold τ= 0.5through trial
and error, if S(ˆyi, y)> τ, the output is considered
factually consistent; otherwise, it is classified as
hallucinated.
A majority voting scheme is then applied over
the 9 samples to determine whether the model, for
that input, produced a valid response or a hallucina-
tion. In Figure 5, we can see the voting under the
full reasoning steps and the final answer that we
received. The algorithm as shown in Algorithm 1was then used for every dataset with changes made
to the labels.
4.7 Self Verification
In our self-verification setup as shown in Figure 4,
we first ask the LLM to generate an answer for a
given query, just like a normal QA setup.
Along with the query, we also have the correct
Figure 4: Self Verification Architecture
answer from the dataset. To check if the model’s
generated answer is trustworthy, we ask the model
to verify its response. This second prompt is given
the original query, the model’s generated answer,

the ground truth answer, and some supporting doc-
uments retrieved from a vector database similar to
how it was done for RAG. These top documents
act as the external context to help the model verify
more accurately. The model used for generation is
then asked to decide whether the generated answer
is factually correct or hallucinated, based on the
given evidence. This extra verification step helps
us reduce hallucinations by allowing the model to
reflect on its outputs in a more informed way. Fig-
ure 3 shows the self verification Output VS Base
Output
5 Experiment
We evaluated hallucination reduction using a step-
wise approach. Starting with baseline LLM out-
puts, we progressively introduce Chain-of-Thought
(CoT) prompting, Retrieval-Augmented Genera-
tion (RAG), self-consistency decoding, and self-
verification. Each step adds reasoning or valida-
tion capabilities to improve factual accuracy. Ex-
periments were conducted across GPT-3.5-Turbo,
LLaMA-2-7b, and DeepSeek-R1 to compare model
behavior under each setting.
5.1 Experimental Settings
We conducted several experimental settings to opti-
mize the performance of our strategies across dif-
ferent datasets.
First, we explored multiple Chain of Thought
prompts to determine which formulation worked
best for our tasks. We tested 3-4 prompt variations
on 20-30 samples per dataset to assess their impact
on the model’s reasoning ability. Outputs from two
of the prompts are shown in Figure 2. While perfor-
mance differences were generally minimal for our
use case, the classic prompt "Let’s think step by
step" yielded the most consistent and interpretable
results across datasets. As such, we adopted it as
our standard CoT prompt for all evaluations.
For the RAG component, we experimented with
different numbers of retrieved documents specif-
ically 2, 5, and 10. Using only 2 documents of-
ten led to incomplete context, while retrieving
10 introduced noise or irrelevant content due to
over-retrieval. We also tested a score-thresholding
strategy, where only documents exceeding a sim-
ilarity threshold were used. However, this led
to retrieval failures for queries with low-scoring
matches. Based on these observations, we settled
on retrieving the top 5 most similar documents,balancing relevance and noise reduction.
Lastly, we tuned the language model’s genera-
tion parameters to optimize response quality across
datasets. We experimented with temperature values
between 0.3 and 0.7 and maximum token limits of
10, 100, and 150. Through these trials, we observed
that a temperature of 0.4 consistently provided a
good balance between determinism and diversity
across all datasets. Since some of the tasks, such as
TruthfulQA, involve open-ended question answer-
ing, we chose a max token limit of 150 to allow
the model enough space to generate complete and
informative responses.
5.2 Baseline LLM
We begin by evaluating different metrics in base-
line LLMs without using techniques like Chain-of-
Thought (CoT) or RAG etc. This serves as a bench-
mark to assess improvements from later methods.
We test models including GPT-3.5-Turbo, LLaMA-
2-7b, and DeepSeek-R1 to examine how hallucina-
tion varies across architectures and how reasoning
or verification strategies affect factual accuracy.
5.3 Evaluation Metrics
In our evaluation, we adopt dataset-specific metrics
tailored to the structure and goals of each bench-
mark:
5.3.1 HaluEval – Hallucination Rate:
We use hallucination rate as the evaluation metric,
which measures the proportion of model outputs la-
beled as hallucinated. Since the dataset provides bi-
nary labels, this metric directly reflects the model’s
tendency to generate false information. A lower
rate indicates better factual accuracy.
Hallucination Rate =Nhallucinated
Ntotal
where Nhallucinated is the number of responses la-
beled as hallucinated, and Ntotalis the total number
of samples.
5.3.2 FEVER – Accuracy:
For FEVER, we evaluate using label accuracy,
which assesses the percentage of claims correctly
classified into one of three categories: Supported ,
Refuted , orNot Enough Info . This metric reflects
the model’s ability to perform evidence-based fact
verification.
Accuracy =Ncorrect
Ntotal

Figure 5: Sample Working of Self Consistency on HaluEval Dataset on our website
where Ncorrect is the number of correctly classified
claims.
5.3.3 TruthfulQA – MC2
For the TruthfulQA dataset, which involves evalu-
ating open-ended generated responses against sets
of correct and incorrect answers, we adopted an
automated strategy inspired by the MC2(Multiple
Choice - 2 Options) algorithm. To enable scalable
evaluation, we assigned a label of 1 to all correct
answers and 2 to all incorrect ones.
For each generated response, we computed co-
sine similarity with every correct and incorrect
answer using sentence embeddings. The label
corresponding to the highest similarity score was
assigned to the response, indicating whether it
aligned more closely with correct or incorrect infor-
mation. Using these predicted labels, we calculated
the model’s truthfulness score as the proportion of
correctly identified responses:
Truthful Accuracy =PN
i=1(ˆyi=yi)
N
where ˆyiis the predicted label, yiis the true label,
andNis the number of samples.
6 Results, Findings, and Insights
In this section, we present the performance of dif-
ferent hallucination mitigation techniques across
the HaluEval, FEVER, and TruthfulQA datasets.
We compare baseline outputs with Chain-of-
Thought (CoT), Retrieval-Augmented Generation
(RAG), self-consistency, and self-verification meth-
ods across multiple LLMs, including GPT, LLaMA,
and DeepSeek.6.1 Results
•For HaluEval(Figure 6), the RAG + CoT ap-
proach using the GPT-3.5-Turbo model, along
with the Self-Verification method, achieved
the lowest hallucination rate of 11%, indicat-
ing strong performance in reducing factual
errors.
•For FEVER(Figure 7), the Self-Verification
strategy yielded the highest accuracy, reaching
approximately 90%.
•For TruthfulQA(Figure 8), Self-Verification
also attained the highest MC2 score, with a
value of around 80%, demonstrating its effec-
tiveness in distinguishing truthful responses.
6.2 Findings and Insights
Figure 6: HaluEval Results
The following are the finding that we can see:
•All the methods performed better than base-
line models.

Figure 7: FEVER Results
Figure 8: TruthfulQA Results
•Every method CoT, RAG, RAG + CoT, Self-
Consistency, and Self-Verification shows im-
provements over the base model across all
datasets. This validates that hallucination mit-
igation strategies, whether through reasoning,
retrieval, or verification, improve significant
performance.
•One of the key findings is that RAG, RAG
+ CoT, and Self-Verification consistently per-
form well, likely due to the use of external
knowledge, which helps ground responses and
reduce hallucinations in challenging settings.
•Combining RAG and CoT outperforms in-
dividual methods, mainly because retrieval
provides factual grounding while CoT guides
step-by-step reasoning, leading to more accu-
rate and structured responses, especially for
complex tasks.
•RAG + CoT and Self Verification have a com-
parable performance with Self-Verification
providing slightly better results in FEVER and
Truthful QA whereas RAG + CoT provides
better hallucination rate in halu eval.
•Overall, Self-Verification had the best perfor-mance, with LLaMA-2 slightly outperform-
ing GPT-3.5-Turbo. This may be attributed
to LLaMA’s open-weight architecture which
makes it more adaptable to verification-style
prompts, helping it better evaluate and cor-
rect its outputs. It may also be less prone
to overconfident generation, leading to fewer
hallucinations compared to GPT models.
7 Future Directions
While our current approach shows promising re-
sults in reducing hallucinations across various lan-
guage models and benchmarks, there are several
directions for future work.
•We can extend the hallucination framework to
multilingual LLMs and assess whether tech-
niques hold across non-English languages.
•Currently, the self-verification architecture
uses the same LLM for both generation and
verification. In future work, this setup could
be extended by using different LLMs for the
verification step to assess cross-model consis-
tency and robustness.
•Future work can focus on improving the qual-
ity of retrieved documents by implementing
dense passage retrieval (DPR) with query re-
formulation and fine-tuning embedding mod-
els for domain-specific relevance. This would
help reduce noise and enhance the factual con-
sistency of model outputs.
•To optimize reasoning, a dynamically adapt-
ing Chain of Thought prompts based on the
query type can be used. Techniques like re-
inforcement learning could be used to train a
prompt selector that chooses the most suitable
reasoning style depending on the domain and
complexity of the question.
•To reduce the computational cost of self-
consistency sampling, a future direction in-
volves integrating an early stopping mecha-
nism that terminates response generation once
a certain number of consistent outputs are de-
tected. This can be achieved using similarity
to identify convergence among sampled re-
sponses.

References
Xiang Chen, Duanzheng Song, Honghao Gui, Chenxi
Wang, Ningyu Zhang, Yong Jiang, Fei Huang,
Chengfei Lv, Dan Zhang, and Huajun Chen. 2024.
Factchd: Benchmarking fact-conflicting hallucina-
tion detection. arXiv preprint arXiv:2310.12086 .
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang,
and Haofen Wang. 2024. Retrieval-augmented gen-
eration for large language models: A survey. arXiv
preprint arXiv:2312.10997 .
T. Kojima, Y . Tsuchiya, and A. Ogawa. 2022. Chain-of-
thought reasoning can enhance llm performance on
complex tasks. arXiv preprint , arXiv:2201.11903.
Feiyang Li, Peng Fang, Zhan Shi, Arijit Khan, Fang
Wang, Dan Feng, Weihao Wang, Xin Zhang, and
Yongjian Cui. 2025. Cot-rag: Integrating chain of
thought and retrieval-augmented generation to en-
hance reasoning in large language models. arXiv
preprint arXiv:2504.13534 .
Junyi Li, Xiaoxue Cheng, Wayne Xin Zhao, Jian-Yun
Nie, and Ji-Rong Wen. 2023. Halueval: A large-
scale hallucination evaluation benchmark for large
language models. arXiv preprint arXiv:2305.11747 .
Stephanie Lin, Jacob Hilton, and Owain Evans. 2021.
Truthfulqa: Measuring how models mimic human
falsehoods. arXiv preprint arXiv:2109.07958 .
Y . Liu, W. Chen, and J. Gao. 2023. Mitigating hal-
lucination in retrieval-augmented chain-of-thought
reasoning. arXiv preprint , arXiv:2303.08896.
Humza Naveed, Asad Ullah Khan, Shi Qiu, Muhammad
Saqib, Saeed Anwar, Muhammad Usman, Naveed
Akhtar, Nick Barnes, and Ajmal Mian. 2024. A
comprehensive overview of large language models.
arXiv preprint arXiv:2307.06435 .
S. Singh and A. Kapoor. 2023. Hallucinations
in open-domain question answering: Solutions
through chain-of-thought reasoning. arXiv preprint ,
arXiv:2407.07071.
James Thorne, Andreas Vlachos, Christos
Christodoulopoulos, and Arpit Mittal. 2018.
Fever: a large-scale dataset for fact extraction and
verification. arXiv preprint arXiv:1803.05355 .
Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le,
Ed Chi, Sharan Narang, Aakanksha Chowdhery, and
Denny Zhou. 2023. Self-consistency improves chain
of thought reasoning in language models. arXiv
preprint arXiv:2203.11171 .
J. Wei, P. Wang, D. Schuurmans, M. Bosma, and
D. Chen. 2022. Chain-of-thought prompting elicits
reasoning in large language models. arXiv preprint ,
arXiv:2201.11903.Yixuan Weng, Minjun Zhu, Fei Xia, Bin Li, Shizhu
He, Shengping Liu, Bin Sun, Kang Liu, and Jun
Zhao. 2023. Large language models are better
reasoners with self-verification. arXiv preprint
arXiv:2212.09561 .
W. Zhou, H. Wang, and Z. Yu. 2023. Reduc-
ing hallucinations in retrieval-augmented generation
with chain-of-thought prompting. arXiv preprint ,
arXiv:2305.13534.