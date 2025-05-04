# EnronQA: Towards Personalized RAG over Private Documents

**Authors**: Michael J. Ryan, Danmei Xu, Chris Nivera, Daniel Campos

**Published**: 2025-05-01 03:07:30

**PDF URL**: [http://arxiv.org/pdf/2505.00263v1](http://arxiv.org/pdf/2505.00263v1)

## Abstract
Retrieval Augmented Generation (RAG) has become one of the most popular
methods for bringing knowledge-intensive context to large language models (LLM)
because of its ability to bring local context at inference time without the
cost or data leakage risks associated with fine-tuning. A clear separation of
private information from the LLM training has made RAG the basis for many
enterprise LLM workloads as it allows the company to augment LLM's
understanding using customers' private documents. Despite its popularity for
private documents in enterprise deployments, current RAG benchmarks for
validating and optimizing RAG pipelines draw their corpora from public data
such as Wikipedia or generic web pages and offer little to no personal context.
Seeking to empower more personal and private RAG we release the EnronQA
benchmark, a dataset of 103,638 emails with 528,304 question-answer pairs
across 150 different user inboxes. EnronQA enables better benchmarking of RAG
pipelines over private data and allows for experimentation on the introduction
of personalized retrieval settings over realistic data. Finally, we use EnronQA
to explore the tradeoff in memorization and retrieval when reasoning over
private documents.

## Full Text


<!-- PDF content starts -->

EnronQA: Towards Personalized RAG over Private Documents
Michael J. Ryan∗
Stanford University
Palo Alto, California, USAChris Nivera
Snowflake
San Mateo, California, USA
Danmei Xu
Snowflake
San Mateo, California, USADaniel Campos
Snowflake
San Mateo, California, USA
…User 1User 2User 3User 150
Search 1242 emails What is the name of the person who is looking into the missing rates issue for Powder River Gathering, according to Rhonda Robinson's email?
Theresa.
What is the deadline for completing the feedback forms for the Year End 2000 Performance Management process?Search 8429 emails 
Friday, November 17, 2000.103,638 emails; 528,304 QA Pairs; 150 UsersSingle User Retrieval
Multi User Retrieval
Figure 1: The EnronQA benchmark enables personalized and private retrieval benchmarking on a cleaned corpus of over 100,000
emails spanning 528,304 quality question-answer pairs over 150 users. We explore both single and multi-user retrieval settings.
Abstract
Retrieval Augmented Generation (RAG) has become one of the
most popular methods for bringing knowledge-intensive context
to large language models (LLM) because of its ability to bring local
context at inference time without the cost or data leakage risks as-
sociated with fine-tuning. A clear separation of private information
from the LLM training has made RAG the basis for many enterprise
LLM workloads as it allows the company to augment LLM’s under-
standing using customers’ private documents. Despite its popular-
ity for private documents in enterprise deployments, current RAG
benchmarks for validating and optimizing RAG pipelines draw their
∗Work completed during Snowflake internship. Email: michaeljryan@stanford.edu
This work is licensed under a Creative Commons Attribution 4.0 International License.corpora from public data such as Wikipedia or generic web pages
and offer little to no personal context. Seeking to empower more
personal and private RAG we release the EnronQA benchmark, a
dataset of 103,638 emails with 528,304 question-answer pairs across
150 different user inboxes. EnronQA enables better benchmarking of
RAG pipelines over private data and allows for experimentation on
the introduction of personalized retrieval settings over realistic data.
Finally, we use EnronQA to explore the tradeoff in memorization
and retrieval when reasoning over private documents.1
1 Introduction
Retrieval is increasingly one of the most common ways to add con-
text to LLMs in a process called Retrieval Augmented Generation
(RAG) [ 20,22,43]. RAG pipelines involve augmenting the natural
1All data released on this Huggingface repo: MichaelR207/enron_qa_0922arXiv:2505.00263v1  [cs.IR]  1 May 2025

Ryan et al.
language generation capabilities of an LLM with an external data
store. This enhancement improves the factuality [ 5,66] and the
explainability [ 68,75] of the LLM by grounding the generation in
documents. Furthermore, RAG has been shown to help LLMs solve
knowledge-intensive tasks [ 36] by retrieving relevant knowledge
rather than relying on the LLM to memorize facts. One of RAG’s
most popular applications is retrieval over private documents, en-
abling companies and users to interact with vast stores of internal
and private knowledge [14, 28].
Although retrieval over private documents is one of RAG’s most
popular use cases, comparatively few large-scale RAG benchmarks
focus on private document retrieval [ 4]. Most popular benchmarks
for validating and optimizing RAG pipelines draw their corpora
from Wikipedia [ 32,41,77] or the public internet [ 6]. We discuss
this further in §2.1.
Additionally, there is recent interest in privacy-preserving RAG
[78,79] where it is important for a model to be able to assist and
access knowledge from private documents without exposing person-
ally identifiable information. Personalized RAG over private docu-
ments [ 21,72,80] and federated information retrieval [ 58,58,65,71]
require personal documents and segmented data stores. Such explo-
rations would benefit from a realistic corpus segmented into several
private users for developing measured approaches to private and
personalized RAG.
To serve these diverse tasks and increase coverage of bench-
marking in private settings, we introduce the EnronQA benchmark
based on the Enron emails corpus [ 17].EnronQA contains 103,638
emails with 528,304 question-answer pairs, spanning 150 distinct
user inboxes. By designing a rigorous question generation pipeline
grounded in specific evaluations we ensure a collection of high
quality and diverse questions. Our QA dataset is unmatched in size
for openly available document retrieval over private documents and
is large enough to enable finetuning, optimization, and evaluation
over this setting. Figure 1 showcases the EnronQA benchmark and
some evaluation settings we test.
To showcase the utility of EnronQA we perform two case studies.
First, we showcase how benchmarking RAG pipelines on EnronQA
has higher headroom for improving retriever quality ( ??). We find
that without retrieval, RAG pipelines score below 5% on EnronQA,
unlike other popular RAG benchmarks where, using parametric
LLM knowledge, it’s possible to score above 60% without any re-
trieval at all. Next we train LoRA adapters to memorize factual
knowledge in our large dataset. Our LoRA adapters for memoriza-
tion reveal that training LLMs to memorize private factual knowl-
edge can perform on par with storing all facts in context, however
retrieving the specific relevant information still outperforms both.
Overall, our contributions are as follows:
(1)TheEnronQA benchmark, a collection of over 100,000 pri-
vate emails and 500,000 questions, segmented into 150 dis-
tinct user inboxes (§3).
(2)We showcase the quality and utility of our benchmark and
compare it with other popular RAG benchmarks (§4.2). We
benchmark popular RAG pipelines on EnronQA as a baseline
for future work (§5).(3)We motivate memorizing private knowledge and show-
case a LoRA-based method for memorizing factual knowl-
edge, which performs competitively to putting a knowledge
base in context (§6). Retrieving the most relevant informa-
tion outperforms both, however we discuss future improve-
ments that motivates further exploration into memorization
adapters.
2 Related Work
We organize our discussion of related work to span our core contri-
butions: benchmarking RAG, and factual memorization in LLMs.
2.1 Retrieval Augmented Generation
Benchmarking
We provide a brief list of popular RAG QA Benchmarks in Table 1
and a more comprehensive list in Table 6. Several common RAG
benchmarks draw documents from Wikipedia [ 2,33,35,41,74,77].
This unfortunately makes the benchmarks less suitable for bench-
marking RAG pipelines using modern LLMs which have memorized
a lot of the contents of Wikipedia and general knowledge [57].
The most related work to ours is ConcurrentQA [4], which cre-
ates a benchmark that relies on multihop reasoning over both the
Enron emails corpus as well as Wikipedia. We consider this an
excellent resource in conjunction with ours, however it is worth
noting that these benchmarks solve distinct problems. For one, the
ConcurrentQA benchmark limits to just one inbox, while EnronQA
spans 150 distinct users, enabling the study of personalized RAG
within the larger benchmark. Second, ConcurrentQA has 18.4k QA
pairs rather than the 528.3k in EnronQA , which makes our bench-
mark more suited for explorations involving fine-tuning factual
knowledge, and continued pretraining. EnronQA will enable the ex-
ploration of these emerging trends in information retrieval. Finally,
ConcurrentQA is comprised of multi-hop queries, which is very
useful for benchmarking sophisticated pipelines. We instead design
EnronQA to be single hop to focus on the interesting cases of per-
sonalization and memorization, however in section 3.1 we discuss
how we make EnronQA fully compatible with ConcurrentQA so
users can benchmark both single and multi-hop retrieval using our
benchmark.
2.2 Factual Memorization
In our case study we explore factual memorization as an alternative
to traditional retrievers for recalling facts. Factual memorization
in LLMs is an exciting and relatively new direction. Much of the
work in LLM memorization comes from work studying unlearn-
ing [ 45,48] in LLMs or understanding memorization from an in-
teroperability lens [ 26]. Some recent works have looked towards
strategies for encouraging memorization in LLMs. One approach
is augmenting LLMs with external memory parameters [ 12,23].
Other work encourages fine-tuning of LLMs for factual memoriza-
tion [46, 62, 69].
Perhaps the most relevant and exciting connection specifically
to information retrieval is continued pretraining [ 24,37] where to
adapt LLMs to new knowledge and domains, it is possible to just
continue the pretraining process on more documents from the new
source. The most promising connection to our setting is "Synthetic

EnronQA: Towards Personalized RAG over Private Documents
Benchmark Documents QA Pairs Domain Source
ConcurrentQA [4] 5.2M + 47k 18.4k General and Private Knowledge Wikipedia + Emails
EManual [54] 308k 3.3k Customer Support TV Manual
HAGRID [35] 32.8M 2.6k General Knowledge Wikipedia
HotPotQA [77] 5.2M 112.8k General Knowledge Wikipedia
INSCIT [74] 6.6M 4.7k General Knowledge Wikipedia
MS Marco [6] 3.6M 1.01M General Knowledge Web Pages
Natural Questions [41] 5.9M 323k General Knowledge Wikipedia
PubMedQA [31] 211.3k 273.5k Academic Research Research Abstracts
QReCC [3] 10M 81k General Knowledge Web pages
SearchQA [16] 6.9M 140.5k General Knowledge Google Search
TechQA [8] 802k 1.4k Customer Support Tech Forums
TopiOCQA [2] 5.9M 50k General Knowledge Wikipedia
TriviaQA [33] 662.7k 96k General Knowledge Wikipedia
EnronQA (Ours) 103.6k 528.3k Private Knowledge Emails
Table 1: Comparison of document-based QA benchmarks. We limit to only resources with a corpus size above 50k documents for
space, but provide a full comparison in Appendix A. EnronQA covers a comparable corpus scale to many popular QA benchmarks
while having vastly more QA pairs enabling training, optimization, and document memorization exploration. Additionally,
EnronQA spans the under explored private document domain using emails.
Continued Pretraining" [ 76] wherein entities are extracted from
documents, and connections are drawn between those entities.
Then, the LLM is continuously pre-trained on these connections. In
doing so, the authors find that this encourages memorization of the
documents, and ultimately, when tested with RAG, the performance
benefits compound. In this work, we release a benchmark of private
and unmemorized documents to benchmark RAG performance.
Such a resource will be a rich test bed, providing a realistic QA task
for continued pretraining methods.
3EnronQA Dataset Construction
We construct the EnronQA benchmark using the Enron emails cor-
pus [ 39]. The Federal Energy Regulatory Commission released the
original corpus during the Western Energy Markets investigation
in 2003. The original dataset contained over 600,000 messages and
158 distinct users (inboxes). We use the 2015 release of the corpus,
which has been cleaned and had emails removed at the request of
Enron participants [ 17]. This version of the corpus contains 517,401
emails across 150 users. To convert the raw emails into a high-
quality RAG benchmark, we devise a 3-stage pipeline: Filtering
(§3.1), QA Generation (§3.2), Post Processing (§3.3).
3.1 Corpus Filtering
For our data filters, we take inspiration from popular pretraining
data filtering pipelines [ 13,19,59,67]. Data filtering from raw emails
draws many parallels to cleaning unstructured web data. Table 2
outlines each step of the filtration process alongside the number of
emails removed and example subjects of emails removed.Deduplication. Web data extraction pipelines use minhash dedu-
plication [ 42] to reduce the number of identical documents and
documents with minimal edits, such as software licenses. In email
inboxes, this can correspond with both email threads and email
subscriptions. For example, a subscription to a weather service
might send the same email each day, with the forecast changed.
We modify the text-dedup [ 52] library’s minhash implementation
to add a final step where the Jaccard similarity between matched
documents is computed, and we run minhash deduplication with
a Jaccard similarity threshold of 0.9, using 9 bands and 27 rows.
To handle thread deduplication, we also remove any emails that
appear in their entirety within the content of another email which
we call "subset" deduplication.
Gopher Quality Filters. [59] outline a few calculations to filter
pretraining documents for quality. They filter based on document
word length, mean word length, number of lines ending in ellipsis,
and the ratio of alphanumeric characters to symbols. Tuning the
cutoff for each of these rules for our email domain, we set the cutoffs
to emails with between 50 and 1000 words, a length between 3 to
10, the ratio of alphanumeric characters to symbols greater than or
equal to 0.65, and fewer than 10% of lines ending in ellipsis. This
helps filter out excessively short emails and excessively long or
low-quality emails, such as automated log files for software systems
and automated financial reports.
Language Identification. Some email threads between Enron em-
ployees occur in non-English languages. Since we are creating an
English benchmark using LLMs with limited multilingual train-
ing, we filter out documents classified as non-English or classified

Ryan et al.
Step Operation Documents Change Example Subjects of Affected Emails
0 (Original) 517,401 – –
1 Minhash Dedup 228,098 -289,303 "daily charts and matrices as hot links 5/15", "Enron Mentions - 05/12/01 - 05/13/01"
2 Subset Dedup 192,759 -35,339 "FW: FW: Research Library", "Re: FW: Fedex", "Pre-Party??", "Re: Reference"
3 Quality Filter 103,513 -89,246 "RE: 606925 Finney Attasseril SRF (mar)", "Copyright Notice for EnronOnline"
4 Language ID 103,164 -349 "Praca dyplomowa", "Saludos, Feicitaciones y Follow up."
5 NSFW Filter 103,015 -149 "RE: bs", "dO yOU wANT tO pLAY wITH mE?", "Re: Richard Sucks!!!!!!"
6 Toxicity Filter 102,603 -412 "Re: LSU/Florida", "FW: Isn’t it Funny", "I AM FEELING NEGLECTED"
7 ConcurrentQA 103,638 +1,035 "Internet Daily for November 16, 2001", "California Energy Crisis Timeline"
Table 2: Enron Email Corpus throughout several steps of corpus filtering along with samples of subjects from emails added/re-
moved at each step.
as English with below 80% confidence with a fastText language
identification model [34].
NSFW/Toxicity Filters. Surprisingly, we identified some toxic and
inappropriate content as the subject of some emails. To preserve the
participants’ privacy and maintain professionalism in our EnronQA
resource, we filter out these documents. We use two fastText clas-
sifiers trained on jigsaw [ 1] for the dolma project [ 67], one for
detecting toxicity and one for detecting NSFW content. We filter
out emails that are not predicted as safe with greater than or equal
to 90% confidence.
ConcurrentQA. To make our resource compatible with the related
ConcurrentQA [ 4] benchmark, we map all of the documents in the
ConcurrentQA Enron corpus back to emails in our corpus. If these
rules filtered out any, we add them back in.
3.2 QA Generation Pipeline
We must generate high-quality questions to convert the cleaned
corpus into a RAG benchmark. We devise a multi-stage compound
LLM system implemented and optimized end-to-end in DSPy [ 38].
The generation of a single question comprises between 10 and 50
distinct LLM calls each designed to serve a single modular purpose.
Our pipeline is described visually in Figure 2, and can be concep-
tualized as 4 main stages: (1) Initial generation, (2) Evaluation, (3)
Feedback generation, (4) Refinement.
Initial Generation. We generate an initial question given a doc-
ument and a set of prior questions for that document (so that the
LLM does not generate repeats). Our question generator model
is llama3.1-70b [ 15] with a prompt optimized by DSPy using the
MIPROv2 optimizer [ 55]. Our optimization objective was to reduce
the number of refinement steps from our early stopping pipeline.
The pipeline repeats if any of the Evaluation steps described below
fail. Our optimized prompts brought the pipeline from an average
of 1.94 repetitions down to 1.64 repetitions. We include our initial
and optimized prompt in appendix B.1.
Evaluation. Given a question, we want to assess whether the
question is high quality. To do this, we introduce four evaluation cri-
teria and concrete measures: Specificity, Objectivity, Groundedness,
and Quality.Specificity. We designate a question as "specific" if given
ten similar emails (including the true email the question is
about) an LLM can pick out which email would answer the
question. We mine hard negative examples by retrieving
the top 10 relevant documents from our corpus given the
question. We use a biencoder built on Snowflake arctic-
embed-m-v1.5 [ 50] to retrieve the top 10 most similar doc-
uments. We use Llama3.1 70b as our selector LLM. The full
prompt is provided in Appendix B.2.
Objectivity. We determine a question to be "objective" if
two models from different families answer the same ques-
tion with the same answer, given the email as context. Here
we use Llama3.1 70b Instruct [ 15] and Mixtral 8x7B Instruct
[30]. We use an LLM as a judge to determine if the answers
match. Our Llama3.1 70B Instruct LLM judge achieves a
0.98 F1-score with human evaluation on a small study of
200 generations. We include all QA prompts in Appendix
B.5 and details of our LLM as a judge evaluation in Appen-
dix B.6. Given the question is deemed objective, we save
the Llama3.1 70B Instruct output as the "gold answer."
Groundedness. We determine a question to be "grounded"
if neither the llama nor mixtral model can answer the same
as the gold answer given no email as context. This both
tests that the answers to our questions are not memorized
and that the questions are not easily guessable. Again, we
use the LLM as a judge, grounded in the email, to determine
if the answers match the "gold answer" obtained from the
previous evaluation. If neither ungrounded answer matches
the gold answer, we deem the question "grounded." All QA
prompts are included in Appendix B.5.
Quality. Our last evaluation step is measuring the "quality"
of the question by an LLM judge aligned with human judg-
ments. We generated 20 questions using the pipeline with
only the specificity, objectivity, and groundedness stages.
We had two authors label them as "high, " "medium, " or "low"
quality based on a rubric assessing specificity, objectivity,
and groundedness. The authors had a 0.5 Spearman corre-
lation of their annotations, and a third author adjudicated
the disagreements. That same author also independently
labeled 21 more questions as "high", "medium", or "low".
Using the 21 singly-labeled questions as a development set

EnronQA: Towards Personalized RAG over Private Documents
email70bQuestion (Q)
Objective?Grounded?Quality?
Specific?Q
Retriever
Which email?Q
A
AQSame?AQ
A
AQA
Not Same?Not Same?Q
RulesQuality Question?“Not Specific”“Not Objective”“Not Grounded”“Needs to follow rule 3: …”
EvaluationFeedbackQQ
QRepeat Up to 5x
123234
Figure 2: Our multistage compound LLM system for QA Generation on the Enron emails corpus. Our pipeline consists of 4 stages
labeled in the diagram above: (1) Initial generation, (2) Evaluation, (3) Feedback generation, and (4) Refinement. Producing a
single high-quality question takes 10-50 distinct LLM calls, and the system is optimized end-to-end. Our pipeline asserts that
questions are specific, objective, grounded, and high-quality (correlating with human judgment). All Llama icons correspond to
Llama3.1 70b Instruct [ 15], the Mistral icon represents the Mixtral-7B-Instruct model [ 30], and our retriever is a bi-encoder
using Snowflake’s artic-embed-m-v1.5 [50].
and the 20 group-labeled questions as a test set, we devised
a list of rules for Llama-3.1 70B Instruct to use to determine
if a question was "high" or "low" quality (wrapping the
"medium" label into low quality). Our ruleset enabled the
judge to achieve 85.7% accuracy on the development set,
and running it once on the test set yielded 85% accuracy.
The final stage of our evaluation pipeline uses a Llama3.1
70B Instruct model augmented with a ruleset to determine
if the question is "high quality." We include our ruleset in
Appendix B.7.
Feedback Generation. Based on the latest stage that the question
made it to in the evaluation phase, we produce feedback to add to
the context of the refinement step.
•If the question is not specific, we handle this in a special
case described in the "Refinement" step.
•If the question is not objective, we provide the feedback:
"Question is not objective. Different annotators answer the
same question differently given the same email as context.
Could benefit from more clarity."
•If the question is not grounded, we provide the feedback:
"Question is not grounded. It is too easy to guess the answer
to this question without having read the email."
•If the question fails the quality check, we use the chain-of-
thought reasoning of the LLM as the feedback for why the
question is not of high quality. This typically cites which
rule the email fails, briefly explaining why.
Refinement. If our question succeeds at all evaluation stages,
it is considered a good question and added to our question bank.
Otherwise, we need to refine it. We have two refinement steps. If the
QA
Q’
AA
Same?Figure 3: Question Rewrite Pipeline. First, we ask Llama 3.1
70B Instruct to rewrite the question, and then we ask it to
answer this new question. Finally, we use Llama 3.1 70B In-
struct to check that the answers match.
question is not specific, we show the LLM the ten retrieved emails
from the specificity check and ask it to rewrite the question to be
more specific to only the gold email. If the question fails the other
steps, we use our generated feedback to ask the LLM to rewrite the
question and address the feedback. Both the specificity and general
feedback rewrite prompts are optimized using DSPy. We include
the initial and optimized specificity refinement prompt in Appendix
B.3 and the initial and optimized feedback prompt in Appendix B.4.
3.3 Additional Data Processing
We provide a rewritten version of our questions to make our dataset
more practical for various downstream tasks, such as our memo-
rization case study (§6). In the case of training LLMs to memorize
specific information, this enables you to train and test different
questions while retaining the informational content. In Figure 3, we
showcase our pipeline for rephrasing questions. We use Llama3.1-
70B-Instruct to rewrite the question, answer the rewritten question,

Ryan et al.
and finally judge if the answer is the same. If the answers don’t
match, we try again up to 15 times before discarding the question.
We discard 265/528,569 questions in this process.
Alongside the core components of our dataset: the questions,
gold answers, emails, and rephrased questions, we also release mis-
cellaneous artifacts produced in creating the core dataset. These
artifacts include the verified answers of the Mixtral-8x7B-Instruct
model from the evaluation step, as well as the chain of thought rea-
soning for both Llama3.1-70b-Instruct and Mixtral-8x7B-Instruct as
they answer each question in the EnronQA benchmark conditioned
on the oracle document.
4 Dataset Quality
We discuss here some of the properties of the EnronQA benchmark
and what makes it a high quality and valuable resource to the
community.
4.1 Dataset Statistics
We report summary statistics for the EnronQA benchmark in Table 3.
Notably the benchmark contains over 333k training questions, and
the median number of questions about a single user’s emails is over
1k. The EnronQA benchmark is suitably large for fine-tuning on
questions, continuously pretraining on documents, and benchmark-
ing RAG pipelines.
Metric Train Dev Test
Email Stats (consistent across splits)
Average emails per user 491.81
Median emails per user 240.5
Mean email length (chars) 2,269.69
Median email length (chars) 1,664.0
Question Stats
Mean questions per user 2,223.15 703.43 595.44
Median questions per user 1,063.0 334.5 278.0
Mean questions per email 4.52 1.43 1.21
Median questions per email 3.0 1.0 1.0
Total questions count 333,473 105,515 89,316
Table 3: Summary statistics EnronQA benchmark. The bench-
mark contains suitably large amounts of documents and
questions for continued pertaining and RAG benchmarking.
4.2 Calibration
One downside to using common RAG benchmarks, which pull doc-
uments from Wikipedia, is the lack of calibration between bench-
mark scores and retrieval quality. A comparative advantage to
EnronQA is that, for the most part, the parametric knowledge en-
coded in the LLMs will not memorize the Enron emails. Thus, you
can expect gains from a better retriever to match gains in accu-
racy on the benchmark. To test this hypothesis, we choose two
standard RAG benchmarks NaturalQuestions [41] and TriviaQA
[32]. NaturalQuestions comprises 323,000 queries to Google, with
answers spanning 5.9M Wikipedia documents. TriviaQA contains
95K question-answer pairs authored by trivia enthusiasts and over600 thousand articles. We specifically use the KILT [ 56] versions of
the datasets.
Experimental Setting. We filter to 10,000 training / 500 validation
examples for NaturalQuestions and TriviaQA. And 1,000 training /
500 validation examples for EnronQA. We use the full validation
sets of NaturalQuestions and TriviaQA as the test set. For EnronQA,
we use the actual test set. We optimize two DSPy programs for each
setting using the MIPROv2 optimized [ 55]. The first program takes
no context and has to answer the question directly. The second
program takes the gold document as context and answers the ques-
tion. We optimize with Llama-3.1-8B-Instruct as our task model
and Llama-3.1-70B-Instruct as our prompt model with 10 candidate
programs. When running the experiment, we use Llama-3.1-70B-
Instruct. We run the no-context case; then, we simulate Recall@1
between 0.0 and 1.0 by randomly including the correct document
as context or a randomly sampled document instead. We tested
with five random seeds and averaged the results. Scores are pro-
duced using Llama-3.1-70B-Instruct as a judge for answer accuracy
compared to the gold answer.
Results. We include results of the experiment in Figure 4. We find
thatEnronQA is the only benchmark where adding context is always
better than the no-context baseline. For NaturalQuestions it takes a
retriever with a Recall@1 above 0.5 in order to outperform the no
context baseline. Likewise on TriviaQA the problem is even worse.
Just asking Llama-3.1-70B-Instruct the question directly without
context outperforms all retrieval-based systems with Recall@1 less
than 0.85! This means that any accuracy changes on RAG pipelines
benchmarking on TriviaQA with Recall@1 less than 0.85 may have
more to do with the memorized knowledge of the LLM rather than
retrieval quality.
In contrast, with EnronQA all improvements with more accurate
context lead directly to higher accuracy on the benchmark. Addi-
tionally, EnronQA showcases the highest improvement in accuracy
for every point of recall gain. Nearly a 0.6% gain in accuracy for ev-
ery 1% increase in recall. This is because the knowledge in EnronQA
has not been memorized by large foundation models, which is the
problem that trivializes NaturalQuestions and TriviaQA.
5 Benchmarking RAG Pipelines
To offer some baseline performance numbers and show off the
utility of EnronQA for RAG benchmarking, we test a sweep of two
popular retrievers, three popular LLMs, and two common RAG
pipeline architectures.
5.1 Experimental Setting
Retrievers. We test BM25 using the PySerini implementation [ 44]
andColBERTv2 [64] over the full set of 103,638 emails. We retrieve
five documents simultaneously for a single call to the retriever. For
each retriever, we also report Recall@5.
Large Language Models. We test with Llama-3.1-8B-Instruct ,
Llama-3.1-70B-Instruct [15], and GPT4o to test models of differ-
ent scales and families.
RAG Architectures. We test two RAG settings; first, we test No
Query Rewrite , where we search the query directly using the

EnronQA: Towards Personalized RAG over Private Documents
0.0 0.2 0.4 0.6 0.8 1.0
Simulated Recall@11020304050607080Accuracy (LLM Judge)
NaturalQuestions
0.0 0.2 0.4 0.6 0.8 1.0
Simulated Recall@1
TriviaQA
0.0 0.2 0.4 0.6 0.8 1.0
Simulated Recall@1
EnronQAIndividual Runs Average No Context
Figure 4: Calibration experiment results. Although all benchmarks scale roughly linearly with more accurate context, EnronQA
is the only benchmark where adding context always outperforms the no-context baseline. For TriviaQA, it takes Recall@1 of
nearly 0.85 to surpass the performance of the no-context baseline.
No Query Rewrite (acc) Query Rewrite (acc) Query Rewrite (R@5)
R@5 Llama8B Llama70b GPT4o Llama8B Llama70b GPT4o Llama8B Llama70b GPT4o
BM25 87.5 72.1 77.1 81.2 66.1 80.0 81.1 82.5 86.6 88.5
ColBERT 54.1 65.8 74.6 74.9 56.3 69.1 73.3 54.1 59.4 59.3
Table 4: Benchmarking several retrieval methods and LLMs on the EnronQA benchmark both with and without query rewriting.
Surprisingly, simple retrieval baselines (BM25) work well on our benchmark. This is likely due to some lexical overlap between
the queries and proper nouns in the emails, such as names and events.
question directly from EnronQA . We then provide the top 5 results
from the retriever and pass the retrieved documents and questions
to the LLM to be answered. We additionally test the Query Rewrite
setting where we first have the LLM rewrite the question into a
search query. Then, we retrieve five emails. Finally, given the five
emails and the question, we have the LLM produce the answer.
Prior works have found query rewriting with LLMs to help with
adapting to the specifics of a particular retriever [ 47] so we test
this on our benchmark. For both settings and with all models and
retrievers, we optimize the prompts and few-shot demonstrations
using DSPy MIPROv2 [55] with 10 candidates and 20 trials.
5.2 Results
We present results in Table 4. We find surprisingly high accuracy
from the BM25 retriever, boasting a Recall@5 of 87.5 without any
additional query rewrite steps. This is likely due to high lexical over-
lap between some of the queries and the email contents. Because
our pipeline for question generation was optimized to be specific
enough to pick one email out of a batch of ten, the queries had to
name particular entities within the emails. This is reflected in the
high BM25 accuracy. We find that unsurprisingly larger models get
better at our benchmark with performance scaling from 8b to 70b
to GPT4o. We also find that query rewriting was not particularlyhelpful for this benchmark, especially for BM25. The highest per-
forming setting was GPTo, both with and without query rewrites
using BM25, which achieved an accuracy of 81.2% on EnronQA .
6 Case Study: Memorized Knowledge
With a growing body of literature on continued pre-training [ 76],
we note that an interesting use case of our benchmark is a large-
scale and realistic test bed for continued pre-training memorization.
Since our benchmark contains private knowledge that LLMs have
not been heavily pretrained on, alongside over 500k question and
answer pairs, there is plenty of data to benchmark and even fine-
tune models on to test parametric knowledge memorization, and
to benchmark this against RAG.
To this end, we provide initial results in this direction, hoping
that this resource will be useful to future researchers exploring
continued pretraining and memorization with LLMs.
6.1 Experimental Setting
We want to explore the memorization/retreival of between 10 and
20,000 facts about documents by three mechanisms: Long Context ,
RAG , and Memorization . For this setting, we simplify the problem
by looking at question-and-answer pairs directly rather than the
documents, though we hope future work can also explore training
on the documents. We use the rephrased question and answer pair

Ryan et al.
# Facts↓Long Context RAG Memorization (LoRA)
Rank→ - - 8 16 32 64 128 256 512 1024 2048
10 0.80 1.00 0.80 0.80 0.80 0.80 0.80 0.80 0.90 0.80 0.80
100 0.91 0.95 0.76 0.76 0.80 0.84 0.83 0.85 0.87 0.88 0.78
500 0.83 0.91 0.75 0.73 0.79 0.81 0.78 0.82 0.79 0.80 0.80
1000 0.79 0.89 0.72 0.71 0.79 0.79 0.79 0.78 0.76 0.78 0.73
5000 NA 0.92 0.53 0.55 0.69 0.75 0.74 0.69 0.78 0.03 —
10000 NA 0.92 0.53 0.61 0.69 0.74 0.77 0.69 0.78 0.79 —
20000 NA 0.93 0.53 0.62 0.69 0.74 0.75 0.08 0.00 0.03 —
Table 5: Factual memorization on subset of EnronQA benchmark. While currently, RAG is the best-performing method of
recalling factual information, training LoRA adapters for memorization can match the performance of putting all the facts in
context, suggesting this is a promising direction for future development.
as the context set, and tested on the true question and answer pair.
For Long Context, we put all the QA-pairs (facts) we were trying to
memorize in the context alongside their answers in the context of
Llama-3.1-8B-Instruct. We could test as high as 1,000 QA-pairs until
the context length was full. For RAG, we build an index over all
the QA-pairs and retrieve the top 100 (selected because it was the
best for Long Context) most relevant question-answer pairs to the
context of Llama-3.1-8B-Instruct. We use ColBERTv2 [ 64] as our
retreiver. Finally, for Memorization, we train a LoRA adapter using
the setup from the "Task of Fictitious Unlearning" paper, which
tests unlearning on LoRA adapters [ 48]. We train LoRA adapters of
rank{8,16 ,32 ,64 ,128 ,256 ,512 ,1024 ,2048}on all the facts for 10
epochs with rate 1×10−4. We set alpha to four times the rank and
used a dropout of 0.05. We test ablations with which layer to adapt,
and find that doing all linear layers works the best. All settings are
evaluated with LLama-3.1-70B-Instruct as a judge.
6.2 Results
We present the results of this experiment in Table 5. Interestingly,
LoRA memorization can match long-context performance at al-
most all scales and continue beyond the 1000 QA-pair cap that
blocks long-context from scaling. In fact, for many of the LoRA
adapters, the performance only starts to degrade around 20,000
facts memorized, showing a surprising capacity packed into just
the LoRA parameters. At all scales, RAG outperforms memorization
and long-context. This is likely due to the simplicity of the task
(retrieving a rephrased QA pair) as well as the strength of current
RAG systems. Memorization is a relatively understudied phenome-
non (mostly explored with LLMs to try to prevent memorization),
so, unsurprisingly, this does not yet outperform RAG. In the future,
with the continued development of pretraining and memorization
methods, it is possible that memorization through LoRA adapters
could match or exceed RAG performance.7 Discussion
Here, we discuss some lessons learned and valuable insights for
researchers working on similar problems.
LLM self-verifying and optimizing pipelines can be pow-
erful synthetic data tools. OurEnronQA benchmark is comprised
of entirely synthetically generated question-and-answer pairs. Past
processes of generating such QA resources would require a massive
human undertaking or need to be crowdsourced from platforms
where people naturally ask questions, such as Google [ 41] or Bing
[6]. Instead with the growing capabilities of LLMs we were instead
able to specify the requirements of our questions and answers into
verifiable unit tests. The questions needed to be "Specific," "Ob-
jective," "Grounded," and "High Quality." By writing each of these
checks as unit tests and optimizing our system end-to-end to pass
these unit tests, we were able to synthetically generate a large scale
dataset while maintaining quality. Questions only made our final
benchmark if they passed through all four of the unit tests success-
fully. We recognize this as an extensible pattern: (1) write specifi-
cations into unit tests, (2) optimize pipeline (fine-tuning, prompts,
etc.), (3) filter synthetic generations based on unit tests. This will
be a way to scale up data collection efforts in the future, which will
be heavily reliant on the design of the unit tests themselves.
Memorization through fine-tuning or continued pretrain-
ing are interesting future directions for retrieval. The current
SOTA for RAG is to retrieve and then pass this context to an LLM.
We showed, however, that LLMs are capable of memorizing large
amounts of data. For example, past RAG benchmarks like Natu-
ralQuestions [ 41] and TriviaQA [ 32] have all been consumed by the
parametric knowledge of LLMs. Right now this parametric knowl-
edge is largely dictated by the composition of the internet which is
the largest source of training data for these models. In the future one
could imagine doing continued pretraining on private documents
or an additional fine-tuning step for memorization. In section 6,
we show some first steps towards this effort and find that LoRA
adapters can match long-context at recalling factual knowledge in
a simplified setting. With more work on continued pretraining, we

EnronQA: Towards Personalized RAG over Private Documents
hope that EnronQA can serve as a resource for testing these sorts
of methods and exploring the limits of LLM memorization in the
future.
8 Conclusion
We introduce EnronQA , a dataset of 103,638 emails with 528,304
question-answer pairs across 150 different user inboxes. EnronQA
enables better benchmarking of RAG pipelines over private data
and allows for experimentation on the introduction of personalized
retrieval settings over realistic data. We showed that the EnronQA
benchmark is better than other single-hop retrieval benchmarks
for measuring the joint accuracy of retrievers and LLMs. We bench-
mark existing RAG pipelines over a sweep of retrievers, LLMs, and
architectures on EnronQA . Finally, we use EnronQA to explore the
tradeoff in memorization and retrieval when reasoning over private
documents. We release this large resource publicly to the commu-
nity for testing private and personalized retrieval and to enable
further research in continued pretraining, which is a potential new
frontier for information retrieval from the parametric weights of
Large Language Models.
Ethics Statement
TheEnronQA benchmark draws from the Enron emails corpus [ 39],
which was a release of corporate emails as a part of the Western
Energy Markets investigation in 2003. Not all Enron employees
whose emails were released were guilty of any crimes, and even
still, we wish to respect the wishes of all the humans behind the
Enron emails regardless of involvement in the criminal activity.
We take two critical steps to support these goals in respecting
the Enron employees behind the dataset. First, we use the 2015
release of the dataset where several people were removed from
the dataset upon request [ 17]. Second, we apply a filter to remove
any NSFW or toxic content from the dataset (§3.1), which can be
particularly personal.
Beyond this, we are more than happy to support any requests for
data removal from any affected parties. The EnronQA dataset will
be continuously maintained and updated should any such removal
requests arise. The Enron emails dataset has been used for about
twenty years in academic research, and we hope to support the
continued ethical use of this resource.
References
[1] CJ Adams, Jeffrey Sorensen, Julia Elliott, Lucas Dixon, Mark McDonald, nithum,
and Will Cukierski. 2017. Toxic Comment Classification Challenge. https:
//kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge
[2]Vaibhav Adlakha, Shehzaad Dhuliawala, Kaheer Suleman, Harm de
Vries, and Siva Reddy. 2022. TopiOCQA: Open-domain Conversa-
tional Question Answering with Topic Switching. Transactions of
the Association for Computational Linguistics 10 (04 2022), 468–483.
https://doi.org/10.1162/tacl_a_00471 arXiv:https://direct.mit.edu/tacl/article-
pdf/doi/10.1162/tacl_a_00471/2008126/tacl_a_00471.pdf
[3] Raviteja Anantha, Svitlana Vakulenko, Zhucheng Tu, Shayne Longpre, Stephen
Pulman, and Srinivas Chappidi. 2021. Open-Domain Question Answering Goes
Conversational via Question Rewriting. Proceedings of the 2021 Conference of the
North American Chapter of the Association for Computational Linguistics: Human
Language Technologies (2021).
[4] Simran Arora, Patrick Lewis, Angela Fan, Jacob Kahn, and Christopher Ré. 2023.
Reasoning over Public and Private Data in Retrieval-Based Systems. Transactions
of the Association for Computational Linguistics (2023). https://aclanthology.org/
2023.tacl-1.51/[5] Orlando Ayala and Patrice Bechard. 2024. Reducing hallucination in structured
outputs via Retrieval-Augmented Generation. In Proceedings of the 2024 Con-
ference of the North American Chapter of the Association for Computational Lin-
guistics: Human Language Technologies (Volume 6: Industry Track) , Yi Yang, Aida
Davani, Avi Sil, and Anoop Kumar (Eds.). Association for Computational Lin-
guistics, Mexico City, Mexico, 228–238. https://doi.org/10.18653/v1/2024.naacl-
industry.19
[6]Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng, Jianfeng Gao, Xiaodong
Liu, Rangan Majumder, Andrew McNamara, Bhaskar Mitra, Tri Nguyen, Mir
Rosenberg, Xia Song, Alina Stoica, Saurabh Tiwary, and Tong Wang. 2018.
MS MARCO: A Human Generated MAchine Reading COmprehension Dataset.
arXiv:1611.09268 [cs.CL] https://arxiv.org/abs/1611.09268
[7] Jon Ander Campos, Arantxa Otegi, Aitor Soroa, Jan Deriu, Mark Cieliebak, and
Eneko Agirre. 2020. DoQA – Accessing Domain-Specific FAQs via Conversational
QA. arXiv:2005.01328 [cs.CL] https://arxiv.org/abs/2005.01328
[8]Vittorio Castelli, Rishav Chakravarti, Saswati Dana, Anthony Ferritto, Radu
Florian, Martin Franz, Dinesh Garg, Dinesh Khandelwal, Scott McCarley, Mike
McCawley, Mohamed Nasr, Lin Pan, Cezar Pendus, John Pitrelli, Saurabh Pujar,
Salim Roukos, Andrzej Sakrajda, Avirup Sil, Rosario Uceda-Sosa, Todd Ward,
and Rong Zhang. 2019. The TechQA Dataset. arXiv:1911.02984 [cs.CL] https:
//arxiv.org/abs/1911.02984
[9] Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena Shah, Iana Borova, Dylan
Langdon, Reema Moussa, Matt Beane, Ting-Hao Huang, Bryan Routledge, and
William Yang Wang. 2021. FinQA: A Dataset of Numerical Reasoning over
Financial Data. In Proceedings of the 2021 Conference on Empirical Methods in
Natural Language Processing , Marie-Francine Moens, Xuanjing Huang, Lucia
Specia, and Scott Wen-tau Yih (Eds.). Association for Computational Linguistics,
Online and Punta Cana, Dominican Republic, 3697–3711. https://doi.org/10.
18653/v1/2021.emnlp-main.300
[10] Zhiyu Chen, Shiyang Li, Charese Smiley, Zhiqiang Ma, Sameena Shah, and
William Yang Wang. 2022. ConvFinQA: Exploring the Chain of Numerical
Reasoning in Conversational Finance Question Answering. In Proceedings of
the 2022 Conference on Empirical Methods in Natural Language Processing , Yoav
Goldberg, Zornitsa Kozareva, and Yue Zhang (Eds.). Association for Compu-
tational Linguistics, Abu Dhabi, United Arab Emirates, 6279–6292. https:
//doi.org/10.18653/v1/2022.emnlp-main.421
[11] Eunsol Choi, He He, Mohit Iyyer, Mark Yatskar, Wen-tau Yih, Yejin Choi, Percy
Liang, and Luke Zettlemoyer. 2018. QuAC: Question Answering in Context. In
Proceedings of the 2018 Conference on Empirical Methods in Natural Language
Processing , Ellen Riloff, David Chiang, Julia Hockenmaier, and Jun’ichi Tsujii
(Eds.). Association for Computational Linguistics, Brussels, Belgium, 2174–2184.
https://doi.org/10.18653/v1/D18-1241
[12] Mark Collier and Joeran Beel. 2019. Memory-Augmented Neural Networks
for Machine Translation. In Proceedings of Machine Translation Summit XVII:
Research Track , Mikel Forcada, Andy Way, Barry Haddow, and Rico Sennrich
(Eds.). European Association for Machine Translation, Dublin, Ireland, 172–181.
https://aclanthology.org/W19-6617/
[13] Together Computer. 2023. RedPajama: an Open Dataset for Training Large Lan-
guage Models . https://github.com/togethercomputer/RedPajama-Data
[14] Vinay Deolalikar. 2014. Distance or Coverage? Retrieving Knowledge-Rich
Documents From Enterprise Text Collections. In Proceedings of the 23rd ACM In-
ternational Conference on Conference on Information and Knowledge Management
(Shanghai, China) (CIKM ’14) . Association for Computing Machinery, New York,
NY, USA, 1771–1774. https://doi.org/10.1145/2661829.2661865
[15] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ah-
mad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela
Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sra-
vankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien
Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Roziere, Bethany Biron, Binh
Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra,
Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne
Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song,
Danielle Pintz, Danny Livshits, David Esiobu, Dhruv Choudhary, Dhruv Maha-
jan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab
AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic,
Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Graeme
Nail, Gregoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah
Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel
Kloumann, Ishan Misra, Ivan Evtimov, Jade Copet, Jaewon Lee, Jan Geffert, Jana
Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer
Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen
Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca,
Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Kartikeya
Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, Khalid El-Arini,
Krithika Iyer, Kshitiz Malik, Kuenley Chiu, Kunal Bhalla, Lauren Rantala-Yeary,
Laurens van der Maaten, Lawrence Chen, Liang Tan, Liz Jenkins, Louis Martin,
Lovish Madaan, Lubo Malo, Lukas Blecher, Lukas Landzaat, Luke de Oliveira,
Madeline Muzzi, Mahesh Pasupuleti, Mannat Singh, Manohar Paluri, Marcin

Ryan et al.
Kardas, Mathew Oldham, Mathieu Rita, Maya Pavlova, Melanie Kambadur, Mike
Lewis, Min Si, Mitesh Kumar Singh, Mona Hassan, Naman Goyal, Narjes Torabi,
Nikolay Bashlykov, Nikolay Bogoychev, Niladri Chatterji, Olivier Duchenne,
Onur Çelebi, Patrick Alrassy, Pengchuan Zhang, Pengwei Li, Petar Vasic, Peter
Weng, Prajjwal Bhargava, Pratik Dubal, Praveen Krishnan, Punit Singh Koura,
Puxin Xu, Qing He, Qingxiao Dong, Ragavan Srinivasan, Raj Ganapathy, Ra-
mon Calderer, Ricardo Silveira Cabral, Robert Stojnic, Roberta Raileanu, Rohit
Girdhar, Rohit Patel, Romain Sauvestre, Ronnie Polidoro, Roshan Sumbaly, Ross
Taylor, Ruan Silva, Rui Hou, Rui Wang, Saghar Hosseini, Sahana Chennabasappa,
Sanjay Singh, Sean Bell, Seohyun Sonia Kim, Sergey Edunov, Shaoliang Nie,
Sharan Narang, Sharath Raparthy, Sheng Shen, Shengye Wan, Shruti Bhosale,
Shun Zhang, Simon Vandenhende, Soumya Batra, Spencer Whitman, Sten Sootla,
Stephane Collot, Suchin Gururangan, Sydney Borodinsky, Tamar Herman, Tara
Fowler, Tarek Sheasha, Thomas Georgiou, Thomas Scialom, Tobias Speckbacher,
Todor Mihaylov, Tong Xiao, Ujjwal Karn, Vedanuj Goswami, Vibhor Gupta,
Vignesh Ramanathan, Viktor Kerkez, Vincent Gonguet, Virginie Do, Vish Vo-
geti, Vladan Petrovic, Weiwei Chu, Wenhan Xiong, Wenyin Fu, Whitney Meers,
Xavier Martinet, Xiaodong Wang, Xiaoqing Ellen Tan, Xinfeng Xie, Xuchao
Jia, Xuewei Wang, Yaelle Goldschlag, Yashesh Gaur, Yasmine Babaei, Yi Wen,
Yiwen Song, Yuchen Zhang, Yue Li, Yuning Mao, Zacharie Delpierre Coudert,
Zheng Yan, Zhengxing Chen, Zoe Papakipos, Aaditya Singh, Aaron Grattafiori,
Abha Jain, Adam Kelsey, Adam Shajnfeld, Adithya Gangidi, Adolfo Victoria,
Ahuva Goldstand, Ajay Menon, Ajay Sharma, Alex Boesenberg, Alex Vaughan,
Alexei Baevski, Allie Feinstein, Amanda Kallet, Amit Sangani, Anam Yunus, An-
drei Lupu, Andres Alvarado, Andrew Caples, Andrew Gu, Andrew Ho, Andrew
Poulton, Andrew Ryan, Ankit Ramchandani, Annie Franco, Aparajita Saraf, Arka-
bandhu Chowdhury, Ashley Gabriel, Ashwin Bharambe, Assaf Eisenman, Azadeh
Yazdan, Beau James, Ben Maurer, Benjamin Leonhardi, Bernie Huang, Beth Loyd,
Beto De Paola, Bhargavi Paranjape, Bing Liu, Bo Wu, Boyu Ni, Braden Hancock,
Bram Wasti, Brandon Spence, Brani Stojkovic, Brian Gamido, Britt Montalvo,
Carl Parker, Carly Burton, Catalina Mejia, Changhan Wang, Changkyu Kim,
Chao Zhou, Chester Hu, Ching-Hsiang Chu, Chris Cai, Chris Tindal, Christoph
Feichtenhofer, Damon Civin, Dana Beaty, Daniel Kreymer, Daniel Li, Danny
Wyatt, David Adkins, David Xu, Davide Testuggine, Delia David, Devi Parikh,
Diana Liskovich, Didem Foss, Dingkang Wang, Duc Le, Dustin Holland, Edward
Dowling, Eissa Jamil, Elaine Montgomery, Eleonora Presani, Emily Hahn, Emily
Wood, Erik Brinkman, Esteban Arcaute, Evan Dunbar, Evan Smothers, Fei Sun,
Felix Kreuk, Feng Tian, Firat Ozgenel, Francesco Caggioni, Francisco Guzmán,
Frank Kanayet, Frank Seide, Gabriela Medina Florez, Gabriella Schwarz, Gada
Badeer, Georgia Swee, Gil Halpern, Govind Thattai, Grant Herman, Grigory
Sizov, Guangyi, Zhang, Guna Lakshminarayanan, Hamid Shojanazeri, Han Zou,
Hannah Wang, Hanwen Zha, Haroun Habeeb, Harrison Rudolph, Helen Suk,
Henry Aspegren, Hunter Goldman, Ibrahim Damlaj, Igor Molybog, Igor Tufanov,
Irina-Elena Veliche, Itai Gat, Jake Weissman, James Geboski, James Kohli, Japhet
Asher, Jean-Baptiste Gaya, Jeff Marcus, Jeff Tang, Jennifer Chan, Jenny Zhen,
Jeremy Reizenstein, Jeremy Teboul, Jessica Zhong, Jian Jin, Jingyi Yang, Joe
Cummings, Jon Carvill, Jon Shepard, Jonathan McPhie, Jonathan Torres, Josh
Ginsburg, Junjie Wang, Kai Wu, Kam Hou U, Karan Saxena, Karthik Prasad,
Kartikay Khandelwal, Katayoun Zand, Kathy Matosich, Kaushik Veeraraghavan,
Kelly Michelena, Keqian Li, Kun Huang, Kunal Chawla, Kushal Lakhotia, Kyle
Huang, Lailin Chen, Lakshya Garg, Lavender A, Leandro Silva, Lee Bell, Lei
Zhang, Liangpeng Guo, Licheng Yu, Liron Moshkovich, Luca Wehrstedt, Madian
Khabsa, Manav Avalani, Manish Bhatt, Maria Tsimpoukelli, Martynas Mankus,
Matan Hasson, Matthew Lennie, Matthias Reso, Maxim Groshev, Maxim Nau-
mov, Maya Lathi, Meghan Keneally, Michael L. Seltzer, Michal Valko, Michelle
Restrepo, Mihir Patel, Mik Vyatskov, Mikayel Samvelyan, Mike Clark, Mike
Macey, Mike Wang, Miquel Jubert Hermoso, Mo Metanat, Mohammad Rastegari,
Munish Bansal, Nandhini Santhanam, Natascha Parks, Natasha White, Navyata
Bawa, Nayan Singhal, Nick Egebo, Nicolas Usunier, Nikolay Pavlovich Laptev,
Ning Dong, Ning Zhang, Norman Cheng, Oleg Chernoguz, Olivia Hart, Omkar
Salpekar, Ozlem Kalinli, Parkin Kent, Parth Parekh, Paul Saab, Pavan Balaji, Pedro
Rittner, Philip Bontrager, Pierre Roux, Piotr Dollar, Polina Zvyagina, Prashant
Ratanchandani, Pritish Yuvraj, Qian Liang, Rachad Alao, Rachel Rodriguez, Rafi
Ayub, Raghotham Murthy, Raghu Nayani, Rahul Mitra, Raymond Li, Rebekkah
Hogan, Robin Battey, Rocky Wang, Rohan Maheswari, Russ Howes, Ruty Rinott,
Sai Jayesh Bondu, Samyak Datta, Sara Chugh, Sara Hunt, Sargun Dhillon, Sasha
Sidorov, Satadru Pan, Saurabh Verma, Seiji Yamamoto, Sharadh Ramaswamy,
Shaun Lindsay, Shaun Lindsay, Sheng Feng, Shenghao Lin, Shengxin Cindy
Zha, Shiva Shankar, Shuqiang Zhang, Shuqiang Zhang, Sinong Wang, Sneha
Agarwal, Soji Sajuyigbe, Soumith Chintala, Stephanie Max, Stephen Chen, Steve
Kehoe, Steve Satterfield, Sudarshan Govindaprasad, Sumit Gupta, Sungmin Cho,
Sunny Virk, Suraj Subramanian, Sy Choudhury, Sydney Goldman, Tal Remez,
Tamar Glaser, Tamara Best, Thilo Kohler, Thomas Robinson, Tianhe Li, Tianjun
Zhang, Tim Matthews, Timothy Chou, Tzook Shaked, Varun Vontimitta, Victoria
Ajayi, Victoria Montanez, Vijai Mohan, Vinay Satish Kumar, Vishal Mangla,
Vítor Albiero, Vlad Ionescu, Vlad Poenaru, Vlad Tiberiu Mihailescu, Vladimir
Ivanov, Wei Li, Wenchen Wang, Wenwen Jiang, Wes Bouaziz, Will Constable,
Xiaocheng Tang, Xiaofang Wang, Xiaojian Wu, Xiaolan Wang, Xide Xia, XilunWu, Xinbo Gao, Yanjun Chen, Ye Hu, Ye Jia, Ye Qi, Yenda Li, Yilin Zhang, Ying
Zhang, Yossi Adi, Youngjin Nam, Yu, Wang, Yuchen Hao, Yundi Qian, Yuzi He,
Zach Rait, Zachary DeVito, Zef Rosnbrick, Zhaoduo Wen, Zhenyu Yang, and
Zhiwei Zhao. 2024. The Llama 3 Herd of Models. arXiv:2407.21783 [cs.AI]
https://arxiv.org/abs/2407.21783
[16] Matthew Dunn, Levent Sagun, Mike Higgins, V. Ugur Guney, Volkan Cirik, and
Kyunghyun Cho. 2017. SearchQA: A New Q&A Dataset Augmented with Context
from a Search Engine. arXiv:1704.05179 [cs.CL] https://arxiv.org/abs/1704.05179
[17] Enron Corp and William W. Cohen. 2015. Enron Email Dataset. https:
//www.loc.gov/item/2018487913/ United States Federal Energy Regulatory Com-
mission, William W. Cohen, MLD, CMU, Philadelphia, PA. [Software, E-Resource].
Retrieved from the Library of Congress.
[18] Song Feng, Hui Wan, Chulaka Gunasekara, Siva Patel, Sachindra Joshi, and
Luis Lastras. 2020. doc2dial: A Goal-Oriented Document-Grounded Dialogue
Dataset. In Proceedings of the 2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP) , Bonnie Webber, Trevor Cohn, Yulan He, and
Yang Liu (Eds.). Association for Computational Linguistics, Online, 8118–8128.
https://doi.org/10.18653/v1/2020.emnlp-main.652
[19] Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles
Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, Shawn Presser, and
Connor Leahy. 2020. The Pile: An 800GB Dataset of Diverse Text for Language
Modeling. arXiv preprint arXiv:2101.00027 (2020).
[20] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi
Dai, Jiawei Sun, Meng Wang, and Haofen Wang. 2024. Retrieval-Augmented
Generation for Large Language Models: A Survey. arXiv:2312.10997 [cs.CL]
https://arxiv.org/abs/2312.10997
[21] Samira Ghodratnama and Mehrdad Zakershahrak. 2024. Adapting LLMs for Effi-
cient, Personalized Information Retrieval: Methods and Implications. In Service-
Oriented Computing – ICSOC 2023 Workshops , Flavia Monti, Pierluigi Plebani,
Naouel Moha, Hye-young Paik, Johanna Barzen, Gowri Ramachandran, Devis
Bianchini, Damian A. Tamburri, and Massimo Mecella (Eds.). Springer Nature
Singapore, Singapore, 17–26.
[22] Grand View Research. 2024. Retrieval Augmented Generation Market Size
Report, 2030. https://www.grandviewresearch.com/industry-analysis/retrieval-
augmented-generation-rag-market-report
[23] Alex Graves, Greg Wayne, and Ivo Danihelka. 2014. Neural Turing Machines.
arXiv:1410.5401 [cs.NE] https://arxiv.org/abs/1410.5401
[24] Kshitij Gupta, Benjamin Thérien, Adam Ibrahim, Mats L. Richter, Quentin
Anthony, Eugene Belilovsky, Irina Rish, and Timothée Lesort. 2023. Contin-
ual Pre-Training of Large Language Models: How to (re)warm your model?
arXiv:2308.04014 [cs.CL] https://arxiv.org/abs/2308.04014
[25] Dan Hendrycks, Collin Burns, Anya Chen, and Spencer Ball. 2021.
CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review.
arXiv:2103.06268 [cs.CL] https://arxiv.org/abs/2103.06268
[26] Jing Huang, Diyi Yang, and Christopher Potts. 2024. Demystifying Verbatim
Memorization in Large Language Models. arXiv:2407.17817 [cs.CL] https:
//arxiv.org/abs/2407.17817
[27] Yulong Hui, Yao Lu, and Huanchen Zhang. 2024. UDA: A Benchmark
Suite for Retrieval Augmented Generation in Real-world Document Analysis.
arXiv:2406.15187 [cs.AI] https://arxiv.org/abs/2406.15187
[28] Infiniflow. 2024. RAGFlow: An open-source RAG (Retrieval-Augmented Gen-
eration) engine based on deep document understanding. https://github.com/
infiniflow/ragflow Accessed: 2024-09-18.
[29] Mohit Iyyer, Wen-tau Yih, and Ming-Wei Chang. 2017. Search-based Neural
Structured Learning for Sequential Question Answering. In Proceedings of the
55th Annual Meeting of the Association for Computational Linguistics (Volume 1:
Long Papers) . Association for Computational Linguistics, Vancouver, Canada,
1821–1831. https://doi.org/10.18653/v1/P17-1167
[30] Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche
Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou
Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample,
Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep
Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Théophile Gervet,
Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2024.
Mixtral of Experts. arXiv:2401.04088 [cs.LG] https://arxiv.org/abs/2401.04088
[31] Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William Cohen, and Xinghua Lu.
2019. PubMedQA: A Dataset for Biomedical Research Question Answering. In
Proceedings of the 2019 Conference on Empirical Methods in Natural Language
Processing and the 9th International Joint Conference on Natural Language Pro-
cessing (EMNLP-IJCNLP) , Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan
(Eds.). Association for Computational Linguistics, Hong Kong, China, 2567–2577.
https://doi.org/10.18653/v1/D19-1259
[32] Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. 2017. triviaqa:
A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehen-
sion. arXiv e-prints , Article arXiv:1705.03551 (2017), arXiv:1705.03551 pages.
arXiv:1705.03551

EnronQA: Towards Personalized RAG over Private Documents
[33] Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. 2017. TriviaQA:
A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehen-
sion. In Proceedings of the 55th Annual Meeting of the Association for Compu-
tational Linguistics (Volume 1: Long Papers) , Regina Barzilay and Min-Yen Kan
(Eds.). Association for Computational Linguistics, Vancouver, Canada, 1601–1611.
https://doi.org/10.18653/v1/P17-1147
[34] Armand Joulin, Edouard Grave, Piotr Bojanowski, and Tomas Mikolov. 2016. Bag
of Tricks for Efficient Text Classification. arXiv preprint arXiv:1607.01759 (2016).
[35] Ehsan Kamalloo, Aref Jafari, Xinyu Zhang, Nandan Thakur, and Jimmy Lin. 2023.
HAGRID: A Human-LLM Collaborative Dataset for Generative Information-
Seeking with Attribution. arXiv:2307.16883 (2023).
[36] Nikhil Kandpal, Haikang Deng, Adam Roberts, Eric Wallace, and Colin Raf-
fel. 2023. Large Language Models Struggle to Learn Long-Tail Knowledge.
arXiv:2211.08411 [cs.CL] https://arxiv.org/abs/2211.08411
[37] Zixuan Ke, Yijia Shao, Haowei Lin, Tatsuya Konishi, Gyuhak Kim, and Bing Liu.
2023. Continual Pre-training of Language Models. In The Eleventh International
Conference on Learning Representations . https://openreview.net/forum?id=m_
GDIItaI3o
[38] Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, Zhiyuan Zhang, Keshav San-
thanam, Sri Vardhamanan, Saiful Haq, Ashutosh Sharma, Thomas T. Joshi, Hanna
Moazam, Heather Miller, Matei Zaharia, and Christopher Potts. 2023. DSPy: Com-
piling Declarative Language Model Calls into Self-Improving Pipelines. arXiv
preprint arXiv:2310.03714 (2023).
[39] Bryan Klimt and Yiming Yang. 2004. The Enron Corpus: A New Dataset for Email
Classification Research. In European Conference on Machine Learning . Springer
Berlin Heidelberg, Berlin, Heidelberg, 217–226.
[40] Tomáš Kočiský, Jonathan Schwarz, Phil Blunsom, Chris Dyer, Karl Moritz Her-
mann, Gábor Melis, and Edward Grefenstette. 2018. The NarrativeQA Reading
Comprehension Challenge. Transactions of the Association for Computational
Linguistics 6 (2018), 317–328. https://doi.org/10.1162/tacl_a_00023
[41] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur
Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Matthew Kelcey, Jacob
Devlin, Kenton Lee, Kristina N. Toutanova, Llion Jones, Ming-Wei Chang, Andrew
Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natural Questions: a
Benchmark for Question Answering Research. Transactions of the Association of
Computational Linguistics (2019).
[42] Jure Leskovec, Anand Rajaraman, and Jeffrey David Ullman. 2020. Mining of
Massive Data Sets . Cambridge University Press.
[43] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen tau Yih, Tim
Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2021. Retrieval-Augmented
Generation for Knowledge-Intensive NLP Tasks. arXiv:2005.11401 [cs.CL]
https://arxiv.org/abs/2005.11401
[44] Jimmy Lin, Xueguang Ma, Sheng-Chieh Lin, Jheng-Hong Yang, Ronak Pradeep,
and Rodrigo Nogueira. 2021. Pyserini: A Python Toolkit for Reproducible Infor-
mation Retrieval Research with Sparse and Dense Representations. In Proceedings
of the 44th Annual International ACM SIGIR Conference on Research and Develop-
ment in Information Retrieval (SIGIR 2021) . 2356–2362.
[45] Sijia Liu, Yuanshun Yao, Jinghan Jia, Stephen Casper, Nathalie Baracaldo, Peter
Hase, Yuguang Yao, Chris Yuhao Liu, Xiaojun Xu, Hang Li, Kush R. Varshney,
Mohit Bansal, Sanmi Koyejo, and Yang Liu. 2024. Rethinking Machine Unlearning
for Large Language Models. arXiv:2402.08787 [cs.LG] https://arxiv.org/abs/2402.
08787
[46] Yougang Lyu, Lingyong Yan, Shuaiqiang Wang, Haibo Shi, Dawei Yin, Pengjie
Ren, Zhumin Chen, Maarten de Rijke, and Zhaochun Ren. 2024. KnowTun-
ing: Knowledge-aware Fine-tuning for Large Language Models. In Proceed-
ings of the 2024 Conference on Empirical Methods in Natural Language Pro-
cessing , Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (Eds.). Associ-
ation for Computational Linguistics, Miami, Florida, USA, 14535–14556. https:
//doi.org/10.18653/v1/2024.emnlp-main.805
[47] Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao, and Nan Duan. 2023. Query
Rewriting in Retrieval-Augmented Large Language Models. In Proceedings of the
2023 Conference on Empirical Methods in Natural Language Processing , Houda
Bouamor, Juan Pino, and Kalika Bali (Eds.). Association for Computational
Linguistics, Singapore, 5303–5315. https://doi.org/10.18653/v1/2023.emnlp-
main.322
[48] Pratyush Maini, Zhili Feng, Avi Schwarzschild, Zachary Chase Lipton, and J Zico
Kolter. 2024. TOFU: A Task of Fictitious Unlearning for LLMs. In First Conference
on Language Modeling . https://openreview.net/forum?id=B41hNBoWLo
[49] Chaitanya Malaviya, Subin Lee, Sihao Chen, Elizabeth Sieber, Mark Yatskar, and
Dan Roth. 2024. ExpertQA: Expert-Curated Questions and Attributed Answers.
arXiv:2309.07852 [cs.CL] https://arxiv.org/abs/2309.07852
[50] Luke Merrick. 2024. Embedding And Clustering Your Data Can Improve Con-
trastive Pretraining. arXiv:2407.18887 [cs.LG] https://arxiv.org/abs/2407.18887
[51] Timo Möller, Anthony Reina, Raghavan Jayakumar, and Malte Pietsch. 2020.
COVID-QA: A Question Answering Dataset for COVID-19. In Proceedings of the
1st Workshop on NLP for COVID-19 at ACL 2020 , Karin Verspoor, Kevin Bretonnel
Cohen, Mark Dredze, Emilio Ferrara, Jonathan May, Robert Munro, Cecile Paris,and Byron Wallace (Eds.). Association for Computational Linguistics, Online.
https://aclanthology.org/2020.nlpcovid19-acl.18
[52] Chenghao Mou, Chris Ha, Kenneth Enevoldsen, and Peiyuan Liu. 2023.
ChenghaoMou/text-dedup: Reference Snapshot . https://doi.org/10.5281/zenodo.
8364980
[53] Kai Nakamura, Sharon Levy, Yi-Lin Tuan, Wenhu Chen, and William Yang Wang.
2022. HybriDialogue: An Information-Seeking Dialogue Dataset Grounded
on Tabular and Textual Data. In Findings of the Association for Computational
Linguistics: ACL 2022 , Smaranda Muresan, Preslav Nakov, and Aline Villavicencio
(Eds.). Association for Computational Linguistics, Dublin, Ireland, 481–492. https:
//doi.org/10.18653/v1/2022.findings-acl.41
[54] Abhilash Nandy, Soumya Sharma, Shubham Maddhashiya, Kapil Sachdeva,
Pawan Goyal, and NIloy Ganguly. 2021. Question Answering over Electronic
Devices: A New Benchmark Dataset and a Multi-Task Learning based QA Frame-
work. In Findings of the Association for Computational Linguistics: EMNLP 2021 ,
Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih
(Eds.). Association for Computational Linguistics, Punta Cana, Dominican Re-
public, 4600–4609. https://doi.org/10.18653/v1/2021.findings-emnlp.392
[55] Krista Opsahl-Ong, Michael J Ryan, Josh Purtell, David Broman, Christo-
pher Potts, Matei Zaharia, and Omar Khattab. 2024. Optimizing In-
structions and Demonstrations for Multi-Stage Language Model Programs.
arXiv:2406.11695 [cs.CL] https://arxiv.org/abs/2406.11695
[56] Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick Lewis, Majid Yazdani,
Nicola De Cao, James Thorne, Yacine Jernite, Vladimir Karpukhin, Jean Mail-
lard, Vassilis Plachouras, Tim Rocktäschel, and Sebastian Riedel. 2021. KILT: a
Benchmark for Knowledge Intensive Language Tasks. In Proceedings of the 2021
Conference of the North American Chapter of the Association for Computational
Linguistics: Human Language Technologies , Kristina Toutanova, Anna Rumshisky,
Luke Zettlemoyer, Dilek Hakkani-Tur, Iz Beltagy, Steven Bethard, Ryan Cotterell,
Tanmoy Chakraborty, and Yichao Zhou (Eds.). Association for Computational
Linguistics, Online, 2523–2544. https://doi.org/10.18653/v1/2021.naacl-main.200
[57] Fabio Petroni, Tim Rocktäschel, Sebastian Riedel, Patrick Lewis, Anton Bakhtin,
Yuxiang Wu, and Alexander Miller. 2019. Language Models as Knowledge Bases?.
InProceedings of the 2019 Conference on Empirical Methods in Natural Language
Processing and the 9th International Joint Conference on Natural Language Pro-
cessing (EMNLP-IJCNLP) , Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan
(Eds.). Association for Computational Linguistics, Hong Kong, China, 2463–2473.
https://doi.org/10.18653/v1/D19-1250
[58] Fabio Pinelli, Gabriele Tolomei, and Giovanni Trappolini. 2023. FLIRT: Federated
Learning for Information Retrieval. In Proceedings of the 46th International ACM
SIGIR Conference on Research and Development in Information Retrieval (Taipei,
Taiwan) (SIGIR ’23) . Association for Computing Machinery, New York, NY, USA,
3472–3475. https://doi.org/10.1145/3539618.3591926
[59] Jack W. Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann,
Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young,
Eliza Rutherford, Tom Hennigan, Jacob Menick, Albin Cassirer, Richard Powell,
George van den Driessche, Lisa Anne Hendricks, Maribeth Rauh, Po-Sen Huang,
Amelia Glaese, Johannes Welbl, Sumanth Dathathri, Saffron Huang, Jonathan Ue-
sato, John Mellor, Irina Higgins, Antonia Creswell, Nat McAleese, Amy Wu, Erich
Elsen, Siddhant Jayakumar, Elena Buchatskaya, David Budden, Esme Sutherland,
Karen Simonyan, Michela Paganini, Laurent Sifre, Lena Martens, Xiang Lorraine
Li, Adhiguna Kuncoro, Aida Nematzadeh, Elena Gribovskaya, Domenic Donato,
Angeliki Lazaridou, Arthur Mensch, Jean-Baptiste Lespiau, Maria Tsimpoukelli,
Nikolai Grigorev, Doug Fritz, Thibault Sottiaux, Mantas Pajarskas, Toby Pohlen,
Zhitao Gong, Daniel Toyama, Cyprien de Masson d’Autume, Yujia Li, Tayfun
Terzi, Vladimir Mikulik, Igor Babuschkin, Aidan Clark, Diego de Las Casas, Au-
relia Guy, Chris Jones, James Bradbury, Matthew Johnson, Blake Hechtman,
Laura Weidinger, Iason Gabriel, William Isaac, Ed Lockhart, Simon Osindero,
Laura Rimell, Chris Dyer, Oriol Vinyals, Kareem Ayoub, Jeff Stanway, Lorrayne
Bennett, Demis Hassabis, Koray Kavukcuoglu, and Geoffrey Irving. 2022. Scal-
ing Language Models: Methods, Analysis & Insights from Training Gopher.
arXiv:2112.11446 [cs.CL] https://arxiv.org/abs/2112.11446
[60] Pranav Rajpurkar, Robin Jia, and Percy Liang. 2018. Know What You Don’t
Know: Unanswerable Questions for SQuAD. arXiv:1806.03822 [cs.CL] https:
//arxiv.org/abs/1806.03822
[61] Siva Reddy, Danqi Chen, and Christopher D. Manning. 2019. CoQA: A Con-
versational Question Answering Challenge. Transactions of the Association for
Computational Linguistics 7 (2019), 249–266. https://doi.org/10.1162/tacl_a_00266
[62] Adam Roberts, Colin Raffel, and Noam Shazeer. 2020. How Much Knowledge
Can You Pack Into the Parameters of a Language Model?. In Proceedings of the
2020 Conference on Empirical Methods in Natural Language Processing (EMNLP) ,
Bonnie Webber, Trevor Cohn, Yulan He, and Yang Liu (Eds.). Association for
Computational Linguistics, Online, 5418–5426. https://doi.org/10.18653/v1/2020.
emnlp-main.437
[63] Mobashir Sadat, Zhengyu Zhou, Lukas Lange, Jun Araki, Arsalan Gundroo,
Bingqing Wang, Rakesh Menon, Md Parvez, and Zhe Feng. 2023. DelucionQA:
Detecting Hallucinations in Domain-specific Question Answering. In Findings
of the Association for Computational Linguistics: EMNLP 2023 . Association for

Ryan et al.
Computational Linguistics, Singapore, 822–835. https://doi.org/10.18653/v1/
2023.findings-emnlp.59
[64] Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei
Zaharia. 2022. ColBERTv2: Effective and Efficient Retrieval via Lightweight Late
Interaction. In Proceedings of the 2022 Conference of the North American Chapter
of the Association for Computational Linguistics: Human Language Technologies ,
Marine Carpuat, Marie-Catherine de Marneffe, and Ivan Vladimir Meza Ruiz
(Eds.). Association for Computational Linguistics, Seattle, United States, 3715–
3734. https://doi.org/10.18653/v1/2022.naacl-main.272
[65] Milad Shokouhi and Luo Si. 2011. Federated Search. Foundations and Trends ®in
Information Retrieval 5, 1 (2011), 1–102. https://doi.org/10.1561/1500000010
[66] Kurt Shuster, Spencer Poff, Moya Chen, Douwe Kiela, and Jason Weston. 2021.
Retrieval Augmentation Reduces Hallucination in Conversation. arXiv preprint
arXiv:2104.07567 (2021). https://arxiv.org/abs/2104.07567
[67] Luca Soldaini, Rodney Kinney, Akshita Bhagia, Dustin Schwenk, David Atkin-
son, Russell Authur, Ben Bogin, Khyathi Chandu, Jennifer Dumas, Yanai Elazar,
Valentin Hofmann, Ananya Harsh Jha, Sachin Kumar, Li Lucy, Xinxi Lyu, Nathan
Lambert, Ian Magnusson, Jacob Morrison, Niklas Muennighoff, Aakanksha Naik,
Crystal Nam, Matthew E. Peters, Abhilasha Ravichander, Kyle Richardson, Ze-
jiang Shen, Emma Strubell, Nishant Subramani, Oyvind Tafjord, Pete Walsh,
Luke Zettlemoyer, Noah A. Smith, Hannaneh Hajishirzi, Iz Beltagy, Dirk Groen-
eveld, Jesse Dodge, and Kyle Lo. 2024. Dolma: An Open Corpus of Three Tril-
lion Tokens for Language Model Pretraining Research. arXiv preprint (2024).
https://arxiv.org/abs/2402.00159
[68] Viju Sudhi, Sinchana Ramakanth Bhat, Max Rudat, and Roman Teucher. 2024.
RAG-Ex: A Generic Framework for Explaining Retrieval Augmented Gener-
ation. In Proceedings of the 47th International ACM SIGIR Conference on Re-
search and Development in Information Retrieval (Washington DC, USA) (SIGIR
’24). Association for Computing Machinery, New York, NY, USA, 2776–2780.
https://doi.org/10.1145/3626772.3657660
[69] Katherine Tian, Eric Mitchell, Huaxiu Yao, Christopher Manning, and Chelsea
Finn. 2023. Fine-tuning Language Models for Factuality. In NeurIPS 2023 Workshop
on Instruction Tuning and Instruction Following . https://openreview.net/forum?
id=kEK08VdSO5
[70] Adam Trischler, Tong Wang, Xingdi Yuan, Justin Harris, Alessandro Sordoni,
Philip Bachman, and Kaheer Suleman. 2017. NewsQA: A Machine Comprehen-
sion Dataset. In Proceedings of the 2nd Workshop on Representation Learning for
NLP, Phil Blunsom, Antoine Bordes, Kyunghyun Cho, Shay Cohen, Chris Dyer,
Edward Grefenstette, Karl Moritz Hermann, Laura Rimell, Jason Weston, and
Scott Yih (Eds.). Association for Computational Linguistics, Vancouver, Canada,
191–200. https://doi.org/10.18653/v1/W17-2623
[71] Shuai Wang, Ekaterina Khramtsova, Shengyao Zhuang, and Guido Zuccon. 2024.
FeB4RAG: Evaluating Federated Search in the Context of Retrieval Augmented
Generation. In Proceedings of the 47th International ACM SIGIR Conference on
Research and Development in Information Retrieval (Washington DC, USA) (SIGIR
’24). Association for Computing Machinery, New York, NY, USA, 763–773. https:
//doi.org/10.1145/3626772.3657853
[72] Zijie J. Wang and Duen Horng Chau. 2024. MeMemo: On-device Retrieval
Augmentation for Private and Personalized Text Generation. In Proceedings
of the 47th International ACM SIGIR Conference on Research and Development
in Information Retrieval (Washington DC, USA) (SIGIR ’24) . Association for
Computing Machinery, New York, NY, USA, 2765–2770. https://doi.org/10.1145/
3626772.3657662
[73] Hui Wei, Shenghua He, Tian Xia, Andy Wong, Jingyang Lin, and Mei Han. 2024.
Systematic Evaluation of LLM-as-a-Judge in LLM Alignment Tasks: Explainable
Metrics and Diverse Prompt Templates. arXiv:2408.13006 [cs.CL] https://arxiv.
org/abs/2408.13006
[74] Zeqiu Wu, Ryu Parish, Hao Cheng, Sewon Min, Prithviraj Ammanabrolu, Mari
Ostendorf, and Hannaneh Hajishirzi. 2023. InSCIt: Information-Seeking Conver-
sations with Mixed-Initiative Interactions. Transactions of the Association for Com-
putational Linguistics 11 (2023), 453–468. https://doi.org/10.1162/tacl_a_00559
[75] Sirui Xia, Xintao Wang, Jiaqing Liang, Yifei Zhang, Weikang Zhou, Jiaji
Deng, Fei Yu, and Yanghua Xiao. 2024. Ground Every Sentence: Improv-
ing Retrieval-Augmented LLMs with Interleaved Reference-Claim Generation.
arXiv:2407.01796 [cs.CL] https://arxiv.org/abs/2407.01796
[76] Zitong Yang, Neil Band, Shuangping Li, Emmanuel Candes, and Tatsunori
Hashimoto. 2025. Synthetic continued pretraining. In The Thirteenth Interna-
tional Conference on Learning Representations . https://openreview.net/forum?
id=07yvxWDSla
[77] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan
Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A Dataset for
Diverse, Explainable Multi-hop Question Answering. In Proceedings of the 2018
Conference on Empirical Methods in Natural Language Processing , Ellen Riloff,
David Chiang, Julia Hockenmaier, and Jun’ichi Tsujii (Eds.). Association for
Computational Linguistics, Brussels, Belgium, 2369–2380. https://doi.org/10.
18653/v1/D18-1259[78] Shenglai Zeng, Jiankun Zhang, Pengfei He, Jie Ren, Tianqi Zheng, Hanqing
Lu, Han Xu, Hui Liu, Yue Xing, and Jiliang Tang. 2024. Mitigating the Pri-
vacy Issues in Retrieval-Augmented Generation (RAG) via Pure Synthetic Data.
arXiv:2406.14773 [cs.CR] https://arxiv.org/abs/2406.14773
[79] Shenglai Zeng, Jiankun Zhang, Pengfei He, Yue Xing, Yiding Liu, Han Xu, Jie Ren,
Shuaiqiang Wang, Dawei Yin, Yi Chang, and Jiliang Tang. 2024. The Good and
The Bad: Exploring Privacy Issues in Retrieval-Augmented Generation (RAG).
arXiv:2402.16893 [cs.CR] https://arxiv.org/abs/2402.16893
[80] Saber Zerhoudi and Michael Granitzer. 2024. PersonaRAG: Enhanc-
ing Retrieval-Augmented Generation Systems with User-Centric Agents.
arXiv:2407.09394 [cs.IR] https://arxiv.org/abs/2407.09394
[81] Fengbin Zhu, Wenqiang Lei, Youcheng Huang, Chao Wang, Shuo Zhang,
Jiancheng Lv, Fuli Feng, and Tat-Seng Chua. 2021. TAT-QA: A Question An-
swering Benchmark on a Hybrid of Tabular and Textual Content in Finance.
InProceedings of the 59th Annual Meeting of the Association for Computational
Linguistics and the 11th International Joint Conference on Natural Language Pro-
cessing (Volume 1: Long Papers) , Chengqing Zong, Fei Xia, Wenjie Li, and Roberto
Navigli (Eds.). Association for Computational Linguistics, Online, 3277–3287.
https://doi.org/10.18653/v1/2021.acl-long.254
A Comparison with other QA and RAG
benchmarks
Table 6 contains a comparison with other popular QA and RAG
benchmarks. EnronQA covers the under explored private knowledge
domain using private emails. It has a comparable or larger num-
ber of documents to other resources while covering vastly more
questions. Having multiple questions per document will facilitate
training memorization of factual information in documents, and
enables research finetuning and optimizing RAG pipelines rather
than just serving as a diagnostic benchmark.

EnronQA: Towards Personalized RAG over Private Documents
Benchmark Corpus Size QA Pairs/Turns Domain Source
ConcurrentQA [4] 5.2M + 47k 18.4k General + Private Knowledge Wikipedia + Emails
ConvFinQA [10] 2k 14k Finance Finance Reports
CoQA [61] 8.4k 127k General KnowledgeLiterature, Academia, News,
Wikipedia, Reddit, Exams
CovidQA [51] 147 2k Academic Research Research Papers
CUAD [25] 510 13k Legal Legal Contracts
DelucionQA [63] 1 2k Customer Support Jeep Manual
Doc2Dial [18] 458 25.7k Government Government Sites
DoQA [7] 2.4k 10.9k Cooking, Travel, Movies Stack Exchange
EManual [54] 308k 3.3k Customer Support TV Manual
ExpertQA [49] – 2.2k Expert Knowledge Google Search
FinQA [9] 2.8k 8.3k Finance Finance Reports
HAGRID [35] 32.8M 2.6k General Knowledge Wikipedia
HotPotQA [77] 5.2M 112.8k General Knowledge Wikipedia
HybriDial [53] 2.9k 22.5k General Knowledge Wikipedia
INSCIT [74] 6.6M 4.7k General Knowledge Wikipedia
MS Marco [6] 3.6M 1.01M General Knowledge Web Pages
NarrativeQA [40] 1.6k 46.8k Movie Scripts, LiteratureProject Gutenberg
+ IMSDB
Natural Questions [41] 5.9M 323k General Knowledge Wikipedia
NewsQA [70] 12.7k 119.6k News CNN
PubMedQA [31] 211.3k 273.5k Academic Research Research Abstracts
QReCC [3] 10M 81k General Knowledge Web pages
QuAC [11] 8.9k 98.4k General Knowledge Wikipedia
SearchQA [16] 6.9M 140.5k General Knowledge Google Search
SQA [29] 2.1k 17.6k General Knowledge Wikipedia Tables
Squad 2.0 [60] 536 151k General Knowledge Wikipedia
TAT-QA [81] 182 16.6k Finance Finance Reports
TechQA [8] 802k 1.4k Customer Support Tech Forums
TopiOCQA [2] 5.9M 50k General Knowledge Wikipedia
TriviaQA [33] 662.7k 96k General Knowledge Wikipedia
UDA [27] 3k 29.6kFinance, Academia,
Knowledge BasesFinance, Research Papers,
Wikipedia
EnronQA (Ours) 103.6k 528.3k Private Knowledge Emails
Table 6: Comparison of document based QA benchmarks. EnronQA covers a comparable or larger corpus scale to many popular
QA benchmarks while having vastly more QA pairs enabling training, optimization, and document memorization exploration.
Additionally EnronQA spans the under explored private document domain using emails.

Ryan et al.
B Language Model Prompts
B.1 Initial QA Generation Prompt
Here, we include both the unoptimized prompt for QA generation and the DSPy MIPROv2 [ 55] optimized prompt, including a rewritten
instruction and four bootstrapped few shot examples. This prompt is to seed the question refinement process by creating an initial question
based on the email and distinctive of the prior questions. We optimize with a training set of thirty emails and 20 validation emails. We run
MIPROv2 for 20 iterations (batches) and generate 10 candidate instructions to search over. We use Llama3.1 70b Instruct [ 15] as our prompt
generator model.

EnronQA: Towards Personalized RAG over Private Documents
Initial QA Generation Prompt (Unoptimized)
Given a particular email and a set of prior questions I’ve already asked about this email, write a new question that tests the reader’s
understanding of the email. The question should be specific to the factual contents of this email, and answerable with a single
sentence, and should not be a repeat of any of the prior questions.
—
Follow the following format.
Email: The email we want to test the reader’s comprehension of
Prior Questions: The prior questions I’ve already asked about this email
Reasoning: Let’s think step by step in order to ${produce the new question}. We ...
New Question: The brand new question we want to ask about the email
—
Email: { email }
Prior Questions: { prior questions }
Reasoning: Let’s think step by step in order to

Ryan et al.
Initial QA Generation Prompt (DSPy Optimized) (Part 1)
Imagine you are a high-stakes business consultant, and you must generate a question that tests the reader’s understanding of this
email in order to provide a crucial bit of information needed to close a multi-million dollar deal. You should skew the question to be
*entirely factual*, i.e. do not offer opinions, test the reader’s own thoughts on matters, or request the emails conclusion. Do not
suggest that you are confused. Additionally, the question cannot be answerable with a subjective judgment call. However, it is okay
to use an interrogative that is compound, e.g. ’What is X, and what is Y?’ Writing such a question quickly and to-the-point will
mean receiving a substantial bonus. Given a particular email and a set of prior questions I’ve already asked about this email, write a
new question to ensure success.
—
Follow the following format.
Email: The email we want to test the reader’s comprehension of
Prior Questions: The prior questions I’ve already asked about this email
Reasoning: Let’s think step by step in order to ${produce the new_question}. We ...
New Question: The brand new question we want to ask about the email
—
Email: Subject: LUNCH
Sender: buylow@houston.rr.com
Recipients: [’dfarmer@enron.com’]
File: farmer-d/personal/292.
=====================================
TREEBEARD’S
I am hooked. Name the time and place to meet......................
Tell Ami good luck for me.
Who would have thunk OU would be back so quickly. Now we have to contend
with them, K-State, and the conference championship (if we survive). I don’t
see the Big 12 being #1 if we don’t lighten up on each other.
You want a chuckle—– My daughter is a big NU fan and has gone to a couple
of Baylor games. Says it’s not quite the same as Lincoln; but the
expectations are not high.
Let me know on lunch and I’ll be there.
Ken
=====================================
Prior Questions: None; Ask Anything
Reasoning: Let’s think step by step in order to produce the new_question. We can start by looking at the content of the email. The
sender is discussing a lunch meeting and also mentions a few sports teams, including OU and K-State. They also mention that their
daughter is a fan of NU and has attended Baylor games. We can ask a question that tests the reader’s understanding of the sender’s
family.
New Question: What team is the sender’s daughter a fan of?
—

EnronQA: Towards Personalized RAG over Private Documents
Initial QA Generation Prompt (DSPy Optimized) (Part 2)
Email: Subject: Re: fritzbrinkman resume
Sender: mike.mcconnell@enron.com
Recipients: [’robert.jones@enron.com’]
File: mcconnell-m/all_documents/442.
=====================================
Robert,
Thanks, I’ve talked with EES - Tom White and I will forward on his resume
for consideration. I will also talk with Stan Horton.
Mike
=====================================
Prior Questions: 1.Who did Mike McConnell say he would forward the resume to for consideration, and who else did he plan to talk
with?
Reasoning: Let’s think step by step in order to produce the new_question. We need to analyze the email and the prior questions to
identify a new aspect of the email that hasn’t been covered yet. The prior question focused on the actions Mike McConnell would
take with the resume. A new question could focus on the context of the email, such as the relationship between the sender and the
recipient.
New Question: What is the apparent relationship between Mike McConnell and Robert Jones, based on the tone and content of the
email about fritzbrinkman’s resume?
—
Email: Subject: Asm Dem Plan
Sender: jeff.dasovich@enron.com
Recipients: [’richard.shapiro@enron.com’]
File: dasovich-j/sent/4786.
=====================================
Here’s the document that I received from the lobbyist today. It’s the same
document that I distributed internally yesterday.
Best,
Jeff
—– Forwarded by Jeff Dasovich/NA/Enron on 05/31/2001 05:26 PM —–
Delaney Hunter" dhunter@s-k-w.com
05/30/2001 12:29 PM
Please respond to dhunter
To: Jeff Dasovich (E-mail) jdasovic@enron.com
cc:
Subject: Asm Dem Plan
A little something we got our hands on.....
- Assembly Demo Plan.doc
=====================================
Prior Questions: 1.What is the name of the document Jeff Dasovich received from the lobbyist, and who originally forwarded it to
him?
Reasoning: Let’s think step by step in order to produce the new_question. We can analyze the email to identify the key elements that
have not been covered by the prior questions. The prior question focused on the document’s name and the person who forwarded it.
We can look for other specific details in the email that can be used to create a new question. One detail that stands out is the date and
time the email was forwarded by Jeff Dasovich. We can create a question that tests the reader’s understanding of this specific detail.
New Question: On what date and time did Jeff Dasovich forward the document from the lobbyist to Richard Shapiro?

Ryan et al.
Initial QA Generation Prompt (DSPy Optimized) (Part 3)
—
Email: Subject: Service deals
Sender: holden.salisbury@enron.com
Recipients: [’portland.shift@enron.com’]
File: guzman-m/all_documents/1285.
=====================================
If you are entering a deal for Harbor, LV Cogen, Willamette, or Saguaro copy
a deal from the DEAL TEMPLATE worksheet on the Services Sheet. These deals
are coded to show our profits. Feel free to change any of the cells except
the BUY PRICE and P L that are highlighted in blue. These cells have
formulas that should not be changed. Make sure that you input the correct
BASIS price as well.
Thanks,
Holden
=====================================
Prior Questions: None; Ask Anything
Reasoning: Let’s think step by step in order to produce the new_question. We need to identify the main topic of the email, which is
about entering deals for specific power plants. We also need to consider what information the sender is trying to convey, which is
about using a template and following certain guidelines. We can ask a question that tests the reader’s understanding of the specific
guidelines mentioned in the email.
New Question: What two cells in the deal template should not be changed by the user, according to Holden’s email about entering
deals in the Services Sheet?
—
Email: { email }
Prior Questions: { prior questions }
Reasoning: Let’s think step by step in order to
B.2 Email Selection Prompt
This prompt is used to measure the specificity of the question. If Llama3.1 70b is able to select the correct email which corresponds to the
question out of a list of 10 emails, then we deem the question to be specific.
Email Selection Prompt
Given a set of emails and a question, select the email that best answers the question.
—
Follow the following format.
Emails: The emails we want to select from
Question: The question we want to answer
Reasoning: Let’s think step by step in order to ${produce the selected_email}
Selected Email: The number corresponding to the email that best answers the question
—
Emails: { emails }
Question: { question }
Reasoning: Let’s think step by step in order to

EnronQA: Towards Personalized RAG over Private Documents
B.3 QA Refinement for Specificity Prompt
Here we include both the unoptimized prompt for QA refinement to make questions more specific as well as the DSPy MIPROv2 [ 55]
optimized prompt including a rewritten instruction and one bootstrapped fewshot example. This prompt is used to rewrite questions so that
they are more specific and cannot accidentally refer to several different emails (or be answered by several different emails). This is optimized
in the same end-to-end optimization described in §B.1.
QA Refinement For Specificity (Unoptimized)
Given a question and an associated email, alongside a set of unrelated but similar emails, refine the question to be more specific to
the EXACT email in question. You may consider adding details from the email to the question. Don’t significantly change the
meaning of the question, just make it more specific to the email in question, and answerable with a single sentence.
—
Follow the following format.
Email: The email we want to refine the question for
Question: The question we want to refine
Other Emails: The set of other emails which we DO NOT want the question to be about. Ensure that the refined question has details
that do not apply to these emails.
Reasoning: Let’s think step by step in order to ${produce the new question}. We ...
New Question: The brand new question we want to ask about the email
—
Email: { email }
Question: { question }
Other Emails: { other emails }
Reasoning: Let’s think step by step in order to

Ryan et al.
QA Refinement For Specificity (DSPy Optimized) (Part 1)
You are a high-stakes investigator tasked with uncovering the truth behind a series of mysterious emails. You have a question about
a specific email, but you need to refine it to ensure it’s specific to that exact email and not applicable to other similar emails. The
fate of the investigation rests on your ability to craft a precise question. Given the email and a set of unrelated but similar emails,
refine the question to make it more specific to the exact email in question, adding details from the email as necessary. The question
must be answerable with a single sentence. The entire investigation is counting on you - refine the question carefully.
—
Follow the following format.
Email: The email we want to refine the question for
Question: The question we want to refine
Other Emails: The set of other emails which we DO NOT want the question to be about. Ensure that the refined question has details
that do not apply to these emails.
Reasoning: Let’s think step by step in order to ${produce the refined_question}. We ...
Refined Question: The refined question that is more specific to the email in question
—
Email: Subject: Asm Dem Plan
Sender: jeff.dasovich@enron.com
Recipients: [’richard.shapiro@enron.com’]
File: dasovich-j/sent/4786.
=====================================
Here’s the document that I received from the lobbyist today. It’s the same
document that I distributed internally yesterday.
Best,
Jeff
—– Forwarded by Jeff Dasovich/NA/Enron on 05/31/2001 05:26 PM —–
Delaney Hunter dhunter@s-k-w.com
05/30/2001 12:29 PM
Please respond to dhunter
To: Jeff Dasovich (E-mail) jdasovic@enron.com
cc:
Subject: Asm Dem Plan
A little something we got our hands on.....
- Assembly Demo Plan.doc
=====================================
Question: On what date and time did Jeff Dasovich forward the email to Richard Shapiro?
Other Emails:\u00ab1\u00bb.test - - - - - original message - - - - - from : dasovich, jeff sent : wednesday, november 21, 2001 12 : 37
pm to : dasovich, jeff subject : test
\u00ab2\u00bb.- - - - - original message - - - - - from : dasovich, jeff sent : saturday, october 06, 2001 2 : 44 pm to :’jdasovich @
sbcglobal. net’subject :
\u00ab3\u00bb.Here’s the document that I received from the lobbyist today. It’s the same document that I distributed internally
yesterday. Best, Jeff —– Forwarded by Jeff Dasovich/NA/Enron on 05/31/2001 05:26 PM —– ¨Delaney Hunter ¨<dhunter@s-k-w.com>
05/30/2001 12:29 PM Please respond to dhunter To: ¨Jeff Dasovich (E-mail) ¨<jdasovic@enron.com> cc: Subject: Asm Dem Plan A little
something we got our hands on..... - Assembly Demo Plan.doc

EnronQA: Towards Personalized RAG over Private Documents
QA Refinement For Specificity (DSPy Optimized) (Part 2)
\u00ab4\u00bb.- - - - - forwarded by jeff dasovich / na / enron on 04 / 19 / 2001 10 : 43 am - - - - - jeff dasovich sent by : jeff dasovich
04 / 19 / 2001 10 : 43 am to : psellers @ haas. berkeley. edu cc : subject :
\u00ab5\u00bb.reply message : reply from rick shapiro is thank you to jeff. dasovich @ enron. com | | talked to pat. he wants
original message : jeff. dasovich @ enron. com | | talked to pat. he wants me to talk by phone rather than come to dc. happy to go if
necessary. chk email for leg update. best, jeff
\u00ab6\u00bb.reply message : reply from rick shapiro is thank you to jeff. dasovich @ enron. com | | talked to pat. he wants
original message : jeff. dasovich @ enron. com | | talked to pat. he wants me to talk by phone rather than come to dc. happy to go if
necessary. chk email for leg update. best, jeff
Reasoning: Let’s think step by step in order to refine the question. We can see that the question is asking for the date and time that
Jeff Dasovich forwarded the email to Richard Shapiro. However, the question does not specify which email is being referred to. We
can refine the question by adding details from the email that make it more specific. We can see that the email is about the "Asm
Dem Plan" and that it was forwarded by Jeff Dasovich on 05/31/2001 05:26 PM. We can also see that the email was originally sent by
Delaney Hunter on 05/30/2001 12:29 PM. We can use these details to refine the question.
Refined Question: On what date and time did Jeff Dasovich forward the email with the subject "Asm Dem Plan" and attachment
"Assembly Demo Plan.doc" to Richard Shapiro?
—
Email: { email }
Question: { question }
Other Emails: { other emails }
Reasoning: Let’s think step by step in order to
B.4 QA Refinement from Feedback Prompt
Here we include both the unoptimized prompt for QA refinement to make questions higher quality as well as the DSPy MIPROv2 [ 55]
optimized prompt including a rewritten instruction and two bootstrapped fewshot examples. This prompt is used in the refinement step
to make questions higher quality based on the automatically generated feedback. This is optimized in the same end-to-end optimization
described in §B.1.
QA Refinement From Feedback (Unoptimized)
Given a question and an associated email, alongside feedback on the quality of the question, refine the question to address the
feedback. Don’t significantly change the meaning of the question, just incorporate the feedback and keep the question answerable
with a single sentence.
—
Follow the following format.
Email: The email we want to refine the question for
Question: The question we want to refine
Feedback: Feedback on the quality of the question
Reasoning: Let’s think step by step in order to ${produce the new question}. We ...
Refined Question: The refined question that addresses the feedback
—
Email: { email }
Question: { question }
Feedback: { feedback }
Reasoning: Let’s think step by step in order to

Ryan et al.
QA Refinement From Feedback (DSPy Optimized) (Part 1)
You are a high-stakes team lead tasked with refining questions for a critical project for a Fortune 500 company. Your goal is to
produce high-quality, objective questions that can be used to train AI models. You have been provided with an email, a question
related to the email, and feedback from an expert on the quality of the question. Your task is to refine the question to address the
feedback, without significantly changing its meaning, and ensuring it remains answerable with a single sentence. The quality of
your refined question will directly impact the success of the project and the company’s reputation. Refine the question carefully,
and provide your reasoning for the changes you make.
—
Follow the following format.
Email: The email we want to refine the question for
Feedback: Feedback on the quality of the question
Reasoning: Let’s think step by step in order to ${produce the new question}. We ...
Refined Question: The refined question that addresses the feedback
—
Email: Subject: FW: List of authorized traders
Sender: alterson_thomas@jpmorgan.com
Recipients: [’sara.shackleton@enron.com’]
File: shackleton-s/all_documents/11472.
=====================================
Have you been able to take a look at this?
I hate to be a pest, but we are in the midst of an internal audit right now
and
we are being pressured for this list.
Thank you for your time.
Tom Alterson
Thomas P Alterson
06/05/2001 12:18 PM
To: Sara.Shackleton@enron.com
cc:
Subject: List of authorized traders (Document link not converted)
Sara:
Would you please send me a list of authorized traders for Enron NA Corp? We
are
updating our files and noticed that we are missing an ATL for Enron.
ATL form:
(See attached file: ATL Enron NA Corp.doc)
Thank you,
Tom Alterson
- ATL Enron NA Corp.doc
=====================================

EnronQA: Towards Personalized RAG over Private Documents
QA Refinement From Feedback (DSPy Optimized) (Part 2)
Question: What is the reason given by Tom Alterson for needing the list of authorized traders for Enron NA Corp?
Feedback: Question is not objective. Different annotators answer the same question differently given the same email as context.
Could benefit from more clarity.
Reasoning: Let’s think step by step in order to refine the question. We need to make the question more objective and clear. The
reason given by Tom Alterson for needing the list of authorized traders for Enron NA Corp is mentioned in the email as "we are
updating our files and noticed that we are missing an ATL for Enron." However, this reason is not explicitly stated as the main
reason, but rather as a secondary reason. The primary reason is mentioned in the forwarded email as "we are in the midst of an
internal audit right now and we are being pressured for this list." To make the question more objective, we can rephrase it to focus
on the primary reason mentioned in the email.
Refined Question: What is the primary reason given by Tom Alterson for needing the list of authorized traders for Enron NA Corp
during the internal audit?
—
Email: Subject: FW: List of authorized traders
Sender: alterson_thomas@jpmorgan.com
Recipients: [’sara.shackleton@enron.com’]
File: shackleton-s/all_documents/11472.
=====================================
Have you been able to take a look at this?
I hate to be a pest, but we are in the midst of an internal audit right now
and
we are being pressured for this list.
Thank you for your time.
Tom Alterson
Thomas P Alterson
06/05/2001 12:18 PM
To: Sara.Shackleton@enron.com
cc:
Subject: List of authorized traders (Document link not converted)
Sara:
Would you please send me a list of authorized traders for Enron NA Corp? We
are
updating our files and noticed that we are missing an ATL for Enron.
ATL form:
(See attached file: ATL Enron NA Corp.doc)
Thank you,
Tom Alterson
- ATL Enron NA Corp.doc
=====================================
Question: What is the primary reason given by Tom Alterson for needing the list of authorized traders for Enron NA Corp during
the internal audit?

Ryan et al.
QA Refinement From Feedback (DSPy Optimized) (Part 3)
Feedback: Question is not objective. Different annotators answer the same question differently given the same email as context.
Could benefit from more clarity.
Reasoning: Let’s think step by step in order to refine the question. We need to make the question more objective and clear. The
primary reason given by Tom Alterson for needing the list of authorized traders for Enron NA Corp is mentioned in the email as
"we are in the midst of an internal audit right now and we are being pressured for this list." This reason is related to the internal
audit, but the question could be more specific. We can refine the question to ask about the specific event or situation that is causing
the pressure for the list.
Refined Question: What event or situation is causing pressure for the list of authorized traders for Enron NA Corp, according to
Tom Alterson?
—
Email: { email }
Question: { question }
Feedback: { feedback }
Reasoning: Let’s think step by step in order to
B.5 Question Answering Prompts
These prompts are used to both answer the question given the context of an email or to produce an answer to the question with no grounding.
Forcing the LLM to answer the question without grounding is to ensure that the questions are not too easy to guess or and are not memorized
by popular LLMs.
QA With Email Prompt
Given an email and a question about that email, write the answer to that question in a single sentence.
—
Follow the following format.
Email: The email we want to answer a question about
Question: The question we want to answer about the email
Reasoning: Let’s think step by step in order to ${produce the answer}
Answer: The answer to the question
—
Email: { email }
Question: { question }
Reasoning: Let’s think step by step in order to

EnronQA: Towards Personalized RAG over Private Documents
QA Without Email Prompt
Given a question, write an answer to that question in a single sentence.
—
Follow the following format.
Question: The question we want to answer
Reasoning: Let’s think step by step in order to ${produce the answer}
Answer: The answer to the question
—
Question: { question }
Reasoning: Let’s think step by step in order to
B.6 LLM as a Judge Prompt
This prompt was used for our LLM as a judge to determine whether or not two answers were the same or different. The LLM as a judge
was grounded in the document which helped it determine if additional details in a particular answer were a hallucination or grounded in
factual information. We sampled 100 instances where the LLM as a judge deemed answers to match, and 100 instances where the LLM as
a judge deemed answers as not matches. An author manually labelled these assessments and we determined the LLM as a judge to have
0.98 F1-score (only differing in 2 judgements in both cases with the human judge). This gave us high confidence in using our LLM judge
thoughout our evaluation. It is important to note that we are not using an LLM as a judge to make subjective judgment calls here, but rather
to determine if two open-ended answers match or not. This explains the high accuracy even when LLM as a judge can be unreliable [73].
LLM as a Judge
Given an email, a question about that email, a gold answer to that question, and a student’s potentially correct or incorrect response,
judge whether the answer matches the gold answer.
—
Follow the following format.
Email: The email we want to judge the answer to
Question: The question we want to judge the answer to
Correct Answer: The correct answer to the question
Student Answer: The student answer we want to judge
Reasoning: Let’s think step by step in order to ${produce the correct}
Answer: Whether the answer is correct
—
Email: { email }
Question: { question }
Correct Answer: { correct answer }
Student Answer: { student answer }
Reasoning: Let’s think step by step in order to
B.7 Rule Based Quality Evaluation Prompt
These prompts are used to both answer the question given the context of an email or to produce an answer to the question with no grounding.
Forcing the LLM to answer the question without grounding is to ensure that the questions are not too easy to guess or and are not memorized
by popular LLMs.

Ryan et al.
Rule Based Quality Evaluation
Given a particular email, a question about the email, and the answer to that question, judge the quality of the question. The question
should follow the following criteria:
1. The question should be specific to the factual contents of this email. In other words, if you were given 100 emails, would you
know that the question was about this email and not another one? Phrases such as "the email" or "the message" are not specific
enough unless grounded by further context. Just the sender alone may not be enough context, but the sender and some detail about
the email might be.
2. The question should focus on the main contents of the message, not on the formatting, the sender, or recipients. It is okay to use
the sender and recipients as context, but the question should not be about them. It is okay to ask about things like the cell phone
numbers or contact details of the people in the email since this is a realistic question.
3. The question should be objective and answerable with a single sentence. It should not be a matter of opinion or require any
interpretation.
4. The question should be realistic to what a person might ask about an email they received, especially in the context of working in
a professional setting and recalling important details.
5. The question should NOT require any external knowledge beyond the contents of the email itself.
6. The question should NOT include counting, math, or any other operations. For example "How many times did the sender mention
the word ’important’?" or "how many recipients were there?" are not allowed. It is okay for the question to ask about a number in
the email such as a percentage, but it should not involve actually counting or calculating.
Given these rules, rate the question as either "good" or "bad".
—
Follow the following format.
Email: The email we want to answer a question about
Question: The question we want to answer about the email
Answer: The answer to the question
Reasoning: Let’s think step by step in order to ${produce the quality}
Quality: Whether the question is good (true) or bad (false)
—
Email: { email }
Question: { question }
Answer: { answer }
Reasoning: Let’s think step by step in order to