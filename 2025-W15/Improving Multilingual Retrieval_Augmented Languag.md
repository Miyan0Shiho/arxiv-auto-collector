# Improving Multilingual Retrieval-Augmented Language Models through Dialectic Reasoning Argumentations

**Authors**: Leonardo Ranaldi, Federico Ranaldi, Fabio Massimo Zanzotto, Barry Haddow, Alexandra Birch

**Published**: 2025-04-07 06:55:15

**PDF URL**: [http://arxiv.org/pdf/2504.04771v1](http://arxiv.org/pdf/2504.04771v1)

## Abstract
Retrieval-augmented generation (RAG) is key to enhancing large language
models (LLMs) to systematically access richer factual knowledge. Yet, using RAG
brings intrinsic challenges, as LLMs must deal with potentially conflicting
knowledge, especially in multilingual retrieval, where the heterogeneity of
knowledge retrieved may deliver different outlooks. To make RAG more
analytical, critical and grounded, we introduce Dialectic-RAG (DRAG), a modular
approach guided by Argumentative Explanations, i.e., structured reasoning
process that systematically evaluates retrieved
  information by comparing, contrasting, and resolving conflicting
perspectives. Given a query and a set of multilingual related documents, DRAG
selects and exemplifies relevant knowledge for delivering dialectic
explanations that, by critically weighing opposing arguments and filtering
extraneous content, clearly determine the final response. Through a series of
in-depth experiments, we show the impact of our framework both as an in-context
learning strategy and for constructing demonstrations to instruct smaller
models. The final results demonstrate that DRAG significantly improves RAG
approaches, requiring low-impact computational effort and providing robustness
to knowledge perturbations.

## Full Text


<!-- PDF content starts -->

Improving Multilingual Retrieval-Augmented Language Models through
Dialectic Reasoning Argumentations
Leonardo Ranaldi(†)Federico Ranaldi(‡)
Fabio Massimo Zanzotto(‡)Barry Haddow(†)Alexandra Birch(†)
(†)School of Informatics, University of Edinburgh, UK
(‡)Department of Enterprise Engineering, University of Rome Tor Vergata, Italy
{first_name.last_name}@ed.ac.uk
Abstract
Retrieval-augmented generation (RAG) is key
to enhancing large language models (LLMs) to
systematically access richer factual knowledge.
Yet, using RAG brings intrinsic challenges, as
LLMs must deal with potentially conflicting
knowledge, especially in multilingual retrieval,
where the heterogeneity of knowledge retrieved
may deliver different outlooks.
To make RAG more analytical, critical and
grounded, we introduce Dialectic-RAG (D-
RAG ), a modular approach guided by Argu-
mentative Explanations , i.e., structured reason-
ing process that systematically evaluates re-
trieved information by comparing, contrasting,
and resolving conflicting perspectives. Given
a query and a set of multilingual related doc-
uments, D-RAG selects and exemplifies rel-
evant knowledge for delivering dialectic ex-
planations that, by critically weighing oppos-
ing arguments and filtering extraneous content,
clearly determine the final response. Through
a series of in-depth experiments, we show the
impact of our framework both as an in-context
learning strategy and for constructing demon-
strations to instruct smaller models. The fi-
nal results demonstrate that D-RAG signifi-
cantly improves RAG approaches, requiring
low-impact computational effort and providing
robustness to knowledge perturbations.
1 Introduction
Retrieval-augmented Generation (RAG) has
emerged as a promising approach for grounding
(LLMs) responses by incorporating relevant knowl-
edge from external sources through structured
retrieval mechanisms (Guu et al., 2020). RAG was
conceived to handle the limitations of LLMs, such
as their inclination towards hallucinations and
the lack of knowledge of the specialized domain
in their training data (Siriwardhana et al., 2023;
Zhang et al., 2023).Contextualising questions by adding relevant in-
context knowledge retrieved from external corpora,
such as Wikipedia, effectively reduced inaccurate
generation, thereby notably improving accuracies.
However, there are still limitations associated
with RAGs; recent studies have shown ongoing
challenges arising from the retrieved knowledge,
where irrelevant or contradictory documents may
introduce biases in the models (Menick et al.,
2022). These weaknesses arise from the inability
of RAG strategies to critically asses the retrieved
knowledge (Ranaldi et al., 2024b).
Prior approaches improve the RAG pipeline by
incorporating external tools (Li et al., 2023; Yoran
et al., 2024) or employ multi-step reasoning strate-
gies (Zhao et al., 2024; Zhang et al., 2024) to de-
termine the relevance of in-context passages. How-
ever, these methods may require high computa-
tional costs and definitely do not impact smaller-
scale LLMs. Recently, Xia et al. (2024); Ranaldi
et al. (2024b) proposed efficient approaches to en-
able LLMs to deliver argumentative reasoning tra-
jectories. Yet, their effort is on English-centric
RAG, and this can be a limitation for the limited
variance of retrieved knowledge and the operability
(Chirkova et al., 2024; Ranaldi et al., 2025).
In this paper, we present Dialectic-RAG (D-
RAG ), a modular framework conceived to enhance
multilingual retrieval-augmented language models
to follow a Dialectic Reasoning , i.e., a structured
analytical process that critically examines retrieved
knowledge, resolves conflicting perspectives or ir-
relevant passages, and constructs well-supported
responses through structured argumentation (Fig-
ure 1). To achieve this, D-RAG , employs Argu-
mentative Explanations , which systematically con-
trast opposing aspects or filter out irrelevant infor-
mation irrelevant, ensuring a coherent and well-
grounded final answer. D-RAG is designed to
enhance the original RAG pipeline by leading the
model to leverage knowledge-intensive questionsarXiv:2504.04771v1  [cs.CL]  7 Apr 2025

英格 兰 有多少位女王 
[1] [ 英国曾有  13 位君主 ...
Retrieval Augmented LLM Baseline 
Dialectic RAG 
- [3] states that eight queens ruled the Kingdom of 
England, naming Matilda, Lady Jane Grey, Mary I, Elizabeth 
I, Mary II, Anne, Victoria, and Elizabeth II. 
- [4] focuses specifically on Queen Anne , who ruled from 
1702 and became the first monarch of the UK in 1707 
following the union of England and Scotland. 
-[1] outlines the formation of the UK and mention 13  
monarchs .
- [5] mentions Elizabeth II  and her rule over the UK  but 
does not address the total number of queens in England. -Doc [3]: Claims there were eight queens  in 
England's history. (Relevant): 
-Doc [4]: Discusses Queen Anne's reign and her  
role as the monarch when England and Scotland 
unified in 1707. (Partially Relevant) 
-Doc [1]: Explains the timeline of the United  
Kingdom's formation but does not reference the 
specific number of queens. (Partially Relevant) 
-Doc [5]: Mentions Elizabeth II's reign over the  
United Kingdom  but focuses on her role in the 
Commonwealth. (Irrelevant) [3] In less than 200 years, eight  queens 
ruled the Kingdom of England . 
The number of queens in England is clearly  
addressed in [3], which identifies eight queens and  
names them. Other documents, such as [4], 
provide detailed context about specific queens like 
Anne but do not add to the overall count. 
Document [1] mentions the monarchs of UK. [5] is 
irrelevant for this specific question as they focus on 
the broader political structure and modern 
monarchy without detailing the number of queens. 英格 兰 有八位女王。 Answer 
813  7
Question Figure 1: Our D-RAG allows LLMs to leverage multilingual knowledge-intensive question answering tasks by
delivering argumentative explanations that support the final answer.
and retrieve supporting evidence through step-wise
reasoning explanations that, starting from a given
query, follow these steps: (a) extraction , where
a multilingual query and documents are analysed
to identify information relevant for answering the
query; (b) explanation , where the LLMs construct
single arguments about the relevance of the ex-
tracted passages, highlighting and distinguishing
the furnished information; (c) dialectic argumenta-
tion, where the arguments are consolidated using
a neutral perspective into a single final explana-
tion; and (d) answer , where a short-form answer is
delivered.
To evaluate the efficacy of D-RAG , we operate
in two different configurations – as an in-context
approach to provide explicit instructions for larger
and more capable LLMs and as a strategy for con-
structing synthetic demonstrations to improve the
performance and align the reasoning capabilities
of smaller LLMs.
Our empirical analysis carried out three different
public knowledge-intensive question-answering
tasks that covered 11 different languages, show-
ing the following results and conclusions:
•D-RAG elicits LLMs to deliver dialectic rea-
soning trajectories by exploiting multilingual
knowledge in the retrieved documents, signifi-
cantly enhancing performance over baselines,leading to an average accuracy increase of
51.6% without RAG and of 12.9% over RAG
when used with GPT-4o.
•Using D-RAG to construct synthetic dialectic
multilingual reasoning demonstrations signif-
icantly improves the performance of smaller
models, leading to an average increase in ac-
curacy of 9.6% over RAG and 5.5% over
instruction-tuning strategies for RAG when
used with Llama3-8B.
•We conduct an in-depth analysis of the com-
ponents of D-RAG , showing the benefits of
the components and the effects they have with
definitely contradictory scenarios (best ex-
emplified by the real uses-case questions pre-
sented in BORDER LINES (Li et al., 2024)
augmented with multilingual retrieved doc-
uments).
•Finally, we show that D-RAG is robust to per-
turbations that are a limitation for traditional
RAG models, including misleading retrieval
and misleading reranking (i.e., random shuf-
fling of the retrieved documents).

2D-RAG Dialectic Reasoning in
Multilingual RAG
Retrieval-augmented generation (RAG) enriches
data access in large language models (LLMs), but
they struggle to critically evaluate retrieved knowl-
edge, handle conflicts, and filter out irrelevant con-
tent. Integrating critical reasoning into LLMs is
essential to resolve information disputes and en-
sure more coherent and grounded responses (Xia
et al., 2024; Ranaldi et al., 2024b). To instruct a
LLM in deliver dialectic multilingual reasoning
trajectories in a Retrieval-Augmented Language
Model (RAG) setting, we propose a modular strat-
egy (Figure 1) formed of: (a) extraction (§2.1),
where, given a query a set of multilingual retrieved
documents, the model identify relevant informa-
tion; (b) argumentation (§2.2), where the model
delivers argumentative motivations about the ex-
tracted information, by displaying and discerning
the relevancy about the aspects; (c) dialectic argu-
mentation (§2.3), where the arguments constructed
in(b)are summarised using a dialectic and neutral
perspective into a single explanation; (d) answer-
ing(§2.3), where a final answer to the query is
generated adhering to query constraints such as
query-language and the compact form of the an-
swer as reported in Appendix B.
We then use D-RAG in two scenarios as an in-
context learning strategy and a synthetic generator
for constructing demonstrations (§2.5). For the
in-context learning strategy, we use D-RAG to in-
struct LLMs to follow step-wise dialectic planning
that improves the base RAG pipelines (§2.5.1). For
the instruction-tuning, we use the synthetic demon-
strations to improve smaller LLMs (§2.6) and trans-
fer to them the capability of leveraging the query
and the retrieved knowledge for delivering a robust
argumentation to reach the answer.
2.1 Extraction
The first step, which we define as α1in the pro-
posed pipeline, concerns extracting relevant re-
trieved knowledge from documents retrieved from
a given knowledge base K. Complementary to
previous approaches (Xia et al., 2024; Ranaldi
et al., 2024b) in this paper, we operate in a multilin-
gual retrieval scope (where documents come from
knowledge bases in multiple languages as defined
in §3.1). We operate via multilingual retriever sys-tems provided by Cohere1as the default retriever
model R. Thereafter, we instruct the model to
analyse the query and understand and identify the
main points from the retrieved documents (i.e.,
"#Reference Evidence ") for answering the ques-
tion and label this phase as " #Extraction ". Since
we work with multilingual queries and documents,
this step is crucial to aid the model in planning the
reasoning.
2.2 Explanations
The second step, defined as α2, concerns instruct-
ing the model to discuss the extracted informa-
tion and deliver argumentations. Specifically, after
identifying and extracting information from the
top-kdocuments, we prompt the model to discuss
whether they are actually relevant or irrelevant to
the query by clearly citing the passages and la-
belling this phase as " #Explanation ".
2.3 Dialectic Reasoning
This step, which we define as α3, concerns generat-
ing a final comprehensive explanatory summary. In
particular, for α3, we leverage the arguments in the
previous steps to deliver the final explanation that
argues the motivations that support the answer us-
ing a dialectic approach, i.e. a critical approach that
relies on systematic comparison to arrive at a more
articulate and well-founded conclusion. Hence, we
instruct the LLM to consider the generated aspects,
summarise the main points into a single argumen-
tation, and head this as “ #Dialectic Argumenta-
tion: ”.
2.4 Final Answer
The last step is defined as α4and results in a short-
form answer used in the final evaluation. We in-
struct the model to generate the final answer in
this form and in the same language as the query
following the pattern " #Answer: ".
2.5 D-RAG Application
2.5.1 D-RAG as in-context Learning
We adopt D-RAG as in-context learning strat-
egy by instructing different LLMs to answer
knowledge-intensive questions by dealing with re-
trieved knowledge. D-RAG , in a modular way,
identify the most critical information from the re-
trieved documents (§2.1), arguing the rationale sup-
porting the selection of appropriate points to an-
1https://huggingface.co/Cohere/Cohere-embed-
multilingual-v3.0

swer the query by explaining the main passages
(§2.2), deliver a single argumentation that best de-
scribes the points (§2.3); and finally, generate the
final short-form answer in a strict format, to have
a more detailed and strict downstream evaluation.
Yet, although the sequence of instructions is well-
structured and defined, the ability to perform se-
quential and complex reasoning tasks is limited to
larger LLMs (such as GPT-4o, as discussed in the
experiments). Hence, we transfer these capabili-
ties to smaller models operating via D-RAG for
building synthetic demonstrations as training sets.
2.5.2 D-RAG as a Synthetic Annotation
We instruct smaller models via demonstrations pro-
duced by high-performing LLMs capable of fol-
lowing structured instructions. In contrast to the
methods proposed in (Asai et al., 2023), we use
a single prompt composed of a sequence of in-
structions in a multilingual setting. To filter the
quality of generated demonstrations, we follow the
method proposed by Xia et al. (2024); Ranaldi et al.
(2024b), which computes the citation precision for
the considered documents as a proxy for the qual-
ity of the demonstrations. However, since D-RAG
employs a different annotation mechanism, our an-
notation pipelines firstly filter out the final correct
answers through a strict, exact match; then, after
the filtering (which removes more than half of the
annotated demonstrations), it verifies that the pro-
vided instructions have been considered. We detail
the description of annotation in Appendix D.
2.6 Tuning Smaller Models
We fine-tune a Language Model θusing the annota-
tions2generated via D-RAG . The annotations are
augmented with demonstrations αusing the stan-
dard language modelling objective to maximize the
expected log-likelihood:
θ∗= arg max
θE(Q,α,Y )∼D[logpθ(Y, α|Q)]
where θ∗denotes the optimal model parameters,
andpθ(Y, α|Q)is the joint probability of the out-
putYand the demonstrations αconditioned on the
query Q, learned from the training corpus Daug-
mented with contrastive reasoning demonstrations.
While α=α1·α2·α3·α4is the combination
of the multiple reasoning steps performed by the
model, " ·" is the concatenation operator, and αi
2we select annotations as described in §2.5.2are the respective paths generated by the overhead
processes. Qis the provided query, and Yis the
output, including the intermediate steps and the
final answer that compose the training corpus D.
3 Experimental Setup
We evaluate D-RAG on five open-domain
question-answering tasks (§3.1). We perform the
retrieval and evaluation phases by following stan-
dard approaches used to assess the RAG pipeline
(§3.2) and perform the tuning phase by using the
setup presented in §3.3.1.
3.1 Tasks & Datasets
We use the following question-answering (QA)
tasks: (i)MLQA (Lewis et al., 2020a), (ii)MKQA
(Longpre et al., 2021) and (iii) XOR-TyDi QA
(Asai et al., 2021) as they best represent multilin-
gual open-ended question-answering tasks. Then,
we use BORDERLINES (Li et al., 2024), which
contains multilingual questions concerning con-
flicts over disputed territories (note: we follow
the questions and targets delivered by Li et al.
(2024)). Finally, we include Natural Questions
(NQ) (Kwiatkowski et al., 2019a), as it is a widely
used English benchmark for assessing RAG sys-
tems. This allows us to establish meaningful base-
lines for comparison. Appendices C and M report
the languages and composition of each dataset. Ap-
pendix N reports detailed information about BOR-
DERLINES .
3.2 Experimental Settings
Retrieval In our work, we employ Wikipedia as
the knowledge base Kand Cohere as the retrieval
system R. Specifically, by working through the
Wikimedia dump provided by Cohere3, individ-
ual articles are embedded with the state-of-the-art
multilingual embedding model Cohere_Embed_V3 .
This pipeline makes it easy to search Wikipedia for
information or to use only specific languages. For
each question in the evaluation data, we retrieve
the top-5 relevant documents (details Appendix I).
Models & Inference Settings To get a com-
prehensive evaluation of existing RAG pipelines
in the main experiments, we use four different
LLMs: GPT-4o (OpenAI, 2023), Llama3-70b-
instruct (Grattafiori et al., 2024) and smaller mod-
3Cohere/wikipedia-2023-11-embed-multilingual-v3

els Llama3-8b-instruct and 1b-instruct4. Detailed
settings and model versions are in Appendix F. We
use greedy decoding in all experiments to ensure a
more deterministic generation process, and we set
the temperature to 0 and the maximum generation
length to 2048. We observed that these settings
deliver better and deterministic performances.
3.3 Evaluation Metrics
We use flexible exact-match accuracy following
Schick et al. (2023), which is based on whether
or not ground-truth answers are included in the
generated answers provided by the models instead
of a strict exact match. Moreover, our prompting
pipelines instruct the models to use as a final label
‘#Answer’ (see Appendices A) to elicit a conclu-
sive generation that contains a short-form answer.
3.3.1 Training Setting
To evaluate the impact of D-RAG reasoning
demonstrations on smaller models (§2), we em-
ploy the annotations produced following the D-
RAG strategy (§2.5.2). Further, for a fair compar-
ison, we deliver annotations using Llama-3-SFT,
where Llama is tuned on training samples without
D-RAG (annotation generated using same query,
retrieved documents and the prompt in Table 5).
We fine-tune the models for three epochs with a
batch size of 32 and a learning rate equal to 1e-5
with a 0.001 weight decay. We use the cosine learn-
ing rate scheduler with a warmup ratio of 0.03. We
conducted our experiments on a workstation with
four Nvidia RTX A6000 and 48GB of VRAM.
3.4 Evaluated Methods
We propose the following settings:
Baseline - without RAG We evaluate the base-
line capabilities of selected models in a zero-shot
way without introducing any documents (without
RAG) using the instruction (prompt) in Table 4.
Retrieval Augmented LLM (RAG) We assess
the impact of retrieved knowledge by instructing
the evaluated models to consider the top-5re-
trieved documents. We use the retrievers in §3.2.
→ICL As baseline settings we use the instruction
in Table 5.
→D-RAG (ICL) To complete the RAG-based
settings, we use D-RAG as an in-context learning
strategy as in Table 6.
4to simplify notation we omit instruct for the rest of the
paper→fine-tuning Finally, we tune Llama models us-
ingSFT andD-RAG as presented in §3.3.1 and
prompt using RAG instruction (Table 5).
Models MKQA MLQA X.TyDi Avg
Baseline
Llama3-1B 32.5 33.7 27.3 31.2
Llama3-8B 38.9 43.4 34.5 38.6
Llama3-70B 40.7 43.9 36.5 40.4
GPT-4o 44.8 46.9 36.7 42.8
RAG
Llama3-1B 50.6 48.6 41.7 46.9
Llama3-8B 57.3 54.5 48.1 53.1
Llama3-70B 60.1 56.6 49.2 55.3
GPT-4o 61.4 58.6 51.2 57.4
RAG→D-RAG as ICL
Llama3-1B 48.6 48.0 38.3 45.0
Llama3-8B 56.7 53.5 48.1 52.8
Llama3-70B 67.3 62.4 55.8 62.4
GPT-4o 68.2 65.5 60.7 64.8
RAG→tuning via SFT andD-RAG
Llama3-1B SFT 52.1 50.0 41.3 47.8
Llama3-8B SFT 60.3 56.3 48.5 55.0
Llama3-1B D-RAG 55.8 53.7 46.6 51.9
Llama3-8B D-RAG 63.6 59.3 52.7 58.5
Table 1: Average results on multilingual QA tasks
(§3.1). Models instructed as detailed in §3.4. In bold,
best performances of ICL and fine-tuned models.
4 Results
The results in Table 1 show that D-RAG aids the
models in leveraging retrieved documents for mul-
tilingual QA tasks, showing the impact of dialectic
reasoning argumentations on RAG. We found that
D-RAG is effective as an in-context learning ap-
proach in larger LLMs and is notably helpful as a
demonstration strategy to improve the performance
of smaller models, achieving solid performance
compared to fine-tuning approaches. To this end,
the following sections analyse the impact of D-
RAG when adopted as both an in-context strategy
(§4.1) and as a framework for generating annota-
tions to instruct LLMs (§4.2). Then, in §4.4, we
study a practical application on BORDER LINES (Li
et al., 2024). Finally, we investigate the role of the
argumentative explanations (§4.3) and revealed ev-
idence of robustness on challenging perturbations
and functionality in low-resource settings (§4.5).
4.1 D-RAG in-context learning
Table 1 reports the results of D-RAG when
adopted as an in-context learning (ICL) strat-

egy for different models. We observe an over-
all improvement over the baseline models with-
out retrieved documents (relative improvements of
51.4% for GPT-4o, 54.5% for Llama3-70B, 36.7%
for Llama3-8B and 44.2% for Llama3-1B on abso-
lute average score); however, the results show that
the impact of D-RAG in a RAG setting emerges for
GPT-4o and Llama3-70B where D-RAG achieves
a general improvement of 12.9% and 11.9% re-
specting to RAG. In contrast, for Llama3-8B and
Llama3-1B, we observe a decrease in performance
compared to the RAG pipeline, suggesting that
these smaller models cannot deliver the dialectic
reasoning explanations required to support their
responses.
4.2 D-RAG Annotation Approach
Table 1 reports the impacts of D-RAG used as an
annotation strategy for different smaller models
(denoted as RAG→tuning via SFT andD-RAG ).
D-RAG effectively enhances the performance of
smaller models when employed to deliver reason-
ing demonstrations via GPT-4o. Hence, we found
thatD-RAG outperform SFT approaches for both
model versions.
It emerges that both tuning strategies work well
and outperform the baseline RAG approaches—for
instance, Llama3-8B improves 52.8 →55.0 av-
erage accuracy comparing RAG and SFT ver-
sions. However, the models tuned via D-RAG an-
notations consistently surpass the SFT (Llama3-
1B improves +3.1 and Llama3-8B +2.9 average
points). These results indicate the benefits pro-
vided by D-RAG demonstrations and their ability
to efficiently elicit argumentations in smaller LMs.
Finally, we compared D-RAG with similar work
focusing on English. We showed that although
tuning is multilingual, D-RAG achieves sustain-
able performance. In contrast, related methods
definitely underperform when operating in multi-
lingual tasks.
4.3 The Role of D-RAG Components
Figures 2 evaluate the impact of our D-RAG frame-
work on the final performance as used as an in-
context learning approach and as a tuning strategy.
The results in Figure 2 demonstrate the importance
of each phase in the reasoning process introduced
in §2. For GPT-4o and Llama3-70B, we observe
the highest decrease in performance when remov-
ing the second and third steps. In particular, remov-
ing the second step (w/o Step 2), also defined as
Figure 2: Performance differences (∆)for GPT-4-o and
Llama3-70B. We analyse the impact of each component
on MKQA by eliminating (w/o) the D-RAG steps.
α2, which is concerned with arguing and breaking
down relevant points of retrieved documents to an-
swer the given query, it is possible to observe an
average decrease of -5.2% compared to D-RAG .
Removing Step 3, which is responsible for deliv-
ering the argumentation, we observe an average
reduction of -6.5% compared to D-RAG . These
results demonstrate the crucial impact of each pas-
sage of D-RAG for eliciting dialectic explanations
from the model. The impact of steps for ICL op-
eration affects the tuning as well. As reported in
detail in appendix the models tuned via modified
D-RAG or randomly mixed steps negatively im-
pact performance (the crucial points are Steps 2
and 3 as in the case of D-RAG as ICL).
Model %Agreement %Agreement
English (En) X,Y,En
GPT-4o 75% 66.6%
+RAG 85% 81.6%
+D-RAG 100% 100%
Llama3-8B ICL 35% 43.3%
+RAG ICL 50% 51.6%
+D-RAG ICL 65% 68.3%
Llama3-8B SFT 65% 70%
Llama3-8B D-RAG 95% 98.3%
Table 2: Agreement rate with controller in BORDER -
LINES dataset (Li et al., 2024). Details in Appendix N.
4.4 Dialectic Reasoning in B ORDER LINES
To investigate the impact of our D-RAG in real
contexts, we used BORDER LINES (Li et al., 2024).
This resource provides questions concerning dis-
puted territories as detailed in Appendix N. These
questions are in English and in two additional lan-
guages, which are the land disputants (defined as

Figure 3: Robustness experiment results on QA datasets (§3.1). We provide retrieved documents by randomly
shuffling them (Random Shuffle) and introducing two misleading (irrelevant) documents (Random Noise).
XandY). Finally, a target or controller value indi-
cates the country that controls the territory5. To
study the consistency and dialectic capabilities of
ourD-RAG , we then conducted a retrieval phase
and evaluated GPT-4o and Llama3-8B (tuned and
not) with the questions in the specific languages
and English using the prompts defined in Appen-
dices A and B. Then, setting the controller as X, we
estimated the percentage of times the answer pro-
vided by the models prompted in English matched
with the target or named controller (denoted as
%Agreement English ), and the percentage when
the models prompted via queries in three languages
matches among them and with the controller.
Table 2 shows that the consistency percentage
increases when D-RAG is used. In particular, in
GPT-4o, there is a 15% and 19.6% increase when
D-RAG is compared with RAG. Similarly, it oc-
curs between Llama3-8B instructed via D-RAG .
Finally, Llama3-8B tuned with DRAG has the most
robust constancy.
4.5 Additional Analysis
Robustness To test the robustness of the pro-
posed framework and avoid the possible perfor-
mance bias obtained from noisy or misleading re-
trieval, we follow the methodology used in previ-
ous works. Hence, we shuffled the order of the
retrieved documents (Random Shuffle) and inserted
two misleading and irrelevant documents (Random
Noise) . Figure 3 reports the results. We show
thatD-RAG consistently outperforms the baseline
model with RAG as ICL and annotation strategy.
In particular, the random shuffling of retrieved doc-
uments minimally impacts performance, demon-
strating the permutation invariance property of D-
RAG (see the subfigure on the right). Moreover,
when noisy documents are added, all the evaluated
models suffer a higher performance drop. How-
ever, the drop for D-RAG is typically lower than
5in some cases, there are no defined places that we do not
consider in our analysis.the standard RAG approach, which shows that the
proposed method is more robust even when dealing
with noisier results.
Figure 4: Performances assessment of Llama3-8B and
-1B by scaling D-RAG (lines) and SFT (bars) tuning
demonstrations on ablation set (Appendix E).
Quantity of Instructions Figure 4 shows the be-
haviour of D-RAG when scaling-up the number
of training examples. While we found that the
quantity of the demonstrations used in D-RAG
is important in determining the final performance,
we found that D-RAG can outperform the baseline
RAG models with only 50% of training demonstra-
tions, also achieving superior training performance
when compared to the fine-tuned SFT model (i.e.,
the model fine-tuned without D-RAG demonstra-
tions as explained in §3). This further highlights
the quality of the training signal provided by the
contrastive explanations.
Quality of Generation Table 3 shows the ten-
dency to generate answers in the same query lan-
guage and follow the provided instructions at infer-
ence time (we describe the experimental method-
ologies in Appendix K ). In particular, two require-
ments that our framework must satisfy are i)all
instructions given in the prompt must be followed,
andii)in the multilingual task, the answer must
be in the same query language. In both cases, we
observe that the GPT4-o and Llama3-70B are con-
sistent with the requirements. On the other hand,

the two Llama3 models do not follow the instruc-
tions, but when tuned, employing demonstrations
from D-RAG, they become consistent.
Models IF CL LR-IF LR-CL
GPT-4o - 85.6% - 72.2%
+D-RAG ICL 90.5% 94.8% 83.6% 86.4%
Llama3-70B - 65.2% - 63.8%
+D-RAG ICL 83.5% 79.4% 77.4% 70.2%
Llama3-8B - 65.9% - 46.0%
+SFT - 72.8% - 64.6%
+D-RAG ICL 58.4% 66.2% 45.5% 44.0%
+D-RAG FT 78.3% 72.0% 67.1% 69.6%
Llama3-1B - 57.2% - 30.4%
+SFT - 66.3% - 48.8%
+D-RAG ICL 40.0% 53.3% 40.7% 32.2%
+D-RAG FT 60.4% 69.5% 45.3% 59.9%
Table 3: Percentage (%) of answers that follow the
prompt instructions (IF) and generate the final answer
in the correct language (CL). FTindicates fine-tuned
models via D-RAG .LRindicates the results for low-
resource languages considering the MKQA answers.
D-RAG Settings & Comparisons We provide
evidence for the robustness of the D-RAG by
proposing three experiments. Firstly, we show that
decomposing our D-RAG into different prompts
delivers benefits which are minimal compared to
the cost of increasing the number of prompts (four
prompts against a single one). Then, in Appendix
J, we analyse the impact of internal argumentation
in the query language. As shown in Table 13, ar-
gumentation in a language other than English (a
language in which the models used in this work
are more profitable) leads to a drop in performance
that will definitely be a matter of future investi-
gation. Finally, we show that D-RAG perform
well even in monolingual tasks (English). In con-
trast, related methods achieve lower performance
in multilingual tasks.
5 Applicability & Future Work
Our experiments evaluate a method to improve
RAG capabilities in multilingual scenarios by elic-
iting LLMs to consider heterogeneous sources of
knowledge and argue the reasons that support the
answer in a dialectic manner. The applicabilities
of our work are related to: (i)improving the an-
swering of questions that involve a retrieval in a
setting with unbalanced resource availability, e.g.,
in the case of Wikipedia, where the number of
documents differs from languages (Table 12). (ii)improving the argumentation in scenarios where
there is an information overlap on retrieved state-
ments that support the outcomes, as studied in §4.4.
(iii)Transferring the capabilities of delivering di-
alectic explanations to smaller LLMs by teaching
them via synthetic demonstrations. In future de-
velopments, we plan to analyse the role different
languages can play in delivering reasoning and
how much the multilingual proficiency of LLMs
can influence this task.
6 Related Work
Lewis et al. (2020b) investigated the advantages
of augmenting LLMs with retrieved knowledge,
a technique known as Retrieval-augmented Lan-
guage Models (RAG). Shi et al. (2023) demon-
strated that the benefits of RAG could be under-
mined by noisy retrieval. Several studies have
enhanced RAG through in-context solutions, tun-
ing, or retriever interventions (Menick et al., 2022;
Jiang et al., 2023; Gao et al., 2023; Sawarkar et al.,
2024). While effective, in-context learning only
partially mitigates retrieval bias, and tuning re-
mains costly (Asai et al., 2023). Xia et al. (2024)
proposed low-impact reasoning techniques, later
enhanced via contrastive reasoning by Ranaldi
et al. (2024b). Unlike these English-centric ap-
proaches, we focus on multilingual knowledge-
intensive tasks. Complementing (Zhang et al.,
2022), we study the inference phase and enrich the
work proposed by Chirkova et al. (2024); Ranaldi
et al. (2025). We propose a framework that allows
the LLMs to leverage different knowledge, reason
about it, and deliver argumentative explanations
by using a dialectic approach. Our effort aims to
improve the limitations of multilingual RAG, bias
towards language, information disparity (Sharma
et al., 2024) or conflicting knowledge (Li et al.,
2024).
7 Conclusion
RAG has demonstrated its potential to improve
LLM performances in knowledge-intensive tasks;
however, a major limitation lies in handling het-
erogeneous retrieved, especially in multilingual
cases. To address this, we propose Dialectic-RAG
(D-RAG ) to improve retrieval-based reasoning
through argumentative explanations. We show
thatD-RAG significantly improves multilingual
retrieval-augmented inference, enhancing both in-
context learning and demonstration-based instruc-

tion for smaller models. Structuring reasoning
over retrieved knowledge mitigates misleading in-
ferences and improves response consistency, rein-
forcing the importance of dialectic reasoning for
reliable multilingual RAG applications.
References
Akari Asai, Jungo Kasai, Jonathan Clark, Kenton Lee,
Eunsol Choi, and Hannaneh Hajishirzi. 2021. XOR
QA: Cross-lingual open-retrieval question answering.
InProceedings of the 2021 Conference of the North
American Chapter of the Association for Computa-
tional Linguistics: Human Language Technologies ,
pages 547–564, Online. Association for Computa-
tional Linguistics.
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2023. Self-rag: Learning to re-
trieve, generate, and critique through self-reflection.
Laurie Burchell, Alexandra Birch, Nikolay Bogoychev,
and Kenneth Heafield. 2023. An open dataset and
model for language identification. In Proceedings
of the 61st Annual Meeting of the Association for
Computational Linguistics (Volume 2: Short Papers) ,
pages 865–879, Toronto, Canada. Association for
Computational Linguistics.
Nadezhda Chirkova, David Rau, Hervé Déjean,
Thibault Formal, Stéphane Clinchant, and Vassilina
Nikoulina. 2024. Retrieval-augmented generation in
multilingual settings.
Jonathan H. Clark, Eunsol Choi, Michael Collins, Dan
Garrette, Tom Kwiatkowski, Vitaly Nikolaev, and
Jennimaria Palomaki. 2020. TyDi QA: A benchmark
for information-seeking question answering in ty-
pologically diverse languages. Transactions of the
Association for Computational Linguistics , 8:454–
470.
Common Crawl. 2021. Common crawl 2021. Web.
Accessed: 2023-12-12.
Luyu Gao, Zhuyun Dai, Panupong Pasupat, Anthony
Chen, Arun Tejasvi Chaganty, Yicheng Fan, Vincent
Zhao, Ni Lao, Hongrae Lee, Da-Cheng Juan, and
Kelvin Guu. 2023. RARR: Researching and revising
what language models say, using language models.
InProceedings of the 61st Annual Meeting of the
Association for Computational Linguistics (Volume 1:
Long Papers) , pages 16477–16508, Toronto, Canada.
Association for Computational Linguistics.
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad Al-
Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten,
Alex Vaughan, Amy Yang, Angela Fan, Anirudh
Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra,
Yuzi He, Zach Rait, Zachary DeVito, Zef Rosnbrick,
Zhaoduo Wen, Zhenyu Yang, Zhiwei Zhao, and
Zhiyu Ma. 2024. The llama 3 herd of models.Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasu-
pat, and Ming-Wei Chang. 2020. Realm: Retrieval-
augmented language model pre-training.
Zhengbao Jiang, Frank Xu, Luyu Gao, Zhiqing Sun,
Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie
Callan, and Graham Neubig. 2023. Active retrieval
augmented generation. In Proceedings of the 2023
Conference on Empirical Methods in Natural Lan-
guage Processing , pages 7969–7992, Singapore. As-
sociation for Computational Linguistics.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Al-
berti, Danielle Epstein, Illia Polosukhin, Jacob De-
vlin, Kenton Lee, Kristina Toutanova, Llion Jones,
Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai,
Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019a.
Natural Questions: A Benchmark for Question An-
swering Research. Transactions of the Association
for Computational Linguistics , 7:453–466.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Al-
berti, Danielle Epstein, Illia Polosukhin, Jacob De-
vlin, Kenton Lee, Kristina Toutanova, Llion Jones,
Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai,
Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019b.
Natural questions: A benchmark for question an-
swering research. Transactions of the Association
for Computational Linguistics , 7:452–466.
Patrick Lewis, Barlas Oguz, Ruty Rinott, Sebastian
Riedel, and Holger Schwenk. 2020a. MLQA: Eval-
uating cross-lingual extractive question answering.
InProceedings of the 58th Annual Meeting of the
Association for Computational Linguistics , pages
7315–7330, Online. Association for Computational
Linguistics.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, et al. 2020b. Retrieval-augmented genera-
tion for knowledge-intensive nlp tasks. Advances
in Neural Information Processing Systems , 33:9459–
9474.
Bryan Li, Samar Haider, and Chris Callison-Burch.
2024. This land is Your, My land: Evaluating geopo-
litical bias in language models through territorial
disputes. In Proceedings of the 2024 Conference of
the North American Chapter of the Association for
Computational Linguistics: Human Language Tech-
nologies (Volume 1: Long Papers) , pages 3855–3871,
Mexico City, Mexico. Association for Computational
Linguistics.
Daliang Li, Ankit Singh Rawat, Manzil Zaheer, Xin
Wang, Michal Lukasik, Andreas Veit, Felix Yu, and
Sanjiv Kumar. 2023. Large language models with
controllable working memory. In Findings of the As-
sociation for Computational Linguistics: ACL 2023 ,
pages 1774–1793, Toronto, Canada. Association for
Computational Linguistics.

Shayne Longpre, Yi Lu, and Joachim Daiber. 2021.
Mkqa: A linguistically diverse benchmark for multi-
lingual open domain question answering.
Jacob Menick, Maja Trebacz, Vladimir Mikulik,
John Aslanides, Francis Song, Martin Chadwick,
Mia Glaese, Susannah Young, Lucy Campbell-
Gillingham, Geoffrey Irving, and Nat McAleese.
2022. Teaching language models to support answers
with verified quotes.
OpenAI. 2023. Gpt-4 technical report.
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and
Percy Liang. 2016. SQuAD: 100,000+ questions for
machine comprehension of text. In Proceedings of
the 2016 Conference on Empirical Methods in Natu-
ral Language Processing , pages 2383–2392, Austin,
Texas. Association for Computational Linguistics.
Leonardo Ranaldi, Barry Haddow, and Alexandra Birch.
2025. Multilingual retrieval-augmented generation
for knowledge-intensive task.
Leonardo Ranaldi, Giulia Pucci, Barry Haddow, and
Alexandra Birch. 2024a. Empowering multi-step rea-
soning across languages via program-aided language
models. In Proceedings of the 2024 Conference on
Empirical Methods in Natural Language Processing ,
pages 12171–12187, Miami, Florida, USA. Associa-
tion for Computational Linguistics.
Leonardo Ranaldi, Marco Valentino, and Andrè Fre-
itas. 2024b. Eliciting critical reasoning in retrieval-
augmented language models via contrastive explana-
tions.
Kunal Sawarkar, Abhilasha Mangal, and Shivam Raj
Solanki. 2024. Blended rag: Improving rag
(retriever-augmented generation) accuracy with se-
mantic search and hybrid query-based retrievers.
Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta
Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola
Cancedda, and Thomas Scialom. 2023. Toolformer:
Language models can teach themselves to use tools.
arXiv preprint arXiv:2302.04761 .
Nikhil Sharma, Kenton Murray, and Ziang Xiao. 2024.
Faux polyglot: A study on information disparity in
multilingual large language models.
Freda Shi, Xinyun Chen, Kanishka Misra, Nathan
Scales, David Dohan, Ed Chi, Nathanael Schärli,
and Denny Zhou. 2023. Large language models can
be easily distracted by irrelevant context.
Shamane Siriwardhana, Rivindu Weerasekera, Elliott
Wen, Tharindu Kaluarachchi, Rajib Rana, and
Suranga Nanayakkara. 2023. Improving the domain
adaptation of retrieval augmented generation (RAG)
models for open domain question answering. Trans-
actions of the Association for Computational Lin-
guistics , 11:1–17.Yuan Xia, Jingbo Zhou, Zhenhui Shi, Jun Chen, and
Haifeng Huang. 2024. Improving retrieval aug-
mented language model with self-reasoning.
Ori Yoran, Tomer Wolfson, Ori Ram, and Jonathan
Berant. 2024. Making retrieval-augmented language
models robust to irrelevant context.
Tianjun Zhang, Shishir G. Patil, Naman Jain, Sheng
Shen, Matei Zaharia, Ion Stoica, and Joseph E. Gon-
zalez. 2024. Raft: Adapting language model to do-
main specific rag.
Xinyu Zhang, Nandan Thakur, Odunayo Ogundepo,
Ehsan Kamalloo, David Alfonso-Hermelo, Xi-
aoguang Li, Qun Liu, Mehdi Rezagholizadeh, and
Jimmy Lin. 2022. Making a miracl: Multilingual in-
formation retrieval across a continuum of languages.
Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu,
Tingchen Fu, Xinting Huang, Enbo Zhao, Yu Zhang,
Yulong Chen, Longyue Wang, Anh Tuan Luu, Wei
Bi, Freda Shi, and Shuming Shi. 2023. Siren’s song
in the ai ocean: A survey on hallucination in large
language models.
Yuetong Zhao, Hongyu Cao, Xianyu Zhao, and Zhijian
Ou. 2024. An empirical study of retrieval augmented
generation with chain-of-thought.

A Prompting Approaches
Baseline Prompt Template (no RAG)
#Role
Please answer the question by following the provided instructions.
#Instructions:
Answer the question as clearly as possible based on your knowledge following the format “#Answer: ”
Note: answer in the query language .
#Question:
{question }
Table 4: Baseline prompting template.
Baseline RAG Prompt Template
#Role
Please answer the question by following the provided instructions.
#Instructions:
Answer the question as clearly as possible using the provided Reference Evidence and follow the format
“#Answer: ”Note: answer in the query language .
#Reference Evidence:
[1]{Document 1}
[2]{Document 2}
[3]{Document 3}
[4]{Document 4}
[5]{Document 5}
#Question:
{question }
Table 5: RAG prompting example.

BD-RAG prompting Template
D-RAG Prompt
#Role
You are helpful assistant. Please answer the question by following the provided instructions.
#Requirements:
Answer the question as clearly as possible using the provided #Reference Evidence and follow the #Instruc-
tions .
#Reference Evidence
[1]{Document 1}
[2]{Document 2}
[3]{Document 3}
[4]{Document 4}
[5]{Document 5}
#Instructions
1)Consider the provided documents labelled “ #Reference Evidence ”, identify and understand the main
points. Follow the directions in detail and use only the information in the documents, exemplifying which
points are most relevant for answering the question #Question .
Note: Ensure all documents are considered and provide a precise and well-structured response using
English as the shared language. Name this passage “ #Extraction: ”.
2)For each document, extract the most relevant information for answering the #Question discussing
whether they are actually relevant or irrelevant.
To ensure clarity, include the exact passages from each supporting document and reference their document
numbers. Organise your argumentation as follows:" Document [1] claims [specific argument], whereas
passage [4] claims... . Name this passage as “ #Explaination:” .
3)Please consider the step 2)in detail, ensure they are correct. Then, provide a single argumentative
explanation that considers the passages and their supporting motivations from a neutral perspective, as
concern argumentative passages.
Note: To enhance clarity, present your detailed explanation under the heading “ #Dialectic Argumenta-
tion: ”
4)Finally, to facilitate the final evaluation, deliver a short-form answer by labelling it as “ #Answer: ”
Note: answer in the query language .
#Question
{question }
Table 6: The Dialectic RAG ( D-RAG ) framework instructs the model to deliver multi-step reasoning paths that
lead the models to solve the task by explaining the perspectives that have emerged.

C Data Composition
In our experiments, we use three knowledge-
intensive question-answering task: (i)MLQA
(Lewis et al., 2020a), (ii)MKQA (Longpre et al.,
2021) and (iii)XOR-TyDi QA (Asai et al., 2021)
as they best represent multilingual open-ended
question-answering tasks. MLQA is manually
translated from SQuAD v1.1 (Rajpurkar et al.,
2016), MKQA and XOR-TyDi QA are machine
translated and manually controlled by Natural
Questions (Kwiatkowski et al., 2019b) and TyDi
QA (Clark et al., 2020), respectively.
We use test sets in the languages in Table 15. For
each language, we used the same questions and,
consequently, the same number of questions to
avoid any imbalance in double-checking by retriev-
ing the corresponding ids. Details on the number
of instances are in Table 7. In addition, since the
experimental setting of our work requires a subset
of examples to conduct the annotation phase (§2.5),
we used instances defined in Table 8 (not present in
the evaluation set) and annotated them as described
in Appendix D.
D Data Annotation
We use D-RAG annotations to fine-tune smaller
models to leverage knowledge-intensive tasks us-
ing retrieved documents (§2.6). To ensure the qual-
ity of the annotations firstly, we use an exact-match
as the first filter then we use GPT-4o-mini as anno-
tator. HThen, after ensuring that the final answer
matches the target, we systematically instruct the
GPT-4o-mini using the D-RAG (Table 6). This
double-check assess the accuracy of the outcomes
delivered. Hence, we prompt the model as follows:
#Role:
You are an experienced expert skilled in answer-
ing complex problems through logical reasoning and
structured analysis.
#Task:
Given the following sentences, you are a decision
maker who decides whether the ‘Response’ provides
the ‘Target’ as the final outcome and follows the given
‘Instructions‘. If the output doesn’t align with the tar-
get answer and doesn’t not follow the instructions,
respond with ’0’, whereas if it’s correct, then respond
with ’1’. Please, ensure that all criteria are complied
with the requests and do not provide any other answer
beyond ‘0’ or ‘1’.
#Senteces:
#Response: {model_result}
#Target: {target_answer}.
#Instructions: {D-RAG_template}.E Splitting Informations
As described in §2.5.2 and detailed in Appendix
C, we conducted an evaluation phase on equally
distributed portions of the data on the analysed
languages shown in Table 7. In addition, we anno-
tated a set of samples (Table 8) equally distributed
among the languages in Table 9. The annotation
data were filtered separately, and although some
questions are repeated for different languages (by
task and dataset construction), the arguments are
different because the documents retrieved are dif-
ferent.
Testing Sets
Dataset # per lang # per lang #Tot. #Tot.
available used used ablation
MLQA 1.5k 0.8k 7.2k 1.8k
MKQA 2k 1.0k 6.0k 1.0k
XOR-TyDi 0.6k 0.4k 2.4k 0.6k
Table 7: Number (#) of instances for evaluation (test/ab-
lation) phases which are equally distributed among the
languages in Table 15. ( kdenotes 1000 instances)
Training Sets
Dataset #example #example #Total
correct used
MLQA 3500 1920 1920
MKQA 2000 1128 920
XOR-TyDi 800 556 200
Total 6.3k 3.6k 3.02k
Table 8: Number of datasets used for evaluation phases
which are equally distributed among the languages in
Table 15. ( kdenotes 1000 instances)
Language used for training
Dataset Languages
MKQA English, Spanish, German, Russian, Chi-
nese, Finnish, Arabic
MLQA English, Chinese, Arabic, German, Spanish
XORTyDi QA English, Chinese, Arabic, Finnish
Table 9: Languages annotation.
F Models Version
Model Version
GPT-4o OpenAI API (gpt-4-o)
Llama3-70B meta-llama/Meta-Llama-3-70B-Instruct
Llama3-8B meta-llama/Meta-Llama-3-8B-Instruct
Llama3-1B meta-llama/Meta-Llama-3.2-1B-Instruct
Table 10: Models versions, found on huggingface.co.
We used the configurations described in §3 in the reposi-
tories for each model *(access verified on 25 Jan 2025).

G Difference between High- and
Low-resource Languages
In this work, we define the differences between
high-resource (HR) and low-resource (LR) using
the consideration already taken in previous works
(?Ranaldi et al., 2024a). We report two tables:
Table 11 reports the language distribution of Com-
monCrawl, and Table 12 the number of documents
in the Wikipedia dump used in our work (§3).
Language Percentage
English (en) 46.3%
Russian (ru) 6.0%
German (de) 5.4%
Chinese (zh) 5.3%
French (fr) 4.4%
Japanese (ja) 4.3%
Spanish (es) 4.2%
Other 23.1%
Table 11: Language distribution of CommonCrawl
(Common Crawl, 2021).
H Documents in Wikimedia_Dump
Language Percentage
English (en) 41,488k
Russian (ru) 13,784k
German (de) 20,772k
Chinese (zh) 7,875k
Italian (it) 10,462k
French (fr) 17,813k
Japanese (ja) 6,626k
Spanish (es) 12,865k
Portuguese (pt) 5,637k
Bengali (bn) 767k
Finnish (fn) 272k
Arabic (ar) 1,050k
Thai (th) 876k
Vietnamese (vi) 2,067k
Telogu (te) 124k
Table 12: Language distribution of Wikimedia Dump
introduced in §3.
I Retrieval Details
Retrieval We use Cohere as the retrieval system
and Wikimedia_dump as the knowledge base K
for all experiments. We use Kprovided by Cohere
wikipedia-2023-11-embed-multilingual-v3 (avail-
able on huggingface). They provide individual
documents embedded with multilingual embed-
ding model Cohere_Embed_V3 (in Table 12 are
reported the dump composition). For each question
in the evaluation data, we retrieve 10 relevant doc-
uments and then filter the top-5 most relevant ones
as done in the related repository (dot score between
query embedding and document embeddings).J Ablation Argumentation Language
D-RAG is instructed to use an English argumenta-
tion (see Table 6). In this experiment, we instruct
the model to operate in Chinese, Arabic and Ger-
man and report the differences with the original
D-RAG, which is in English.
Models +D-RAG ∆DE ∆ZH∆AR
GPT-4o -2.4 -6.3 -8.6
Llama3-70B -6.8 -9.5 -12.6
Llama3-8B -8.1 -9.3 -14.6
Llama3-1B -12.8 -16.6 -18.4
Table 13: Ablation on argumentation language impacts
onD-RAG using MKQAs’ ablation set.
K Ablation Output Analysis
To control the quality of the generations, we de-
fined two different metrics: Instruction Following
(IF) and Correct Language (CL). The role of IF is
to investigate whether the models followed the in-
structions given in the prompt. The role of CL, on
the other hand, is to analyse whether the language
of the final response is the same as that of the query
(note that this requirement was well defined in the
prompt. In order to have a robust result, we con-
ducted these two analyses using GPT-4o-mini as
an instructed evaluator, using the prompt in Ap-
pendix D and avoiding the target part in the case
of IF. We computed the CL using OpenLID frame-
work (Burchell et al., 2023). For both values, we
reported the percentage of correctness (accuracy).
L Ablation number of Steps
D-RAG operates via a single instruction. To ob-
serve the impact of instruction splitting on the final
performances, we apply the same prompt shown in
Table 6 by giving the model one step at a time.
Models MKQA MLQA XoR TyDi
GPT-4o
Single Step 68.6 65.8 61.3
4 Steps 68.4 66.9 63.0
Llama3-70B
Single Step 67.0 62.9 56.2
4 Steps 67.5 63.4 56.0
Llama3-8B tuned via D-RAG
Single Step 62.4 59.8 52.1
4 Steps 63.5 60.9 53.6
Llama3-8B tuned via D-RAG
Single Step 55.9 53.2 46.4
4 Steps 57.4 55.3 48.9
Table 14: D-RAG using Single Step prompting (tra-
ditional approach) and breaking the steps into single
phases on ablation set of proposed QA tasks.

M Proposed Task
Dataset Languages #Languages
MKQA English, Spanish, German, Russian, Chinese, Finnish,
Arabic, Italian ,Korean9
MLQA English, Chinese, Arabic, German, Spanish, Hindi 6
XORTyDi QA English, Chinese, Arabic, Finnish, Korean ,Telugu 6
Table 15: Languages present in datasets used in this work. *In bold , the languages are used only for evaluation as
described in Appendix C.
N Experiment on B ORDER LINES
To investigate the impact of our D-RAG in real contexts, we used examples from the BORDER LINES (Li
et al., 2024). This resource has questions concerning disputed territories between two nations that bu-
reaucratically belong to a specific country. The questions have the form IsPlace P a territory
ofA) Country X orB) Country Y ?. These questions are in English, language XandY(are
the languages spoken in the countries). Finally, a target or controller value indicates the country that
controls the P. (in some cases, there are no defined places that we do not consider in our analysis)
To study the consistency and dialectic capabilities of our D-RAG , we selected a small set consisting of
60 instances (20 questions in English, l20 language Xand 20 Y). We then conducted a retrieval phase and
prompt GPT-4o with the questions in the specific languages and English using the prompts defined in
Appendices A and B. Then, we set the controller as Xwe estimated the percentage of times the answer
provided by the model prompted in English matched with the controller (denoted as %Agreement
English , and the percentage when the models prompted via queries in three languages matches among
them and with the controller.

OD-RAG and GPT-4o on B ORDER LINES
• QEn:Is Sixty-Four Villages East of the River a territory of A) Russia or B) Republic of China?
• QRu: vlc li <Xest~des t qetyre derevni k vostoku ot reki> territorie A) R ossii B) R espubliki Kit a?
• QZh:河东六十四村是A）俄罗斯还是B）中华民国的领土？
•Target: A)Russia (as reported in B ORDER LINES (Li et al., 2024))
Baseline
Answer to QEn:I couldn’t find any information about ’Sixty-Four Villages East of the River’.
Answer to QZh:河六十四是B)中民的土。
Answer to QRu: Pravil~ny otvet: A) R ossi.
Retrieved documents
[1] The Sixty-Four Villages East of the River were inhabited villages by Manchu, Daur, and Han, located on the left (north) bank
of the Amur River opposite Heihe and the Blagoveshchensk. Among Russian historians, the district occupied by the villages
referred to as Zazeysky [...] (from WEn)
[2] The Treaty of Aigun of 1858, signed by the Chinese Qing Dynasty and the Russian Empire, ceded the lands north of the
Amur to Russia. Manchu residents who lived on the left bank of the Amur received the right to stay and live on Russian territory.
During the Boxer Rebellion of 1900, Russia sent its troops to occupy 64 villages and forced the people to cross the Amur River
and settle in China. Currently, most of this territory belongs to the Blagoveshchensky and Tambov districts of the Amur region.
(from WRu)
[3] The ‘Treaty of Peking’ signed between the Qing Dynasty and the Russian Empire stated: ‘The land mentioned above is
open land. Wherever the Chinese live, or the Chinese occupy the fishing and hunting land, Russia is not allowed to occupy
it. The Chinese may still occupy it by fishing and hunting as usual. The Boxer Rebellion took place in 1900, and the Qing
government had no time to consider the situation in the northeast. Russia used the destruction of the Dongqing railway by the
Boxer Rebellion as an excuse to send troops [...] (from WZh)
[4] The 1900 Amur anti-Chinese pogroms were a series of ethnic killings (pogroms) and reprisals undertaken by the Russian
Empire against subjects of the Qing dynasty of various ethnicities. They took place in the Sixty-Four Villages East of the River
in the Amur region during the same time as the Boxer Rebellion. The events ultimately resulted in thousands of deaths, the loss
of residency for Chinese subjects living in the villages, and increased Russian control over the region.[...] (from WEn)
[5] The Boxer Rebellion, also known as the Boxer Uprising, was an anti-foreign, imperialist, and Christian uprising in North
China between 1899-1901, towards the end of the Qing dynasty, by the Society of Righteous and [...] (from WEn)
RAG
Answer to QEn:The Sixty-Four Villages East of the River The villages are currently a territory of A) Russia .
Answer to QZh:河东六十四村历史上属于清朝（中华民国前身）领土，根据《北京条约》规定，俄罗斯不得占据中
国居民捕鱼狩猎的土地。[...]尽管该地区现由俄罗斯实际控制，但从历史和法理来看，应属中华民国领土。
答案：B）中华民国(B) China)
Answer to QRu: Xest~des t qetyre derevni k vostoku ot reki st ali qast~ R ossii po Agunsk omu (1858)
i Pekinsk omu (1860) dogovoram. V 1900 go du sobyti vosst ani ihtu ane u silili rossiski k ontrol~.
Sego dn t a territori vh o dit v sost av Amursk o oblasti R ossii. Otvet: A) R ossi. (A) Russia)
D-RAG
#Extraction:
The Sixty-Four Villages East of the River were historically inhabited by Manchu, Daur, and Han peoples, located on the bank of
the Amur and east of the Zeya (Ref [1],[3]). These villages became a contested region following treaties and military actions
between China and Russia. Key points include:
1. Territorial Cession via Treaties: [...] (Ref [2], [3]).
2. Russian Control Post-1900 [...] (References [3], [4]).
3. Modern Status: region currently controlled by Russia (References [2], [3]).
#Explaination:
Doc [1]: Describes the geographic and ethnic composition of the villages and emphasizes their location and their original
habitation by Chinese citizens. Relevant information as it establishes the historical context of the region.
Doc [2]: States that the Aigun Treaty ceded lands north of the Amur to Russia but allowed Chinese residents to remain and
mentions Russian actions during the Boxer Rebellion. Relevant Information as it explains the historical transition.
Doc [3]: Adds detail about the Beijing Treaty, confirming Chinese residents retained rights to the land even after it was ceded.
It highlights that Russian forces took advantage of the Qing’s preoccupation with the Boxer Rebellion to fully seize control.
Relevant Information to the question.
Doc [4]: Focuses on the anti-Chinese pogroms during the rebellion, detailing Russian motivations for taking control of the region.
Relevant Information as it corroborates the transition of control through violence and expulsion.
Doc [5]: Provides context on the Boxer Rebellion but is less directly relevant to the territorial question. It helps explain the
instability that allowed Russian actions. Partially relevant information.
#Dialectic Argumentation:
The Sixty-Four Villages East of the River historically belonged to China, as evidenced by treaties such as the Aigun and Beijing
agreements, which ceded land to Russia but retained Chinese habitation rights. While the land was historically connected to Qing
China, its effective governance shifted to Russia after 1900, and it remains under Russian control today as part of Amur Oblast.
Historically, the territory transitioned from Chinese habitation to Russian control due to military force and treaty reinterpretations.
This aligns with contemporary boundaries where the area is recognized as Russian.
#Answer:
A)俄罗斯
A) R ossi.
A) Russia
Table 16: D-RAG inference compared to other approaches on B ORDER LINES (Li et al., 2024).

P Example Outputs MKQA
Question: ¿quién escribió variaciones de Campanita del lugar?
Target: Wolfgang Amadeus Mozart.
Llama-3-8B (baseline no RAG) Respuesta: Ludwig van Beethoven.
Llama-3-8B (RAG)
The answer is Wolfgang Amadeus Mozart.
D-RAG
#Extraction:
The question asks who wrote variations of Campanita del lugar, which is the Spanish version of Twinkle, Twinkle, Little Star.
#Explaination:
Document [1] states that Wolfgang Amadeus Mozart composed Twelve Variations on "Ah vous dirai-je, Maman", a French folk
song, which is the same melody used for Campanita del lugar.
Document [2] mentions that the melody of Campanita del lugar is based on Mozart’s music, but it does not explicitly state that
Mozart wrote variations of Campanita del lugar.
Document [5] explains that "Twinkle, Twinkle, Little Star" is sung to the tune of the French melody "Ah! vous dirai-je, maman",
which was first published in 1761 and later arranged by several composers, including Mozart with Twelve Variations on "Ah
vous dirai-je, Maman".
Documents [3] and [4] are irrelevant to the question.
#Dialectic Argumentation:
Based on the provided documents, it is clear that Mozart wrote Twelve Variations on "Ah vous dirai-je, Maman", which is the
same melody used for Campanita del lugar. While Mozart did not specifically write variations of Campanita del lugar, his work
on the French folk song is the foundation of the melody used for the Spanish version.
#Answer:
Wolfgang Amadeus Mozart escribió variaciones de la melodía utilizada para Campanita del lugar, en particular Doce variaciones
sobre ’Ah vous dirai-je, Maman’.
La respuesta es Wolfgang Amadeus Mozart.
Table 17: An example of D-RAG inference compared to other approaches. While the baseline answer is wrong, the
RAG answer is correct but delivers an English text. D-RAG deliver the reasoning steps that supports the answer
and the final answer in query language.