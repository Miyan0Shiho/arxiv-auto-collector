# Derivation Prompting: A Logic-Based Method for Improving Retrieval-Augmented Generation

**Authors**: Ignacio Sastre, Guillermo Moncecchi, Aiala Rosá

**Published**: 2026-05-13 19:20:16

**PDF URL**: [https://arxiv.org/pdf/2605.14053v1](https://arxiv.org/pdf/2605.14053v1)

## Abstract
The application of Large Language Models to Question Answering has shown great promise, but important challenges such as hallucinations and erroneous reasoning arise when using these models, particularly in knowledge-intensive, domain-specific tasks. To address these issues, we introduce Derivation Prompting, a novel prompting technique for the generation step of the Retrieval-Augmented Generation framework. Inspired by logic derivations, this method involves deriving conclusions from initial hypotheses through the systematic application of predefined rules. It constructs a derivation tree that is interpretable and adds control over the generation process. We applied this method in a specific case study, significantly reducing unacceptable answers compared to traditional RAG and long-context window methods.

## Full Text


<!-- PDF content starts -->

Derivation Prompting: A Logic-Based Method for
Improving Retrieval-Augmented Generation
Ignacio Sastre, Guillermo Moncecchi, and Aiala Rosá
Instituto de Computación, Facultad de Ingeniería, Universidad de la República
Montevideo, Uruguay
{isastre,gmonce,aialar}@fing.edu.uy
Abstract.The application of Large Language Models to Question An-
swering has shown great promise, but important challenges such as hal-
lucinations and erroneous reasoning arise when using these models, par-
ticularly in knowledge-intensive, domain-specific tasks. To address these
issues, we introduce Derivation Prompting, a novel prompting technique
for the generation step of the Retrieval-Augmented Generation frame-
work. Inspired by logic derivations, this method involves deriving con-
clusions from initial hypotheses through the systematic application of
predefined rules. It constructs a derivation tree that is interpretable and
adds control over the generation process. We applied this method in a
specificcasestudy,significantlyreducingunacceptableanswerscompared
to traditional RAG and long-context window methods.1
Keywords:Large Language Models·Retrieval-Augmented Generation
·Question Answering
1 Introduction
Question Answering (QA) has improved substantially with the advent of Large
Language Models (LLMs). However, these models face important challenges,
particularly in knowledge-intensive, domain-specific tasks, such as hallucina-
tions and faulty reasoning [13,5]. The Retrieval-Augmented Generation (RAG)
framework addresses these limitations by retrieving the most relevant document
chunks from a trusted domain-specific document base, and grounding the LLM’s
generation on the retrieved information [2].
Substantial work has been done to improve the reasoning ability of LLMs [3].
Techniques like Chain-of-Thought (CoT) [15] have reliably improved perfor-
mance across various tasks, including QA. However, these techniques do not
explicitly definehowthe model should reason, as there are no restrictions on
how the intermediate reasoning steps should be constructed.
In this work, we propose Derivation Prompting, an alternative approach for
the generation step in the RAG framework inspired by logical derivations. In this
method, a conclusion is inferred from initial hypotheses by applying well-defined
1Repo with all prompts:https://github.com/nsuruguay05/derivation-promptingarXiv:2605.14053v1  [cs.CL]  13 May 2026

2 I. Sastre et al.
Fig.1.Schematic illustration of a derivation tree constructed using derivation prompt-
ing.
rules to transform and/or combine these hypotheses. This novel approach offers
some advantages over existing methods, mainly:
– Interpretability:The method not only generates a final answer but also
produces a tree structure, referred to as a derivation (see Figure 1). Each
node in the derivation represents the application of an easily interpretable
rule that transforms some of its children. This structure provides a straight-
forward way to identify errors the model could have made and to understand
how it arrived at the final answer.
– Controlled generation:By generating the answer through the sequen-
tial application of predefined rules, this method provides a clearer reasoning
path for the model to follow. This reduces hallucinations and faulty rea-
soning while ensuring that the generated answers remain grounded in the
information from the documents.
The paper is structured as follows: Section 2 presents related work. Section 3
provides a detailed explanation of Derivation Prompting. Section 4 describes the
case study conducted. Section 5 explains the evaluation method used. Section 6
presents the results and analysis. Finally, Section 7 contains the conclusions and
outlines future work.
2 Related work
Retrieval Augmented Generation
Derivation Prompting, as proposed in this work, is a prompting technique
applied within the context of Retrieval-Augmented Generation (RAG). RAG
augments LLMs with a retrieval component that recovers the most relevant
information from an external knowledge base [8,4,11].
The naive RAG paradigm consists of two main steps: retrieval and genera-
tion. First, documents are segmented into smaller units, referred to as chunks,
which are subsequently converted into vector representations. Upon a user query,
during the retrieval step the query is converted to its vector representation, and
similarity scores between the query and the indexed chunks are computed. The
topkchunks are retrieved and used as context in the prompt for the generation
step [2].
An alternative to using vector representations is to employ Cross-Encoder
models that directly process each chunk with the query and return a similarity

Derivation Prompting: A Logic-Based Method for Improving RAG 3
score [9,10]. While this approach tends to yield better results, it is significantly
less compute-efficient, as it requires as many model inferences per query as num-
ber of chucks we have, compared to only one inference when using sentence
embeddings.
Prompting techniques for enhancing reasoning
Substantial work has been done to improve the reasoning ability of LLMs [3].
Chain-of-Thought (CoT) [15] involves prompting the model to generate a coher-
ent series of intermediate reasoning steps that lead to the final answer. Few-Shot
prompting [1] is applied and a chain-of-thought is added to each example. They
show that sufficiently large LLMs can generate these reasoning chains, yielding
promising results in arithmetic, commonsense, and symbolic reasoning tasks.
The Tree of Thoughts (ToT) framework [17] is an evolution of CoT that
enables models to explore several different reasoning paths. In this approach,
reasoning is conceptualized as searching through a tree, where each node repre-
sents a thought. This framework addresses some limitations of CoT, such as the
inability to explore different continuations within the same reasoning chain or
to backtrack when incorrect conclusions are reached.
While these methods significantly improve performance on various tasks,
there is no control over how each thought is generated in the chain, as there
is no systematic methodology the model has to follow. This lack of control can
lead to erroneous reasoning and susceptibility to hallucinations, which are well-
known problems when working with LLMs [13,5].
Logic and LLMs
Similar to Derivation Prompting, some works explore combining classical
logic with prompting techniques to improve reasoning. The Logical Thoughts
(LoT) prompting framework [18] uses logical equivalence, expressing premises
in various logically equivalent forms to encourage the exploration of different
solutions. This is achieved by incorporating a verification step for each thought,
where an explanation is generated for both the thought as-is and its logical
negation. The LLM is then tasked to decide between the two.
SymbolicCoT(SymbCoT)[16]isanotherproposedmethodthatinvolvesfour
LLM modules: (i) Translator: translating premises and the question to First-
Order Logic formulas, (ii) Planner: dividing the original problem into smaller
subproblems and developing a step-by-step plan, (iii) Solver: deriving the answer
through a logical inference process and (iv) Verifier: validating the correctness
of the translations and the Solver’s output.
3 Derivation prompting
This technique focuses on the generation step of the Retrieval-Augmented Gen-
eration (RAG) framework. It relies on the premise that the expected answer to
a given query must be obtained by combining and/or transforming the most rel-
evant information extracted from a document base, since our objective is to rely

4 I. Sastre et al.
only on the information available in the documents and not on the information
the model could have learned on its training phase.
The idea for this technique is inspired by how a derivation tree in proposi-
tional logic is constructed. In this context, a conclusionφis derived from a set of
premises or hypothesesΓ={δ 1, . . . , δ n}. We denoteΓ⊢φif such a derivation
exists. The class of derivations forms an inductively defined set characterized by
a list of inference rules that explicitly state how to derive new conclusions from
existing ones. These rules operate by systematically applying logical operations
to the premises to construct a tree where each node represents an application of
a rule, culminating in the conclusion at the root [14]. Figure 2 shows an example
of a logic derivation.
Fig.2.Example of a derivation proving the statementp 1∧p2⊢p2∧p1, wherep 1and
p2are proposition symbols andE∧andI∧are the elimination and introduction rules
respectively for∧, as defined in [14].
In a typical RAG framework, documents are divided into smaller units called
chunks. Given a query, thenmost relevant chunks are selected and used as
context for generating the answer. Following the analogy with logic derivations,
in Derivation Prompting, we consider these most relevant chunks as a set of
hypotheses{h 1, . . . , h n}. The objective is to construct a derivation tree using
predefined natural language rules, ultimately deriving a conclusionc, such that
h1, . . . , h n⊢c, as depicted in Figure 1.
In contrast to logic derivations, where we usually start from a candidate
conclusion and seek to construct the proof, in this case, the conclusion is not
known beforehand. Therefore, a queryqis needed to guide the construction of
the derivation tree, with the goal that the resulting conclusion serves as the
answer to the queryq.
For each step in the construction of the derivation, the task the LLM has to
follow involves deciding which rule to apply, selecting the appropriate hypothe-
ses, and generating the conclusion that arises from the application of the chosen
rule. Although it may seem counter-intuitive to let the LLM decide which rule
to apply, this is a key aspect for making this method viable due to the LLM’s
ability to disambiguate natural language in both the rule explanation and the
hypotheses.
3.1 Rules
For Derivation Prompting, a set of derivation rules must be defined. These rules
are specified in natural language and used by the language model to construct
a derivation tree. We define a set of derivation rules that are convenient for our
use case (Section 4). It is important to notice that these rules are specific to

Derivation Prompting: A Logic-Based Method for Improving RAG 5
this problem, and any set of rules could be defined that best fits the type of
combinations and/or transformations necessary in different use cases. Table 1
presents each rule with a description and figure 3 shows a toy example for each
rule.
Table 1.List of defined rules with a brief description.
Name Description
Extract Given a hypothesish, this rule extracts a specific part ofhas a conclu-
sion.
Concat Combines two independent hypotheses to generate the conclusion.
Instantiate Generates a conclusion by instantiating a generic hypothesis into a par-
ticular case.
Compose Combines two hypotheses that share a common element to generate a
new conclusion.
Refine Given a hypothesish, it slightly adapts it to better fit the question,
without modifying the semantics or content ofh.
NoInfo This rule is used when none of the hypotheses provide information to
answer the question (or part of the question).
Fig.3.Toy examples of application for each rule. Examples (E) and (F) have informa-
tion of the query for better understanding.
3.2 Algorithm
Algorithm 1 presents the pseudo-code for constructing a derivation. When look-
ing at the algorithm in detail, it is important to notice that lines 3, 4, and 5
correspond to steps that the LLM should execute. The responsibilities of the
LLM are to decide which rule to apply and which hypotheses to use, as well as

6 I. Sastre et al.
to construct the conclusion that arises from the application of the chosen rule.
Additionally, the LLM is used to determine whether the conclusion serves as the
final answer to the user’s query.
Algorithm 1Derivation Prompting pseudo-code
Require:hypotheses_list={h 1, . . . , h n}: List of hypotheses,q: Query
1:final_answer←False
2:while notfinal_answerdo
3: Decide which rulerto apply
4: Decide which hypotheses{h i, . . . , h k}to applyrto
5:conclusion←apply rulerover{h i, . . . , h k}and queryq
6:ifconclusionis the final answerthen
7:final_answer←True
8:else
9:hypotheses_list.append(conclusion)
10:end if
11:end while
12:returnconclusion
The rest of the algorithm is straightforward. If the conclusion is considered
the final answer, the derivation is complete, and the last conclusion is used as
the answer. If not, the conclusion is added to the list of hypotheses and can be
used in subsequent rule applications (though it might potentially never be used).
Optionally, each rule application can be stored with pointers to its arguments
and conclusion to later reconstruct the derivation tree.
We explored different ways of implementing the aforementioned algorithm
and considered two main alternatives:
1.One-step prompt:This approach isolates each rule application as an inde-
pendent LLM call. Given the list of hypotheses in the middle of a derivation,
the model is prompted to produce, in a single inference, the rule to apply,
the hypotheses to use, the resulting conclusion, and whether it is the final
answer. The algorithm is implemented similarly to the Algorithm 1, with
lines 3, 4, and 5 replaced by this single call to the LLM, followed by parsing
the result.
2.Whole derivation prompt:In contrast to the previous alternative, this
approach allows the LLM to construct the entire derivation in one inference
call, effectively emulating the execution of Algorithm 1. To achieve this, we
applied a Few-Shot strategy [1], crafting six complete examples of manual
executions of the algorithm to create different derivations using all the rules
(Appendix A shows one of these examples). The model is then prompted to
follow the same steps with a new query and initial hypotheses. The result is
then parsed to obtain each rule execution and intermediate hypotheses.
In our experiments, the second approach yielded results as good as the first
one but was much faster and computationally cheaper, as it replacesnLLM
inferences with just one. Therefore, we decided to extensively use the whole
derivation approach.

Derivation Prompting: A Logic-Based Method for Improving RAG 7
4 Case study
We investigated this idea in the context of a specific use case: developing a plat-
form for question answering in the domain of administrative information for the
School of Engineering at Universidad de la República (UDELAR), for the Span-
ish language. Currently, the school operates the Orientation and Consultation
Space (OCS), where students can ask questions via email or in person, and OCS
staff provide answers. We explored the feasibility of building a tool to assist
with this work by offering an automated system for students to obtain answers
to their questions.
We gathered a small set of documents available on the school’s webpage.
Specifically, 17 websites were scraped and converted to markdown format using
LangChain’sHtml2TextTransformerclass.
For evaluation purposes, we constructed a QA dataset consisting of 135 real
user queries. These queries were derived from past emails sent to the OCS over
the last few years. Each student email was preprocessed using the Llama 2 7B
model[12]toremoveirrelevantinformationtypicalofemailcommunication(e.g.,
greetings,apologies)and,mostimportantly,personalinformationsuchasnames,
identification numbers, and phone numbers.
In the context of this project, we explored several methods using LLMs which
are explained below:
Retrieval Augmented Generation (RAG)
For the retrieval step, we explored using sentence embeddings generated
withintfloat/multilingual-e5-largeas well asBAAI/bge-reranker-large
Cross-Encoder model. Given that the use case involves fewer than a hundred
chunks,usingaCross-Encoderwasfeasibleand,asexpected,consistentlyyielded
better results than using sentence embeddings.
For the generation step, we experimented with models from the Anthropic’s
Claude 3 family2(specifically, Haiku, which is faster but less capable, and Opus,
whichisthebestperformingandcompetitivewithOpenAI’sGPT-4).Thekmost
relevant chunks to the user’s query were added as context, and a prompt was
crafted to explicitly instruct the model to use them for generating the answer.
Using Long Context Windows
Another approach we explored was leveraging the long context windows of
closed models, specifically the Claude models, which support up to 200k tokens.
In this method, we inserted all the full documents as context, thereby avoiding
the retrieval step.
Derivation Prompting
Utilizing the retrieval method described in the RAG experiment, we explored
the use of Derivation Prompting for the generation step, as detailed in Section
3. For this method, we used three initial hypotheses, corresponding to the three
most relevant chunks obtained in the retrieval step. We did not explore using
more hypotheses because of how the Few-Shot examples were designed, but this
is an area for future work that we plan to investigate.
2https://www.anthropic.com/news/claude-3-family

8 I. Sastre et al.
5 Evaluation
Evaluating Open-Domain Question Answering, especially when using LLMs, re-
mains an open problem, and human evaluation still appears to have no substi-
tute [6]. Nevertheless, it has been shown that state-of-the-art LLMs tend to ex-
hibitahighdegreeofagreementwithhumanevaluationwhenusedasjudges[19].
Therefore,wedecidedtofollowthisapproachforevaluatingeachexperimentsep-
arately. We are currently conducting human evaluation on the best performing
experiments and results will be presented in future work.
We designed an evaluation prompt following the format defined for the Feed-
back Collection dataset [7], which encompasses four components:
1.Instruction to evaluate:The instruction for the task to evaluate. In our
case, this is the particular question that the answer addresses.
2.Response to Evaluate:The response to the question that the LLM has
to evaluate (with a score on a scale from 1 to 5).
3.Reference Answer:A reference answer that corresponds to a score of 5.
4.Customized Score Rubric:Specific criteria defined for our use case, spec-
ifying what the evaluator should focus on. This includes a description of the
criteria and a detailed explanation for each possible score (1 to 5).
We defined the score rubric criteria as determining whether the generated
answer is correct and truthful. This is clearly specified in each score description.
Table 2 presents a brief explanation of each score.
Table 2.Score rubric criteria defined for each score, for evaluating generated answers.
ScoreExplanation Classification
1Candidate contradicts reference; false information. Unacceptable
2Candidate has conflicts with reference; partially false infor-
mation.Unacceptable
3Candidatedoesnotcontradictreferencebutdoesnotprovide
any information either.Acceptable
4Candidate partially matches reference; correct but incom-
plete information.Acceptable
5Candidate completely matches reference; correct and com-
plete information.Acceptable
Additionally, we classified scores 1 and 2 as unacceptable and scores 3 to 5 as
acceptable, thereby obtaining an aggregated metric for evaluation. Scores 1 and
2 correspond to answers that fully or partially contradict the reference answer
and are therefore considered unacceptable. Scores 3 to 5 may have none of the
information correct (but are not incorrect either, as no information is provided
at all), part of the information correct, or be fully correct. In all these cases, the
answers do not provide false or contradictory information and are considered
acceptable.

Derivation Prompting: A Logic-Based Method for Improving RAG 9
6 Results
The evaluation was carried out using Claude Opus as the evaluator. Table 3
shows the percentage of acceptable answers and the distribution for each score
from 1 to 5 for the best performing experiments. These experiments utilize
Claude Opus and Claude Haiku, and where applicable, the Cross-Encoder for
the retrieval step, with the number of chunks used as context set tok= 3.
Table3.Percentageofacceptableanswers,distributionofscores,averageandstandard
deviation metrics for each experiment. CH is Claude Haiku and CO is Claude Opus.
Experiment % Accep. #1#2#3#4#5AvgStd Dev
Long context - CH 65.2 3512612252.63 1.14
RAG - CH 72.6 1720742042.81 0.94
Derivation Prompting - CH 82.2 213951242.81 0.91
Long context - CO 76.3 1715732552.90 0.97
RAG - CO 77.8 1515752732.91 0.92
Derivation Prompting - CO 89.6104922543.070.79
As can be observed in Table 3, Derivation Prompting with Claude Opus sig-
nificantly reduces the number of unacceptable answers compared to the other
experiments. However, it does not necessarily increase the number of answers
with scores of 4 and 5. Many unacceptable answers from the other experiments
receive a score of 3 in Derivation Prompting. There are two primary reasons for
this: (1) The NoInfo rule has a more direct impact than simply prompting the
model not to answer questions when the information is not available, as done in
both the RAG and Long Context experiments; (2) Generating answers through
the application of explicitly defined and constrained rules reduces the likelihood
of hallucinations or misinterpretations of the context chunks and minimizes the
potential for faulty reasoning. These results suggest that while there is mini-
mal impact on recall (i.e., answering as much as possible), there is a significant
improvement in precision (i.e., avoiding incorrect answers).
It is important to note that while Derivation Prompting with Claude Haiku
also reduces unacceptable answers, it does have an impact on the number of
answers with scores of 4 and 5, resulting in fewer such answers compared to
RAG. This suggests that the size of the model is an important factor. Larger
and more powerful models, such as Claude Opus, have a better understanding
of the task of constructing the derivation and are more capable of applying the
rules effectively, yielding better results.
Although unacceptable answers have been reduced, there are still some ex-
amples that scored 1 and 2. A significant advantage of Derivation Prompting is
that the resulting derivation is interpretable, and it is easy to identify mistakes
in the application of rules. This, when compared to simple RAG, is a notable
advantage for users, as it often eliminates the need to verify answers directly

10 I. Sastre et al.
from the source. Instead, users can follow the reasoning in the derivation and
identify faulty steps. Figure 4 presents a real example of an incorrect derivation.
In the application of the Refine rule, it is clear that the model has hallucinated
facts not present in the hypotheses.
Fig.4.Exampleofanincorrectderivation(translatedfromSpanish).Intheapplication
of the Refine rule, the model hallucinates that having completed 5th year of high school
inbiologyfulfillstherequiredpre-universitystudies(hallucinationisunderlinedinred).
7 Conclusions
In this paper we introduced Derivation Prompting, a new prompting technique
inspired by logic derivations, to improve the generation step in the Retrieval-
Augmented Generation framework for open-domain question answering. Our
experiments showed that Derivation Prompting significantly reduces the occur-
rence of unacceptable answers compared to traditional RAG and long-context
window approaches.
However, the performance of Derivation Prompting is influenced by the size
and capability of the underlying LLM. While Claude Opus exhibited robust
performance, smaller models like Claude Haiku showed a decrease in useful an-
swers (though unacceptable answers were reduced), indicating the importance
of model capacity when constructing effective derivations.
Future work will focus on refining Derivation Prompting by experimenting
with different sets of rules and adjusting the number of initial hypotheses. We
are also working on formalizing the underlying formal language behind the appli-
cation of the rules, and using this to add further verification methods to ensure
the correctness of the resulting derivation.
We believe that this method can be applied to additional use cases and may
be generalized to non-RAG scenarios with different sets of rules. However, it is
important to evaluate this method further to ensure its utility in such cases.
References
1. Brown, T.B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., et al.:
Language models are few-shot learners. In: Advances in Neural Infor-

Derivation Prompting: A Logic-Based Method for Improving RAG 11
mation Processing Systems. vol. 33, pp. 1877–1901. Curran Associates,
Inc. (2020),https://proceedings.neurips.cc/paper_files/paper/2020/file/
1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf
2. Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun, J., Wang, M.,
Wang, H.: Retrieval-augmented generation for large language models: A survey
(2024),https://arxiv.org/abs/2312.10997
3. Huang, J., Chang, K.C.C.: Towards reasoning in large language models: A survey.
In: Rogers, A., Boyd-Graber, J., Okazaki, N. (eds.) Findings of the Association for
Computational Linguistics: ACL 2023. pp. 1049–1065. Association for Computa-
tional Linguistics, Toronto, Canada (Jul 2023).https://doi.org/10.18653/v1/
2023.findings-acl.67,https://aclanthology.org/2023.findings-acl.67
4. Izacard, G., Lewis, P., Lomeli, M., Hosseini, L., Petroni, F., Schick, T., Dwivedi-
Yu, J., Joulin, A., Riedel, S., Grave, E.: Atlas: few-shot learning with retrieval
augmented language models. J. Mach. Learn. Res.24(1) (mar 2024)
5. Ji,Z.,Lee,N.,Frieske,R.,Yu,T.,Su,D.,Xu,Y.,Ishii,E.,Bang,Y.J.,Madotto,A.,
Fung, P.: Survey of hallucination in natural language generation. ACM Comput.
Surv.55(12) (mar 2023).https://doi.org/10.1145/3571730,https://doi.org/
10.1145/3571730
6. Kamalloo, E., Dziri, N., Clarke, C., Rafiei, D.: Evaluating open-domain question
answering in the era of large language models. In: Rogers, A., Boyd-Graber, J.,
Okazaki, N. (eds.) Proceedings of the 61st Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers). pp. 5591–5606. Association
for Computational Linguistics, Toronto, Canada (Jul 2023).https://doi.org/10.
18653/v1/2023.acl-long.307,https://aclanthology.org/2023.acl-long.307
7. Kim, S., Shin, J., Cho, Y., Jang, J., Longpre, S., Lee, H., Yun, S., Shin, S., Kim,
S., Thorne, J., Seo, M.: Prometheus: Inducing fine-grained evaluation capability in
language models. In: The Twelfth International Conference on Learning Represen-
tations (2024),https://openreview.net/forum?id=8euJaTveKw
8. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N.,
Küttler, H., Lewis, M., Yih, W.t., Rocktäschel, T., Riedel, S., Kiela, D.:
Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. In: Ad-
vances in Neural Information Processing Systems. vol. 33, pp. 9459–9474.
Curran Associates, Inc. (2020),https://proceedings.neurips.cc/paper/2020/
hash/6b493230205f780e1bc26945df7481e5-Abstract.html
9. Nogueira, R.F., Cho, K.: Passage re-ranking with BERT. CoRRabs/1901.04085
(2019),http://arxiv.org/abs/1901.04085
10. Reimers, N., Gurevych, I.: Sentence-BERT: Sentence Embeddings using Siamese
BERT-Networks. In: Inui, K., Jiang, J., Ng, V., Wan, X. (eds.) Proceedings
of the 2019 Conference on Empirical Methods in Natural Language Process-
ing and the 9th International Joint Conference on Natural Language Process-
ing (EMNLP-IJCNLP). pp. 3982–3992. Association for Computational Linguis-
tics, Hong Kong, China (Nov 2019).https://doi.org/10.18653/v1/D19-1410,
https://aclanthology.org/D19-1410
11. Shi, W., Min, S., Yasunaga, M., Seo, M., James, R., Lewis, M., Zettlemoyer,
L., Yih, W.t.: REPLUG: Retrieval-augmented black-box language models. In:
Duh, K., Gomez, H., Bethard, S. (eds.) Proceedings of the 2024 Conference of
the North American Chapter of the Association for Computational Linguistics:
Human Language Technologies (Volume 1: Long Papers). pp. 8371–8384. Asso-
ciation for Computational Linguistics, Mexico City, Mexico (Jun 2024),https:
//aclanthology.org/2024.naacl-long.463

12 I. Sastre et al.
12. Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., et al.: Llama 2:
Open Foundation and Fine-Tuned Chat Models (Jul 2023).https://doi.org/10.
48550/arXiv.2307.09288,http://arxiv.org/abs/2307.09288, arXiv:2307.09288
[cs]
13. Valmeekam, K., Olmo, A., Sreedharan, S., Kambhampati, S.: Large language mod-
els still can’t plan (a benchmark for LLMs on planning and reasoning about
change). In: NeurIPS 2022 Foundation Models for Decision Making Workshop
(2022),https://openreview.net/forum?id=wUU-7XTL5XO
14. Van Dalen, D.: Logic and Structure. Universitext, Springer, London (2013).
https://doi.org/10.1007/978-1-4471-4558-5,https://link.springer.com/
10.1007/978-1-4471-4558-5
15. Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E.H., Le,
Q.V., Zhou, D.: Chain-of-thought prompting elicits reasoning in large language
models.In:Proceedingsofthe36thInternationalConferenceonNeuralInformation
Processing Systems. NIPS ’22, Curran Associates Inc., Red Hook, NY, USA (2024)
16. Xu, J., Fei, H., Pan, L., Liu, Q., Lee, M.L., Hsu, W.: Faithful logical reasoning via
symbolic chain-of-thought (2024),https://arxiv.org/abs/2405.18357
17. Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T.L., Cao, Y., Narasimhan,
K.R.: Tree of thoughts: Deliberate problem solving with large language models.
In: Thirty-seventh Conference on Neural Information Processing Systems (2023),
https://openreview.net/forum?id=5Xc1ecxO1h
18. Zhao, X., Li, M., Lu, W., Weber, C., Lee, J.H., Chu, K., Wermter, S.: Enhanc-
ing zero-shot chain-of-thought reasoning in large language models through logic.
In: Calzolari, N., Kan, M.Y., Hoste, V., Lenci, A., Sakti, S., Xue, N. (eds.) Pro-
ceedings of the 2024 Joint International Conference on Computational Linguis-
tics, Language Resources and Evaluation (LREC-COLING 2024). pp. 6144–6166.
ELRA and ICCL, Torino, Italia (May 2024),https://aclanthology.org/2024.
lrec-main.543
19. Zheng, L., Chiang, W.L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin,
Z., Li, Z., Li, D., Xing, E., Zhang, H., Gonzalez, J.E., Stoica, I.: Judging
llm-as-a-judge with mt-bench and chatbot arena. In: Oh, A., Naumann, T.,
Globerson, A., Saenko, K., Hardt, M., Levine, S. (eds.) Advances in Neural
Information Processing Systems. vol. 36, pp. 46595–46623. Curran Associates,
Inc. (2023),https://proceedings.neurips.cc/paper_files/paper/2023/file/
91f18a1287b398d378ef22505bf41832-Paper-Datasets_and_Benchmarks.pdf
A Few-shot example
Hypotheses:{retrieved chunks removed for brevity, enumerated from 1 to 3}
User question: What is the meaning of a credit and how many hours of weekly study should be
dedicated to a 13-credit course?
Extract | 2 | A credit is a measure of the dedication required for a subject. It is assumed that if a
subject has more credits, it requires more hours of dedication. | Not a final answer
New hypothesis: a.{Previous conclusion is repeated, removed for brevity}
Extract | 2 | A credit approximately equals one hour of weekly study, throughout an entire semester.
| Not a final answer
New hypothesis: b.{Previous conclusion is repeated, removed for brevity}
Instantiate | b | A credit approximately equals one hour of weekly study, throughout an entire
semester. Therefore, a subject with 13 credits implies an approximate dedication of 13 hours per
week. | Not a final answer
New hypothesis: c.{Previous conclusion is repeated, removed for brevity}
Concat | a,c | A credit is a measure of the dedication required for a subject. It is assumed that if
a subject has more credits, it requires more hours of dedication. A credit approximately equals one

Derivation Prompting: A Logic-Based Method for Improving RAG 13
hour of weekly study, throughout an entire semester. Therefore, a subject with 13 credits implies an
approximate dedication of 13 hours per week. | Final answer