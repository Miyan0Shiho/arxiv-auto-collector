# Chatbot Arena Meets Nuggets: Towards Explanations and Diagnostics in the Evaluation of LLM Responses

**Authors**: Sahel Sharifymoghaddam, Shivani Upadhyay, Nandan Thakur, Ronak Pradeep, Jimmy Lin

**Published**: 2025-04-28 17:24:36

**PDF URL**: [http://arxiv.org/pdf/2504.20006v1](http://arxiv.org/pdf/2504.20006v1)

## Abstract
Battles, or side-by-side comparisons in so called arenas that elicit human
preferences, have emerged as a popular approach to assessing the output quality
of LLMs. Recently, this idea has been extended to retrieval-augmented
generation (RAG) systems. While undoubtedly representing an advance in
evaluation, battles have at least two drawbacks, particularly in the context of
complex information-seeking queries: they are neither explanatory nor
diagnostic. Recently, the nugget evaluation methodology has emerged as a
promising approach to evaluate the quality of RAG answers. Nuggets decompose
long-form LLM-generated answers into atomic facts, highlighting important
pieces of information necessary in a "good" response. In this work, we apply
our AutoNuggetizer framework to analyze data from roughly 7K Search Arena
battles provided by LMArena in a fully automatic manner. Our results show a
significant correlation between nugget scores and human preferences, showcasing
promise in our approach to explainable and diagnostic system evaluations.

## Full Text


<!-- PDF content starts -->

Chatbot Arena Meets Nuggets: Towards Explanations and Diagnostics
in the Evaluation of LLM Responses
Sahel Sharifymoghaddam*, Shivani Upadhyay*, Nandan Thakur*,
Ronak Pradeep, Jimmy Lin
David R. Cheriton School of Computer Science,
University of Waterloo, Canada
{sahel.sharifymoghaddam, sjupadhyay, nandan.thakur
rpradeep, jimmylin@}uwaterloo.ca
Abstract
Battles, or side-by-side comparisons in so-
called arenas that elicit human preferences,
have emerged as a popular approach to assess-
ing the output quality of LLMs. Recently, this
idea has been extended to retrieval-augmented
generation (RAG) systems. While undoubt-
edly representing an advance in evaluation, bat-
tles have at least two drawbacks, particularly
in the context of complex information-seeking
queries: they are neither explanatory nor diag-
nostic. Recently, the nugget evaluation method-
ology has emerged as a promising approach to
evaluate the quality of RAG answers. Nuggets
decompose long-form LLM-generated answers
into atomic facts, highlighting important pieces
of information necessary in a “good” response.
In this work, we apply our AutoNuggetizer
framework to analyze data from roughly 7K
Search Arena battles provided by LMArena in
a fully automatic manner. Our results show a
significant correlation between nugget scores
and human preferences, showcasing promise
in our approach to explainable and diagnostic
system evaluations.
1 Introduction
The notion of “battles”, or side-by-side compar-
isons of responses from large language models
(LLMs), has become a popular method for evaluat-
ing the quality of LLMs (Zheng et al., 2023; Chiang
et al., 2024). In the “arena” setup, users are shown
two LLM outputs and asked to indicate which one
they prefer. This approach was popularized by
LMSYS through MT-Bench (Zheng et al., 2023)
and later expanded into the Chatbot Arena (Chi-
ang et al., 2024). The popularity of these arenas
has made them a key marketing tool when launch-
ing new LLMs from companies such as Google,
OpenAI, and Meta, who regularly tout leaderboard
rankings on Chatbot Arena in model releases. In-
spired by this increased popularity, arena-based
*Equal Contributionevaluations have been extended to a variety of do-
mains, including AI agents (Yekollu et al., 2024),
vision and image generation (Lu et al., 2024; Jiang
et al., 2024), and multilingual generation (Thakur
et al., 2025a).
Most recently, the idea of battles was ex-
tended to search-augmented LLMs in the Search
Arena (Miroyan et al., 2025). Unlike the original
setup, which focused on “closed-book” generation
of responses by LLMs directly, Search Arena eval-
uates systems that apply retrieval-augmented gener-
ation (RAG) to retrieve relevant source documents,
which are then used by LLMs to generate long-
form answers with citations (Pradeep et al., 2025a;
Han et al., 2024). While such side-by-side compar-
isons enable the evaluation of search-augmented
LLM-based systems at scale, we see them having at
least two drawbacks: they are neither explanatory
nor diagnostic, especially in scenarios where de-
termining the better answer is not straightforward.
It would be desirable for an evaluation to (at least
attempt to) explain whya user might have preferred
one response over another. Furthermore, we argue
that evaluations should be diagnostic in providing
actionable guidance on how to improve systems.
We hypothesize that the recently introduced
nugget evaluation methodology (Pradeep et al.,
2024, 2025b) can be adapted to potentially address
these two drawbacks for complex information-
seeking queries. The basic idea is to assess an-
swer quality in terms of the recall of information
nuggets, or atomic facts, that should be present in
good responses. In our previous work, we have
shown that this can be accomplished in a fully au-
tomatic manner using LLMs.
In this paper, we adapt the AutoNuggetizer
(Pradeep et al., 2024) implementation of the nugget
evaluation methodology to analyze recently re-
leased public data from Search Arena in a fully
automatic manner (see Figure 1). We find that hu-
man preferences correlate well with the distributionarXiv:2504.20006v1  [cs.IR]  28 Apr 2025

Originated in early
2024, resurged in late
2024The "Low Taper Fade" meme,
which began in early 2024,
experienced a resurgence in late
2024. [...] This remark led to a
new wave of memes humorously
depicting Ninja as being out of
touch with current trends. As of
March 2025, the meme continues
to be referenced, indicating its
lasting presence in internet
culture.The low taper fade n
meme remains
massive in 2025
GPT4.1The low taper fade meme is
indeed still massive in 2025,
having evolved and persisted in
popularity since its origins in early
2024. [...] Overall, the low taper
fade meme has demonstrated
unusual longetivity in the fast-
paced world of internet trends,
remaining a significant part of
online culture well into 2025. is the low taper fade meme still massive?
Answer Reponse (model A) Answer Reponse (model B)Widely referenced on
Tiktok, Instagram and
Youtube
Persistence of the
meme is itself a
running jokevital
vital
okay
(1) Nugget GenerationGPT4.1
(2) Nugget AssignmentokayUser QueryThe "Low Taper Fade" meme,
which began in early 2024,
experienced a resurgence in
late 2024. [...] This remark led
to a new wave of memes
humorously depicting Ninja as
being out of touch with current
trends. As of March 2025, the
meme continues to
be referenced, indicating its
lasting presence in internet
culture.Answer Reponses (model A & B)Nuggets model A model B
#1 (vi.) support support
#2 (vi.)no
supportsupport
#3 (ok.) supportpartial
support
#4 (ok.)no
supportsupport+
(3) Nugget
ScoringmodelA
modelBall score  =
0.5
all score =
0.875
Outcome:
ModelB
Search Arena (Test) InstanceFigure 1: An end-to-end example from Search Arena illustrating both nugget generation and assignment. First,
GPT 4.1generates nuggets based on the query and the responses from both models. Each nugget is then labeled with
an importance level—either “vital” or “okay”. Next, GPT 4.1evaluates whether each model supports each nugget,
assigning one of three labels: “support”, “partial support”, or “no support”. Finally, these support judgments are
scored and aggregated to determine the overall outcome (the model with the higher score is preferred).
of nugget scores, conditioned on these preferences,
which can be plotted as density functions. Further
analyses reveal that these distributions differ signifi-
cantly from one another, providing strong evidence
for the explanatory power of nugget scores in cap-
turing human preferences. From here, nugget score
differences provide actionable guidance to improve
RAG systems.
2 Related Work
Nugget-based evaluation. Originally introduced
in the TREC QA Track in 2003 (V oorhees,
2003a,b), the nugget-based evaluation methodol-
ogy focuses on identifying essential atomic facts—
called nuggets—that are relevant to a given ques-
tion. This methodology was later extended to
tasks like summarization and broader conceptions
of question answering (Nenkova and Passonneau,
2004; Lin and Demner-Fushman, 2006b; Dang and
Lin, 2007; Lin and Zhang, 2007), and researchers
have explored automation to improve its scalabil-
ity (Lin and Demner-Fushman, 2005, 2006a; Pavlu
et al., 2012).
The recent emergence of large language models
(LLMs) has enabled automated, reliable nugget-
based evaluation (Pradeep et al., 2024; Alaofi et al.,
2024; Pradeep et al., 2025b; Thakur et al., 2025b;
Abbasiantaeb et al., 2025). Several RAG evalua-
tion frameworks—such as FactScore (Min et al.,
2023), RUBRIC (Farzi and Dietz, 2024), and oth-
ers (Arabzadeh and Clarke, 2024; Mayfield et al.,
2024)—incorporate the nugget concept, although
most of these proposed approaches are either not
validated or primarily validated on traditional ad
hoc retrieval, and hence their applicability to long-
form answers is unclear. We refer readers to
Pradeep et al. (2025b) for a more detailed dis-
cussion of related work. Here, we use the Auto-Nuggetizer framework from Pradeep et al. (2024),
applying it to side-by-side comparisons of LLM
responses within the Search Arena.
Related arena benchmarks. Search Arena, in-
troduced by LMArena (Miroyan et al., 2025), is a
popular benchmark focused on evaluation of LLMs
with access to a search tool. Other notable efforts
include the MTEB Arena ( Hugging Face , 2023),
which extends the Massive Text Embedding Bench-
mark (MTEB) framework (Muennighoff et al.,
2023) to head-to-head evaluation across embed-
ding models, and Ragnarök (Pradeep et al., 2025a),
which released a new variant of the MS MARCO
collection (V2.1) and offers a framework for evalu-
ation in the TREC 2024 RAG Track.
3 Experimental Design
Search Arena overview. The Search Arena, in-
troduced by LMArena in Miroyan et al. (2025),
is a crowd-sourced evaluation platform for search-
augmented LLM systems, evaluated in terms of
side-by-side comparisons that solicit human pref-
erences (Chiang et al., 2024). The V1 version of
the publicly released dataset1contains in total 7K
samples for which two RAG focused systems (e.g.,
Gemini-2.5-Pro-Grounding vs. Perplexity-Sonar-
Reasoning-Pro) battle each other. For each battle, a
human assessor judges with one of four responses,
whether model A(or model B) is the winner or a tie
where both are responses are good (or bad).
The Search Arena dataset contains both single-
turn and multi-turn battles. In this work, we focus
exclusively on single-turn battles, evaluating 5,103
instances. We excluded multi-turn battles from our
experiments because human votes at the overall
battle level do not reliably reflect turn-level (per-
1huggingface.co/datasets/lmarena-ai/search-arena-v1-7k

query) preferences, which is what AutoNuggetizer
evaluates. Search Arena also contains battles for
several non-English languages, e.g., Chinese or
Russian. Non-English languages collectively ac-
count for less than 40% of the dataset, with English
comprising the remaining majority. Detailed statis-
tics for single-turn battles used in this work are pre-
sented in Appendix A.1. Queries in Search Arena
are diverse; they can be very long (a code snippet),
reasoning-intensive, ambiguous, and occasionally
vague. We show some examples of queries from
the dataset in Appendix A.2.
Nugget evaluations. Nugget generation creates
atomic facts that highlight the essential information
required in a RAG answer. Following Pradeep
et al. (2024), we use the AutoNuggetizer tool in
thenuggetizer code repository2to generate and
assign information nuggets to model responses. As
shown in Figure 1, there are two steps in nugget
generation and assignment:
1.Nugget generation: For each prompt extracted
from the dataset, we construct a request to Auto-
Nuggetizer consisting of a query (the prompt
itself) and two documents, which are responses
from each model, randomly ordered to avoid po-
sitional bias. The tool identifies nuggets that are
relevant to the query from the provided LLM re-
sponses. Furthermore, each nugget is assigned
an importance label: “vital” or “okay”, reflect-
ing its relevance or significance to the query.
Following previous guidelines, “vital” nuggets
are those that must be present in a “good” an-
swer, while “okay” nuggets are nice to have,
but are not absolutely necessary. Nugget impor-
tance labeling is run in a separate pass, indepen-
dent of the actual responses.
2.Nugget assignment: Once nuggets and their
importance labels are generated (from the previ-
ous step), we then use AutoNuggetizer to assign
them to model responses, determining whether
the nugget is found in the answer. This step
categorizes each nugget into “supported”, “par-
tially supported”, or “not supported”. Among
the four combinations of evaluation methods—
“vital” vs. “all” (vital + okay) and “strict” vs.
“non-strict” (full + partial support)—we adopt
the “All Score” metric, which achieves the high-
est recall by counting nuggets of all importance
2https://github.com/castorini/nuggetizer
Figure 2: Empirical probability density function (PDF)
of nugget score differences ( scoreB−scoreA) grouped
by human vote category: model Awins, tie, or model B
wins. A Kernel Density Estimation (KDE) with a band-
width of 0.5 is fitted separately for each group.
and support levels. We find that the “Strict Vital”
metric, which is the primary metric used in the
TREC 2024 RAG Track (Pradeep et al., 2025b),
is too strict for our use case, particularly when
only a small number of nuggets are available.
We emphasize that while the AutoNuggetizer
framework supports different degrees of manual
intervention, here we are running the entire evalua-
tion pipeline end-to-end in an automatic manner.
4 Experimental Results
All experiments in this paper are conducted using
GPT 4.1, the latest language model from OpenAI, as
the underlying model used by the AutoNuggetizer
via Microsoft Azure. Out of the 5,103 single-turn
battles in the Search Arena dataset, 51 were ex-
cluded from our analysis due to issues such as
Azure content filtering, invalid output formats, or
other nugget generation failures.
Figure 2 presents our main results, the probabil-
ity densities of nugget score differences ( scoreB
−scoreA) conditioned on the human preference
judgment (i.e., the battle outcomes). On the top,
we show the distribution when model Awins; on
the bottom, we show the distribution when model B
wins; and in the middle, ties. Battles where the
output of both models is considered to be equally
bad are excluded from the distributions.
These results appear to support our hypothesis
that nugget scores correlate with human prefer-

Figure 3: Empirical cumulative distribution functions (CDFs) comparing nugget score differences ( scoreB−
scoreA) across human vote categories. Each subplot shows a Kolmogorov-Smirnov (K-S) test between two groups:
(left) model Awins vs. model Bwins, (center) model Awins vs. tie, and (right) model Bwins vs. tie. The K-S statistic
and corresponding p-value are annotated in each plot, quantifying the distributional differences between groups.
ences. In the case where model Awins (top row),
the distribution skews to the left (negative values),
indicating that model Atypically gets higher nugget
scores than model B. Conversely, when model B
wins (bottom row), the distribution skews to the
right (positive values), suggesting that model Bgen-
erally obtains higher nugget score. When the hu-
man indicates a tie (middle row), the distribution
peaks around zero, as expected, indicating similar
nugget scores between models.
To analyze the statistical differences among
these three conditional distributions, we performed
pairwise Kolmogorov-Smirnov (K-S) tests. As
shown in Figure 3, the K-S statistic values range
from 0.211 to 0.324, with p-values of 1.3e−25or
lower, indicating that all three distributions differ
significantly from one another (i.e., we have high
confidence that these samples were drawn from
different distributions). These findings validate
our hypothesis that nugget score differences align
with human preferences, reinforcing the potential
of nugget-based metrics as reliable evaluators of
model quality in head-to-head evaluations.
Figure 4 presents a confusion matrix that vi-
sually compares the distribution of human pref-
erences (rows) in Search Arena against “nugget
preferences” (columns). For “nugget preference”,
we use a threshold of 0.1, meaning that when the
nugget score difference for the two model outputs
falls within ±0.1, the comparison is considered a
tie. The diagonal cells in the confusion matrix re-
veal the instances where nugget preferences align
with human preferences. Conversely, off-diagonal
cells illustrate the types and frequencies of disagree-
ments between the human and nugget scores.
In particular, the nugget-based evaluation prefers
model Ain 933 out of 1704 (54.8%) of the battles
Figure 4: Confusion matrix comparing human and
nugget preferences. Threshold of 0.1 is used to treat the
nugget preference as a tie.
where model Awins the battle (first row in Fig-
ure 4). Similarly, model Bis preferred in 911 out
of 1740 (52.4%) battles where it wins the battle
(third row in Figure 4). Lastly, when a tie occurs,
roughly similar preferences are assigned to all three
choices (second row in Figure 4). We investigate
the anti-diagonal corners where nugget and human
preferences disagree in the studies below.
4.1 Query Classification Analysis
In this analysis we use query classification to bet-
ter understand the cases where nugget preferences
and human preferences are not aligned. When the
nugget scores and the human prefer opposite sides
of a battle, we refer to this situation as a “preference
inversion”, or simply inversion.
We suspect that inversions might vary across dif-
ferent types of queries. To investigate, we followed
Rosset et al. (2024) but used the newer GPT 4.1to
rate each query on a scale of 0–10 across eight

(a) Multi-faceted
 (b) Ambiguous
 (c) Assumptive
 (d) Incompleteness
(e) Knowledge-intensive
 (f) Reasoning-intensive
 (g) Subjective
 (h) Harmful
Figure 5: Confusion matrices comparing human and nugget preferences across eight query classes from the Search
Arena dataset. Threshold of 0.1 is used to treat the nugget preference as a tie.
Language Inversion (%) Query Count
(1) Multi-faceted 19% 298
(2) Ambiguous 19% 194
(3) Assumptive 18% 28
(4) Incompleteness 18% 627
(5) Knowledge-intensive 16% 1135
(6) Reasoning-intensive 16% 283
(7) Subjective 13% 598
(8) Harmful 5% 82
Table 1: Inversion percentages and query frequencies of
eight different query classes in the Search Arena dataset.
different categories. Then, we classify each query
into its maximum scoring category or categories
(allowing for ties). To further strengthen the cate-
gory signals, we exclude queries with a maximum
score less than seven from this classification. Raw
distributions of the query ratings per category and
sample queries from each class are available in Ap-
pendix A.2.
As shown in Table 1, the portion of inversions
for multi-faceted, ambiguous, assumptive, and in-
complete queries is higher than that of subjec-
tive, knowledge-, and reasoning- intensive queries.
This suggests that inversions are more likely when
queries allow for multiple valid interpretations or
are under-specified.
We followed up by manually examining the in-
versions for these categories. As a case study, we
encountered a query categorized as ambiguous with
the text “Potatoes”. In our opinion, both model A
and model Bprovided relevant responses. However,
model Afocused on the historical aspects and nutri-Language Inversion (%) Query Count
(1) German 19% 240
(2) English 16% 3089
(3) Portuguese 16% 148
(4) Chinese 16% 324
(5) Russian 14% 459
(6) French 13% 150
(7) Others 15% 642
Table 2: Inversion percentages and query frequencies
of the six most common languages in the Search Arena
dataset.
tional value of potatoes, whereas model Bdiscussed
cooking methods and varieties. The user judge
preferred model B’s answer, while model Awas se-
lected based on the nugget score. The inherent
ambiguity of the query likely led to this inversion,
as it permitted various valid interpretations.
Overall, the knowledge-intensive class shows the
highest preference alignment—58.8% and 53.9%
for model Aand model Bwins, respectively (see Fig-
ure 5). This finding suggests that nuggetization is
most effective for research-y queries requiring re-
trieval augmentation.
4.2 Query Language Analysis
We next analyzed AutoNuggetizer effectiveness
across the six most frequent query languages, each
representing at least 3% of the dataset. Previously,
the tool had only been run on English responses,
and there are likely to be language effects in the
breakdown of inversions.
As shown in Table 2, German exhibits the high-

(a) German
 (b) English
 (c) Portuguese
(d) Chinese
 (e) Russian
 (f) French
Figure 6: Confusion matrices comparing human and nugget preferences across six different languages that each
account for at least 3% of the Search Arena dataset. Threshold of 0.1 is used to treat the nugget preference as a tie.
est inversion rate (19%), while French shows the
lowest (13%). The confusion matrix for German
(see Figure 6) reveals that it has the smallest por-
tion of ties in human preferences, leading to more
anti-diagonal disagreements. Limited human-voted
ties suggest that the LLMs participating in the
battles often differ in their ability to handle Ger-
man queries. Additionally, assuming a similar
distribution of query categories across languages,
the higher inversion rate among German queries
points to the AutoNuggetizer being less effective
in this language as well. Due to the limited dataset
size, we leave language-specific query classifica-
tion analysis for future work.
5 Discussion
In this work, we hypothesized that the nugget eval-
uation methodology can be applied to both explain
human preferences in side-by-side comparisons
and offer diagnostic guidance on how to improve
models. Our underlying intuition is quite simple:
humans prefer LLM responses that contain more
facts, operationalized in terms of atomic nuggets.
With our AutoNuggetizer framework, nugget ex-
traction and scoring can be accomplished automati-
cally. We find that differences in nugget scores are
strongly correlated with human preferences, which
can be seen in our density plots.
At a high level, we find that these results are
quite strong, given that we have only examined onefactor that might influence LLM response quality.
For example, human preferences might be affected
by how citations are presented, the fluency or or-
ganization of the responses, the presence of aids
such as tables and headings, as well as a myriad of
other factors. Nevertheless, with our automatically
computed fact recall metric, we are able to predict
human preferences over 50% of the time. This is
quite remarkable in our opinion, and potentially
points to the explanatory power of nugget scores.
Although we have only begun explorations, from
here it is possible to imagine how our approach can
be extended into providing diagnostic information
to system builders. Missing nuggets can be at-
tributed to different causes: a relevant document
that was not retrieved or the LLM ignoring relevant
context present in the prompt. Different failure
modes would lead to different courses of action.
For example, a retrieval failure could point to the
need for improvements to the embedding model,
and perhaps the battle result can be adapted into an
additional training example.
Building on these initial insights, this paper
serves as the first stage of exploring nugget evalua-
tion for search-based arena battles. In our prelimi-
nary analyses, we missed out on evaluating a few
things that we keep as future work:
First, we would like to consider an LLM-as-a-
judge evaluation and check correlations between an
LLM assessor and our nugget evaluation. Next, we

currently filter to include only single-turn conversa-
tions; in the future we would like to evaluate multi-
turn conversations whenever per-turn user votes be-
come available. Finally, battles in the Search Arena
dataset include URLs to source documents as well
as web search traces such as search queries. This
information can be valuable for generating nuggets
and assessing the factuality of LLM-generated re-
sponses. We leave as future work the exploration
of leveraging grounded URLs to automatically gen-
erate nuggets based on the retrieved content.
6 Conclusion
This work explores the use of nugget-based evalu-
ation to assess large language model (LLM) com-
petitions in Search Arena, a benchmark for side-
by-side comparisons of search-augmented model
responses. By generating and scoring atomic facts
(nuggets), we offer a more interpretable and diag-
nostic alternative to traditional human preference-
based evaluations.
Our results show strong alignment between
nugget-based preferences and human judgments,
particularly for knowledge-intensive queries. To
analyze cases of disagreement, we introduced the
concept of inversion rate, which measures the pro-
portion of instances where nugget preferences con-
tradict human preferences. Higher inversion rates
were found in multi-faceted, ambiguous, and as-
sumptive queries, suggesting these query types are
more challenging for automated evaluation. Addi-
tionally, language-level analysis reveals that Ger-
man queries have the highest inversion rate among
the major languages, pointing to potential limita-
tions in nuggetization quality for certain languages.
Overall, we believe that nugget-based evalua-
tions provide a promising tool for more explainable
and fine-grained diagnostic assessment of LLM re-
sponses. Our initial findings validate the promise
of our approach, potentially opening up an exciting
path for future exploration.
Acknowledgments
This research was supported in part by the Nat-
ural Sciences and Engineering Research Council
(NSERC) of Canada. Additional funding is pro-
vided by Microsoft via the Accelerating Foundation
Models Research program.References
Zahra Abbasiantaeb, Simon Lupart, Leif Azzopardi,
Jeffery Dalton, and Mohammad Aliannejadi. 2025.
Conversational gold: Evaluating personalized conver-
sational search system using gold nuggets. preprint
arXiv:2503.09902 .
Marwah Alaofi, Negar Arabzadeh, Charles LA Clarke,
and Mark Sanderson. 2024. Generative information
retrieval evaluation. In Information Access in the Era
of Generative AI , pages 135–159.
Negar Arabzadeh and Charles LA Clarke. 2024. A
comparison of methods for evaluating generative IR.
preprint arXiv:2404.04044 .
Wei-Lin Chiang, Lianmin Zheng, Ying Sheng, Anasta-
sios Nikolas Angelopoulos, Tianle Li, Dacheng Li,
Banghua Zhu, Hao Zhang, Michael Jordan, Joseph E
Gonzalez, and Ion Stoica. 2024. Chatbot arena: An
open platform for evaluating LLMs by human pref-
erence. In Forty-first International Conference on
Machine Learning .
Hoa Trang Dang and Jimmy Lin. 2007. Different struc-
tures for evaluating answers to complex questions:
pyramids won’t topple, and neither will human asses-
sors. In Proceedings of the 45th Annual Meeting of
the Association for Computational Linguistics (ACL
2007) , pages 768–775, Prague, Czech Republic.
Naghmeh Farzi and Laura Dietz. 2024. Pencils
down! automatic rubric-based evaluation of re-
trieve/generate systems. In Proceedings of the 2024
ACM SIGIR International Conference on Theory of
Information Retrieval , ICTIR ’24, pages 175–184,
Washington, D.C.
Rujun Han, Yuhao Zhang, Peng Qi, Yumo Xu, Jenyuan
Wang, Lan Liu, William Yang Wang, Bonan Min, and
Vittorio Castelli. 2024. RAG-QA arena: Evaluating
domain robustness for long-form retrieval augmented
question answering. In Proceedings of the 2024 Con-
ference on Empirical Methods in Natural Language
Processing , pages 4354–4374, Miami, Florida.
Dongfu Jiang, Max Ku, Tianle Li, Yuansheng Ni,
Shizhuo Sun, Rongqi Fan, and Wenhu Chen. 2024.
GenAI arena: An open evaluation platform for gener-
ative models. Advances in Neural Information Pro-
cessing Systems , 37:79889–79908.
Jimmy Lin and Dina Demner-Fushman. 2005. Automat-
ically evaluating answers to definition questions. In
Proceedings of the 2005 Human Language Technol-
ogy Conference and Conference on Empirical Meth-
ods in Natural Language Processing (HLT/EMNLP
2005) , pages 931–938, Vancouver, Canada.
Jimmy Lin and Dina Demner-Fushman. 2006a. Meth-
ods for automatically evaluating answers to complex
questions. Information Retrieval , 9(5):565–587.
Jimmy Lin and Dina Demner-Fushman. 2006b. Will
pyramids built of nuggets topple over? In Proceed-
ings of the Human Language Technology Conference

of the NAACL, Main Conference , pages 383–390,
New York, New York.
Jimmy Lin and Pengyi Zhang. 2007. Deconstructing
nuggets: the stability and reliability of complex ques-
tion answering evaluation. In Proceedings of the
30th Annual International ACM SIGIR Conference
on Research and Development in Information Re-
trieval (SIGIR 2007) , pages 327–334, Amsterdam,
the Netherlands.
Yujie Lu, Dongfu Jiang, Wenhu Chen, William Yang
Wang, Yejin Choi, and Bill Yuchen Lin. 2024.
Wildvision: Evaluating vision-language models
in the wild with human preferences. preprint
arXiv:2406.11069 .
James Mayfield, Eugene Yang, Dawn Lawrie, Sean
MacAvaney, Paul McNamee, Douglas W. Oard, Luca
Soldaini, Ian Soboroff, Orion Weller, Efsun Kayi,
Kate Sanders, Marc Mason, and Noah Hibbler. 2024.
On the evaluation of machine-generated reports. In
Proceedings of the 47th International ACM SIGIR
Conference on Research and Development in Infor-
mation Retrieval (SIGIR 2024) , pages 1904–1915,
Washington, D.C.
Hugging Face . 2023. MTEB arena. https://
huggingface.co/spaces/mteb/arena . Accessed:
2025-04-24.
Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis,
Wen-tau Yih, Pang Koh, Mohit Iyyer, Luke Zettle-
moyer, and Hannaneh Hajishirzi. 2023. FActScore:
Fine-grained atomic evaluation of factual precision
in long form text generation. In Proceedings of the
2023 Conference on Empirical Methods in Natural
Language Processing , pages 12076–12100, Singa-
pore.
Mihran Miroyan, Tsung-Han Wu, Logan Kenneth King,
Tianle Li, Anastasios N. Angelopoulos, Wei-Lin
Chiang, Narges Norouzi, and Joseph E. Gonzalez.
2025. Introducing the search arena: Evaluating
search-enabled AI. https://blog.lmarena.ai/
blog/2025/search-arena/ .
Niklas Muennighoff, Alexandre Tazi, Tom Magister,
Mohammad Shoeybi, Teven Le Scao, Dragomir
Radev, and Alex Wang. 2023. MTEB: Massive text
embedding benchmark. preprint arXiv:2302.08968 .
Ani Nenkova and Rebecca Passonneau. 2004. Evaluat-
ing content selection in summarization: The pyramid
method. In Proceedings of the Human Language
Technology Conference of the North American Chap-
ter of the Association for Computational Linguistics:
HLT-NAACL 2004 , pages 145–152, Boston, Mas-
sachusetts.
Virgil Pavlu, Shahzad Rajput, Peter B. Golbus, and
Javed A. Aslam. 2012. IR system evaluation using
nugget-based test collections. In Proceedings of the
Fifth ACM International Conference on Web Search
and Data Mining (WSDM 2012) , pages 393–402,
Seattle, Washington.Ronak Pradeep, Nandan Thakur, Sahel Sharifymoghad-
dam, Eric Zhang, Ryan Nguyen, Daniel Campos,
Nick Craswell, and Jimmy Lin. 2025a. Ragnarök: A
reusable RAG framework and baselines for TREC
2024 retrieval-augmented generation track. In Ad-
vances in Information Retrieval , pages 132–148,
Cham.
Ronak Pradeep, Nandan Thakur, Shivani Upadhyay,
Daniel Campos, Nick Craswell, and Jimmy Lin. 2024.
Initial nugget evaluation results for the TREC 2024
RAG Track with the AutoNuggetizer Framework.
preprint arXiv:2411.09607 .
Ronak Pradeep, Nandan Thakur, Shivani Upadhyay,
Daniel Campos, Nick Craswell, and Jimmy Lin.
2025b. The great nugget recall: Automating fact
extraction and RAG evaluation with large language
models. preprint arXiv:2504.15068 .
Corby Rosset, Ho-Lam Chung, Guanghui Qin, Ethan C
Chau, Zhuo Feng, Ahmed Awadallah, Jennifer
Neville, and Nikhil Rao. 2024. Researchy ques-
tions: A dataset of multi-perspective, decomposi-
tional questions for LLM web agents. preprint
arXiv:2402.17896 .
Nandan Thakur, Suleman Kazi, Ge Luo, Jimmy
Lin, and Amin Ahmad. 2025a. MIRAGE-
bench: Automatic multilingual benchmark arena for
retrieval-augmented generation systems. preprint
arXiv:2410.13716 .
Nandan Thakur, Jimmy Lin, Sam Havens, Michael
Carbin, Omar Khattab, and Andrew Drozdov. 2025b.
Freshstack: Building realistic benchmarks for eval-
uating retrieval on technical documents. preprint
arXiv:2504.13128 .
Ellen M. V oorhees. 2003a. Evaluating answers to defini-
tion questions. In Companion Volume of the Proceed-
ings of HLT-NAACL 2003 — Short Papers , pages
109–111, Edmonton, Canada.
Ellen M. V oorhees. 2003b. Overview of the TREC
2003 question answering track. In Proceedings of
the Twelfth Text REtrieval Conference (TREC 2003) ,
pages 54–68, Gaithersburg, Maryland.
Nithik Yekollu, Arth Bohra, Ashwin Chirumamilla,
Kai Wen, Sai Kolasani Wei-Lin Chiang, Anas-
tasios Angelopoulos, Joseph E. Gonzalez, Ion
Stoica, and Shishir G. Patil. 2024. Agent arena.
https://gorilla.cs.berkeley.edu/blogs/14_
agent_arena.html .
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan
Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin,
Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang,
Joseph E. Gonzalez, and Ion Stoica. 2023. Judging
LLM-as-a-judge with MT-Bench and Chatbot Arena.
InAdvances in Neural Information Processing Sys-
tems 36 (NeurIPS 2023) Datasets and Benchmarks
Track , pages 46595–46623, New Orleans, Louisiana.

A Supplemental Data
A.1 Dataset Statistics
Figure 7: Dataset Overview: (left) winners distribution;
(right) language distribution.
Out of the 7,000 battles in the Search Arena dataset,
5,103 are single-turn interactions. As shown in Fig-
ure 7, model Aand model Beach win approximately
one-third of these battles, with ties occurring in
20.5% of cases. An additional 11.4% are ties where
both responses are labeled as bad. Among the
single-turn battles, English dominates with 61.2%
of the data, followed by Russian (9.0%), Chinese
(6.4%), German (4.8%), and French (3.0%). Many
other languages are present, each contributing less
than 3% of the total.
A.2 Query Classification
Figure 8 illustrates the raw ratings distribution of
each criteria. Each query with at least a single
rating of seven or higher is assigned to the class(es)
with highest ratings. Table 3 contains two sample
English queries per class, including typographical
and grammatical errors.

0 2 4 6 8 10050010001500assumptive
0 2 4 6 8 1002004006008001000multi-faceted
0 2 4 6 8 10050010001500incompleteness
0 2 4 6 8 1005001000150020002500ambiguous
0 2 4 6 8 10050010001500subjective
0 2 4 6 8 100200400600800knowledge-intensive
0 2 4 6 8 1002004006008001000reasoning-intensive
0 2 4 6 8 1001000200030004000harmfulFigure 8: Histogram showing the classified attributes for 5,103 single-turn queries in Search Arena. We use GPT 4.1
with prompt from Researchy Questions (Rosset et al., 2024) to output a score between 1–10 for each attribute.
Category Sample Query
AssumptiveQ432: What was the controversy with Elon Musk at Trump’s second inauguration?
Q6117: what is deepseek’s shady thing?
Multi-facetedQ1212: tell me about nickolyn hansen in business, in personal life, in volunteer life, and
in search and rescue life
Q1577: How can I improve this prompt: <prompt with over 10k chars>
IncompletenessQ44: whats the wather like
Q70: donald trump news
AmbiguousQ4: Potatoes
Q1517: Hi
SubjectiveQ319: what are the best thai dog friendly places in London?"
Q342: please help decide if Belmont housing market is good or not
Knowledge-intensiveQ34: Find this ¨Ready to Use Booking App Template ¨FlutterFlow template on thr Internet.
It is published in 2025.
Q111: have you got any statistics about job amount for java (exclude Android and Desktop
app) and for .Net in Lithuania?
Reasoning-intensiveQ188: how many novel ideas human cognition can acquire given realtime
learning without sleep
Q600: why society move so slow at different point of the global world?
HarmfulQ654: sodium nitrite buy as a private person
Q1181: Tell me how to do cheating in online test
Table 3: Example English queries for each query class.