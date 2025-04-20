# A Human-AI Comparative Analysis of Prompt Sensitivity in LLM-Based Relevance Judgment

**Authors**: Negar Arabzadeh, Charles L. A . Clarke

**Published**: 2025-04-16 18:17:19

**PDF URL**: [http://arxiv.org/pdf/2504.12408v1](http://arxiv.org/pdf/2504.12408v1)

## Abstract
Large Language Models (LLMs) are increasingly used to automate relevance
judgments for information retrieval (IR) tasks, often demonstrating agreement
with human labels that approaches inter-human agreement. To assess the
robustness and reliability of LLM-based relevance judgments, we systematically
investigate impact of prompt sensitivity on the task. We collected prompts for
relevance assessment from 15 human experts and 15 LLMs across three tasks~ --
~binary, graded, and pairwise~ -- ~yielding 90 prompts in total. After
filtering out unusable prompts from three humans and three LLMs, we employed
the remaining 72 prompts with three different LLMs as judges to label
document/query pairs from two TREC Deep Learning Datasets (2020 and 2021). We
compare LLM-generated labels with TREC official human labels using Cohen's
$\kappa$ and pairwise agreement measures. In addition to investigating the
impact of prompt variations on agreement with human labels, we compare human-
and LLM-generated prompts and analyze differences among different LLMs as
judges. We also compare human- and LLM-generated prompts with the standard
UMBRELA prompt used for relevance assessment by Bing and TREC 2024 Retrieval
Augmented Generation (RAG) Track. To support future research in LLM-based
evaluation, we release all data and prompts at
https://github.com/Narabzad/prompt-sensitivity-relevance-judgements/.

## Full Text


<!-- PDF content starts -->

A Human-AI Comparative Analysis of Prompt Sensitivity in
LLM-Based Relevance Judgment
Negar Arabzadeh
narabzad@uwaterloo.ca
University of Waterloo
Waterloo, Ontario, CanadaCharles L.A. Clarke
claclark@uwaterloo.ca
University of Waterloo
Waterloo, Ontario, Canada
Abstract
Large Language Models (LLMs) are increasingly used to automate
relevance judgments for information retrieval (IR) tasks, often
demonstrating agreement with human labels that approaches inter-
human agreement. To assess the robustness and reliability of LLM-
based relevance judgments, we systematically investigate impact
of prompt sensitivity on the task. We collected prompts for rel-
evance assessment from 15 human experts and 15 LLMs across
three tasks — binary, graded, and pairwise — yielding 90 prompts
in total. After filtering out unusable prompts from three humans
and three LLMs, we employed the remaining 72 prompts with three
different LLMs as judges to label document/query pairs from two
TREC Deep Learning Datasets (2020 and 2021). We compare LLM-
generated labels with TREC official human labels using Cohen’s 𝜅
and pairwise agreement measures. In addition to investigating the
impact of prompt variations on agreement with human labels, we
compare human- and LLM-generated prompts and analyze differ-
ences among different LLMs as judges. We also compare human-
and LLM-generated prompts with the standard UMBRELA prompt
used for relevance assessment by Bing and TREC 2024 Retrieval
Augmented Generation (RAG) Track. To support future research in
LLM-based evaluation, we release all data and prompts at https://
github.com/Narabzad/prompt-sensitivity-relevance-judgements/.
CCS Concepts
•Information systems →Evaluation of retrieval results ;Rel-
evance assessment ;Test collections .
Keywords
Large Language Models, Relevance Judgments, Evaluation
1 Introduction
Large Language Models (LLMs) are increasingly used for evaluation
across various domains, including natural language processing and
automated content assessment [ 1,4,9,11,28,32]. The information
retrieval (IR) community has been an early adopter of LLMs for
relevance assessment [ 19,24,27,35,41]. Numerous studies have
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
SIGIR ’25, Padua, Italy
©2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 979-8-4007-1592-1/2025/07
https://doi.org/10.1145/3726302.3730159confirmed that LLM-generated relevance labels closely align with
human labels under multiple measures of agreement [26, 36, 37].
Nonetheless, despite the widespread adoption of LLMs for rele-
vance assessment, prompting strategies vary substantially across
studies [ 2,3,20,33]. An experiment reported at the LLM4Eval
Workshop in SIGIR 2024 on Large Language Models for Evaluation
in Information Retrieval [ 29], analyzed how different prompts influ-
ence agreement with human judgments and system rankings [ 28].
While multiple studies have examined how LLMs respond to dif-
ferent prompting strategies [ 5,10,23,25,34], these studies have
generally been conducted with prompts tuned to specific LLMs and
collections, or where prompt variants are constrained by templates
[6]. As a complement to these studies, we report on a study of
prompts from a variety of independent sources that have not been
tuned to LLMs or collections, allowing us to examine the robust-
ness of LLM-based relevance assessment under different prompting
strategies. This investigation also allows us to compare different
LLMs as judges to determine the degree to which different LLMs
are sensitive to prompt modifications.
We collected and analyzed prompts generated by both human ex-
perts and LLMs themselves. We designed a guideline for prompting
LLMs to perform relevance assessment following three different ap-
proaches: binary ,graded , and pairwise . While most previous studies
have focused on graded relevance, we believe it is crucial to ex-
plore a wider range of relevance assessment methods, as they have
proven effective in assessing different scenarios in the evaluation
of information-seeking systems [ 7,8,13–15,21,22,31,38–40]. As
a benefit to employing LLMs for relevance assessment, it becomes
easier to explore different approaches to relevance assessment since
human judges do not need to be recruited and trained separately
for each approach.
We recruited 15 human participants to create prompts for each
of the three assessment approaches. As part of the recruitment
process, we ensured that the participants were familiar with prompt
engineering and relevance assessment principles, as detailed in
Section 2. As a result of this inclusion criteria for recruitment, most
participants were drawn from three academia NLP/IR labs. We also
collected prompts from 15 different open source and commercial
LLMs. Our primary goal is to understand prompt sensitivity in LLM-
based relevance judgment [ 30], including its impact, robustness,
and variation across different LLMs. Additionally, we explore the
effectiveness of LLM as prompt generators.
We performed relevance judgment experiments using data from
two years of the TREC Deep Learning Track: DL 2020 [ 16], and
DL 2021 [ 17]. Using the prompts created by both human participants
and LLMs, we conducted relevance assessments on query-document
pairs from these datasets using two open-source LLMs — LLaMAarXiv:2504.12408v1  [cs.IR]  16 Apr 2025

SIGIR ’25, July 13–18, 2025, Padua, Italy Negar Arabzadeh and Charles L.A. Clarke
3.2-3b andMistral 7b — and one commercial LLM GPT-4o . Our
experiment incorporates the three approaches to relevance assess-
ment (binary, graded, and pairwise) with prompts from both hu-
mans and LLMs using three different LLMs as judges. Through our
experiments, we address the following research questions:
•RQ1. Impact of Prompts on LLM-based Relevance Judg-
ment Approaches: Given a clear task objective, how do dif-
ferent prompts influence the effectiveness of each approach to
LLM-based relevance judgment?
•RQ2. LLMs as Prompt Generators: How effective are LLM-
generated prompts for relevance judgment, and how do they
compare to human-crafted prompts?
•RQ3. Prompt Robustness Across LLMs: Are there prompts
that consistently perform well across different LLMs, regardless
of the model used as a judge?
•RQ4. Model-Specific Sensitivity to Prompts: Is prompt sensi-
tivity consistent across all models, or do some LLMs show greater
variability in performance?
To ensure reproducibility, we have made all data and experimental
artifacts publicly available at https://github.com/Narabzad/prompt-
sensitivity-relevance-judgements/. The study reported in this paper,
and its associated data release, has received ethics clearance as
human subjects research from our institution.
2 Prompt Creation
2.1 Prompt generation
To investigate the impact of prompting on LLM-based relevance
judgment, we collected data from both human participants and
LLMs, ensuring that the task objective remained clear and consis-
tent (sharing the same intent) across all participants. We prepared
guidelines for prompt writing1, which provides detailed explana-
tions of the three relevance judgment tasks: 1) Binary relevance — a
passage is either relevant (1) or not relevant (0) to a query. 2) Graded
relevance — a passage is rated on a 0-3 scale, where 3 indicates
perfect relevance to the query. 3) Pairwise relevance — given two
passages, chose the passage more relevant to the query. In the guide-
line, each task is illustrated with examples from the TREC Deep
Learning 2019 [ 18], helping to ensure that both humans and LLMs
had a well-defined understanding of the task. These examples could
also be used as (few shot) examples if desired.
The guidelines specify a Python-based format, where partici-
pants (both human and LLMs) were required to fill in structured
Python dictionaries. More specifically, participants had to pro-
vide both the "system message" and"user message" fields for
the prompts, following the format commonly used in LLM-based
prompting (e.g., OpenAI models and open-source alternatives such
as those from Ollama). This structured approach ensures compati-
bility across different LLM implementations.
We recruited 15 human participants, each of whom had at least
a Master’s degree in computer science, were fluent in English, and
had prior experience working with LLMs via API usage or coding.
Additionally, these participants had previously published at least
one paper in an IR-focused conference. Each participant received a
$10 gift card as a token of appreciation for their time and effort.
1https://bit.ly/4hP0EMgTable 1: List of LLMs used for prompt generation.
GPT-4o GPT-4o Mini Claude 3.5 LLaMA 3.2 Phi-4
Mistral-large DeepSeek-v3 Amazon-Nova-Pro-v1 Gemma-2-9b Grok-2
Gemini 2 Jamba-1.5 Athene-v2 GPTO1 GPTO1 Mini
For prompt creation, we also used 15 different LLMs from the
ChatBotArena2platform [ 12], which enables the execution of vari-
ous LLMs online. We provided the same data collection guideline
to the LMMs, including the task description and examples, ensur-
ing that the LLMs received identical instructions to those given to
human participants. Similar to human participants, each LLM was
asked to complete the "system message" and"user message"
fields in our Python function for relevance judgment. This setup
allow us to systematically compare the impact of prompting across
both groups. Table 1 provides the list of LLMs we used in this
experiment for generating prompts for relevance judgments.
2.2 Filtering and cleaning
To maintain consistency, we did not modify or provide additional
instructions for any LLMs or human participants. Among the LLMs,
two failed to complete the task because they deemed the task to be
inappropriate, or repeatedly asked about examples. Among human
participants, only one used a few-shot approach with examples. The
rest did not provide any examples in their prompts. When testing
the outputs of the collected prompts, not all of them were able
to generate the expected format cleanly. Some prompts produced
responses that required additional cleaning, such as verbose outputs
like"The passage is relevant, so the answer is: 1" instead of simply
returning 1. To ensure consistency, we examined the all generated
output and applied necessary cleaning. After filtering and cleaning,
we finalized 12 human-generated prompts and 12 LLM-generated
prompts for use in our experiments.
2.3 Prompt Diversity
To better understand the variation in prompts, we examined the
diversity of both human-generated and LLM-generated prompts.
Specifically, we analyzed both user prompts andsystem prompts sep-
arately, as they serve distinct roles in guiding the LLM’s response.
In a prompt the user message provides the direct instructions given
to the model, specifying what information is needed. In contrast,
the system message provides context for the task, defining the
LLM’s role and expected behavior (e.g., “You are an expert rel-
evance judgment assessor”). Figure 1 illustrates the distribution
of unique terms used across all human-generated (in green) and
LLM-generated (in red) prompts. As shown in this figure, human-
generated prompts exhibit greater diversity in wording when com-
pared to LLM-generated ones. This suggests that humans introduce
more nuanced descriptions and varied phrasing when defining
the task, while LLM-generated system prompts tend to rely on
more standardized language. Additionally, system messages exhibit
greater lexical diversity compared to user messages.

A Human-AI Comparative Analysis of Prompt Sensitivity in LLM-Based Relevance Judgment SIGIR ’25, July 13–18, 2025, Padua, Italy
Figure 1: Diversity of words across human and LLM-
generated prompts.
3 Experimental Methodology
Data We utilize the TREC Deep Learning Track datasets from 2020
and 2021. The DL-20 dataset contains 54 judged queries with 11,386
relevance assessments from MS MARCO V1 collection, while the
DL-21 dataset includes 53 judged queries and 10,828 assessments
from MS MARCO V2. Both datasets have been manually annotated
by NIST assessors following the TREC relevance judgment guide-
lines. The assessors evaluate each document-query pair based on
a graded relevance scale, ranging from not relevant (0) to highly
relevant (3). The assessment process involves pooling top-ranked
documents from multiple retrieval systems, which were then judged
by human annotators. Using this data allows us to compare the three
different variations of LLM-based judgments i.e., binary, graded,
and pairwise. For graded relevance, we compare against the actual
graded labels. For binary judgments, following prior work [ 19,37],
we classify levels 2 and 3 as relevant and levels 0 and 1 as non-
relevant. For pairwise judgments, we compare documents with
different relevance levels, assuming that a document with a higher
relevance level should be ranked as more relevant than one with a
lower relevance level.
LLMs for Relevance Judgments. To perform relevance assessment,
we employed three different LLMs: one commercial model, GPT-4o ,
and two open-source models, LLaMA 3.2-3B andMistral-7B . We
implemented our experiments using OpenAI and Ollama, running
all prompts with a temperature setting of 0.
Data Sampling. We conducted experiments on all query-document
pairs for binary and graded relevance judgments using the open-
source models. However, due to computational constraints, we were
unable to run all 24 valid prompts across all query-document pairs
forGPT-4o . Instead, we randomly sampled up to 10 documents per
query for each of the four relevance levels (0-3). If fewer than 10
documents were available for a given relevance level, we included
all available documents. For pairwise judgments, evaluating all pos-
sible pairs was not feasible due to their quadratic growth. Instead,
we categorized documents for each query into three groups: “highly
relevant”, “relevant”, and “non-relevant”. The “highly relevant” cat-
egory corresponds to the highest available relevance level for that
query, which in TREC-style annotations could be level 3 or level 2,
depending on availability. The “non-relevant” category includes all
level 0 documents, while any intermediate relevance level (typically
level 1, or levels 1 and 2 if level 3 exists) was classified as “relevant”.
2https://lmarena.ai/Table 2: Mean and variance of agreement between LLM-based
and human relevance judgments across different settings.
Model crafted byBinary Graded Pairwise
Mean Variance Mean Variance Mean Variance
GPT-4oLLM 0.434 0.003 0.215 0.001 0.849 0.000
Human 0.270 0.098 0.215 0.001 0.578 0.139
LLaMA 3.2LLM 0.303 0.010 0.033 0.002 0.439 0.066
Human 0.167 0.041 0.102 0.003 0.330 0.073
MistralLLM 0.405 0.001 0.008 0.004 0.574 0.014
Human 0.243 0.051 0.004 0.005 0.442 0.073
From these three categories, we constructed document pairs for
pairwise judgments. Specifically, we sampled 10 pairs per query
from each of the following comparisons: “highly relevant vs. non-
relevant”, “relevant vs. non-relevant”, and “highly relevant vs. rele-
vant” (up to 30 pairs in total). If fewer than 10 pairs were available
for a given comparison, we included as many as possible. Addi-
tionally, for the pairwise setting, we minimized positional bias by
evaluating each document pair twice, swapping the order of the
documents in the second run. The result is counted as “agree” if the
LLM favors the more relevant passage in both comparisons, “tie”
if the LLM’s decisions are inconsistent when the passage order is
swapped, and “disagree” if the LLM consistently selects the passage
with a lower relevance level assigned by human annotators.
4 Results and Findings
In order to explore the research questions raised in the introduction,
we investigated the agreement of LLM-based relevance judgments
from different prompts with human annotations on TREC 2020 and
2021 using three different LLMs, as shown in Figure 2. For binary
and graded relevance judgments, agreement is measured using Co-
hen’s Kappa ( 𝜅). For pairwise judgments, since the task involves
assessing agreement with the actual ranking of pairs, we report the
percentage of cases where the LLM’s preference agrees with the
expected order. In this figure, the leftmost two columns represent
the results for binary, the middle two columns correspond to graded,
and the rightmost two columns display the results from pairwise
relevance judgment. The green, blue, and red bars indicate agree-
ment for GPT-4o ,LLAMA 3.2 , and Mistral , respectively. In each
pair of plots, the left plot presents results for DL-20, while the right
plot corresponds to DL-21. The bottom 12 bars represent prompts
crafted by LLMs; on top of them there are 12 bars corresponding to
prompts created by humans.
In addition to results from the human- and LLM-written prompts,
we also report the results of UMBRELA assessments at the top of the
graded relevance sub-figure (middle). UMBRELA is an open-source
reproduction of Microsoft’s Bing LLM-based relevance assessor
[35], designed to automate relevance judgments effectively [ 36,37].
It follows a structured prompting approach and has demonstrated
high correlation with both human annotations and system rankings
across multiple TREC Deep Learning Tracks (2019–2023). Notably,
UMBRELA has been integrated into TREC 2024 RAG for automated
evaluation, which further validated its reliability as an alternative
to human assessors. We consider UMBRELA a reliable and effective

SIGIR ’25, July 13–18, 2025, Padua, Italy Negar Arabzadeh and Charles L.A. Clarke
0.2
0.1
0.0
0.1
0.2
0.3
0.4
0.5
Cohen's Kappa 
 DL-20Athene-v2Grok-2Gemma-2-9bDeepseek-v3Mistral-largePhi-4Claude3.5Gemini 2GPTO1 MiniGPTO1GPT4o MiniGPT4oH-12H-11H-10H-9H-8H-7H-6H-5H-4H-3H-2H-1
-0.15-0.20
-0.14-0.12
-0.45-0.35
0.2
0.1
0.0
0.1
0.2
0.3
0.4
0.5
Cohen's Kappa 
 DL-21-0.27-0.30
-0.29-0.28
-0.45-0.38
0.1
0.0
0.1
0.2
Cohen's Kappa 
 DL-20Athene-v2Grok-2Gemma-2-9bDeepseek-v3Mistral-largePhi-4Claude3.5Gemini 2GPTO1 MiniGPTO1GPT4o MiniGPT4oH-12H-11H-10H-9H-8H-7H-6H-5H-4H-3H-2H-1UMBRELA
0.1
0.0
0.1
0.2
Cohen's Kappa 
 DL-21
0.2
0.1
0.0
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
Pairwise Agreement 
 DL-20Athene-v2Grok-2Gemma-2-9bDeepseek-v3Mistral-largePhi-4Claude3.5Gemini 2GPTO1 MiniGPTO1GPT4o MiniGPT4oH-12H-11H-10H-9H-8H-7H-6H-5H-4H-3H-2H-1
0.2
0.1
0.0
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
Pairwise Agreement 
 DL-21Binary Graded PairwiseGPT-4o LLaMA 3.2 Mistral
Figure 2: Agreement of LLM-based relevance judgments with human annotations across different prompts and relevance
judgment tasks. UMBRELA represents the reproduction of Bing’s LLM assessor introduced in [ 37]. Otherwise, the top 12 bars
(H-*) represent human-crafted prompts, while the bottom 12 correspond to LLM-generated prompts. The dashed lines show the
mean of agreement in LLM -crafted prompts and human-crafted prompts separately.
prompt and we believe comparing its performance against human-
crafted and LLM-generated prompts in graded relevance judgments
would bring interesting insights. Additionally, Table 2 summarizes
Figure 2 by providing the mean and variance of agreement scores
across the two datasets and different relevance judgments.
We now consider investigating each of our research questions
in light of these agreement results.
RQ1. Impact of Prompts on LLM-based Relevance Judg-
ment Approaches: Figure 2 and Table 2 reveal significant variance
across different LLM-based relevance judgment approaches. Binary
and pairwise methods exhibit the least sensitivity to input prompts,
maintaining more consistent agreement. In contrast, graded rel-
evance judgments are highly sensitive to prompt variations. We
note that while binary and pairwise methods operate with only two
choices, graded relevance introduces greater variability. Particularly
on graded judgments, GPT-4o demonstrates relatively stable perfor-
mance but LLaMA 3.2 andMistral show considerable fluctuations
across different prompts.RQ2. LLMs as Prompt Generators: Table 2 shows that LLM-
generated prompts generally yield higher average agreement with
human annotations. However, for graded relevance judgments, the
difference is minimal. This may be due to (i) participants’ greater
familiarity with graded assessments or (ii) the inherently subjec-
tive nature of assigning relevance levels, which may require more
calibration with human annotators. Additionally, LLM-generated
prompts exhibit lower variance in agreement compared to human-
crafted prompts, indicating less sensitivity to prompt variations.
RQ3. Prompt Robustness Across LLMs: Figure 3 analyzes
inter-agreement rates among different prompt groups using Krip-
pendorff’s alpha. Here we measure agreement between different
prompt’s output, regardless of their alignment with human judg-
ments. The results show that LLM-generated prompts exhibit higher
inter-agreement than human-crafted ones, likely due to the greater
linguistic diversity in human-generated prompts, as seen in Fig-
ure 1. This suggests that LLM-generated prompts are more robust

A Human-AI Comparative Analysis of Prompt Sensitivity in LLM-Based Relevance Judgment SIGIR ’25, July 13–18, 2025, Padua, Italy
Figure 3: Krippendorff’s inter-agreement rate between all
the prompts on two datasets.
than human-crafted ones. While some human-crafted prompts per-
formed well across all models, prompt effectiveness varies signifi-
cantly between LLMs, with no single prompt consistently excelling
across all models. However, for graded assessments, UMBRELA
consistently demonstrated high performance across different LLMs
and it emerged as one of the most effective prompts across all
models. UMBRELA had previously shown strong correlation with
human judgments on TREC DL tracks [ 37]. We hypothesize that
UMBRELA’s strong and consistent performance may stem from
how its prompt deconstructs the concept of relevance into finer-
grained aspects, such as trustworthiness and alignment with intent.
This structured approach likely prevents the LLM from relying on
its own interpretation of relevance.
RQ4. Model-Specific Sensitivity to Prompts: From Figure 2,
we observe that GPT-4o demonstrates high consistency across most
prompts and all relevance assessment approaches. In contrast, the
performance of LLaMA 3.2 andMistral varies significantly de-
pending on the prompt and assessment method. This variability is
further confirmed by the variance of agreement reported in Table 2.
Notably, GPT-4o exhibits consistently low variance in agreement,
particularly when prompted with LLM-crafted prompts.
5 Conclusion and Limitations
In this study, we investigated the sensitivity of LLM-based relevance
judgments to different prompting strategies across multiple models.
We examined how prompts, whether human- or LLM-generated,
influence judgment effectiveness, their robustness across different
LLMs, and the extent to which models exhibit variability in response
to prompt modifications. One specific outcome is to confirm the
performance of UMBRELA as a leading prompt for LLM-based
graded relevance assessment. Despite these contributions, our study
has limitations. Our human participants primarily had a computer
science background with experience writing prompts for LLMs.
Additionally, we evaluated only three LLMs as judges, limiting the
generalizability of our findings.
References
[1]Marwah Alaofi, Negar Arabzadeh, Charles LA Clarke, and Mark Sanderson. 2024.
Generative information retrieval evaluation. In Information Access in the Era of
Generative AI . Springer, 135–159.
[2]Negar Arabzadeh, Amin Bigdeli, and Charles L. A. Clarke. 2024. Adapting
Standard Retrieval Benchmarks to Evaluate Generated Answers. In 46th European
Conference on Information Retrieval . Glasgow, Scotland.[3]Negar Arabzadeh and Charles LA Clarke. 2024. A Comparison of Methods for
Evaluating Generative IR. arXiv preprint arXiv:2404.04044 (2024).
[4]Negar Arabzadeh, Siqing Huo, Nikhil Mehta, Qingyun Wu, Chi Wang,
Ahmed Hassan Awadallah, Charles L. A. Clarke, and Julia Kiseleva. 2024.
Assessing and Verifying Task Utility in LLM-Powered Applications. In Pro-
ceedings of the 2024 Conference on Empirical Methods in Natural Language
Processing , Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (Eds.). As-
sociation for Computational Linguistics, Miami, Florida, USA, 21868–21888.
doi:10.18653/v1/2024.emnlp-main.1219
[5]Simran Arora, Avanika Narayan, Mayee F. Chen, Laurel Orr, Neel Guha, Kush
Bhatia, Ines Chami, Frederic Sala, and Christopher Ré. 2022. Ask Me Anything:
A simple strategy for prompting language models. arXiv:2210.02441 [cs.CL]
https://arxiv.org/abs/2210.02441
[6]Leif Azzopardi, Charles LA Clarke, Paul Kantor, Bhaskar Mitra, Johanne R Trippas,
Zhaochun Ren, Mohammad Aliannejadi, Negar Arabzadeh, Raman Chandrasekar,
Maarten de Rijke, et al .2024. Report on The Search Futures Workshop at ECIR
2024. In ACM SIGIR Forum , Vol. 58. ACM New York, NY, USA, 1–41.
[7]Chris Buckley and Ellen M Voorhees. 2004. Retrieval evaluation with incomplete
information. In Proceedings of the 27th annual international ACM SIGIR conference
on Research and development in information retrieval . 25–32.
[8]Ben Carterette, Paul N. Bennett, David Maxwell Chickering, and Susan T. Dumais.
2008. Here or there: Preference Judgments for Relevance . Computer Science
Department Faculty Publication Series 46. University of Massachusetts Amherst.
[9]Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Linyi Yang, Kaijie Zhu, Hao
Chen, Xiaoyuan Yi, Cunxiang Wang, Yidong Wang, et al .2024. A survey on
evaluation of large language models. ACM Transactions on Intelligent Systems
and Technology 15, 3 (2024), 1–45.
[10] Anwoy Chatterjee, HSVNS Kowndinya Renduchintala, Sumit Bhatia, and Tanmoy
Chakraborty. 2024. POSIX: A Prompt Sensitivity Index For Large Language
Models. arXiv preprint arXiv:2410.02185 (2024).
[11] Cheng-Han Chiang and Hung-yi Lee. 2023. Can large language models be an
alternative to human evaluations? arXiv preprint arXiv:2305.01937 (2023).
[12] Wei-Lin Chiang, Lianmin Zheng, Ying Sheng, Anastasios Nikolas Angelopoulos,
Tianle Li, Dacheng Li, Hao Zhang, Banghua Zhu, Michael Jordan, Joseph E
Gonzalez, et al .2024. Chatbot arena: An open platform for evaluating llms by
human preference. arXiv preprint arXiv:2403.04132 (2024).
[13] Charles L. A. Clarke, Alexandra Vtyurina, and Mark D. Smucker. 2021. Assessing
Top-𝑘Preferences. ACM Trans. Inf. Syst. 39, 3, Article 33 (may 2021), 21 pages.
doi:10.1145/3451161
[14] Charles L. A. Clarke, Alexandra Vtyurina, and Mark D. Smucker. 2021. Assessing
top-𝑘preferences. ACM Transactions on Information Systems 39, 3 (July 2021).
[15] Cyril W Cleverdon. 1991. The significance of the Cranfield tests on index lan-
guages. In Proceedings of the 14th annual international ACM SIGIR conference on
Research and development in information retrieval . 3–12.
[16] Nick Craswell, Bhaskar Mitra, Emine Yilmaz, and Daniel Campos. 2021. Overview
of the TREC 2020 deep learning track. arXiv:2102.07662 [cs.IR] https://arxiv.org/
abs/2102.07662
[17] Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Daniel Campos, and Jimmy Lin.
2022. Overview of the TREC 2021 deep learning track. In Text REtrieval Conference
(TREC) . NIST, TREC. https://www.microsoft.com/en-us/research/publication/
overview-of-the-trec-2021-deep-learning-track/
[18] Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Daniel Campos, and Ellen M
Voorhees. 2020. Overview of the TREC 2019 deep learning track. arXiv preprint
arXiv:2003.07820 (2020).
[19] Guglielmo Faggioli, Laura Dietz, Charles LA Clarke, Gianluca Demartini, Matthias
Hagen, Claudia Hauff, Noriko Kando, Evangelos Kanoulas, Martin Potthast,
Benno Stein, et al .2023. Perspectives on large language models for relevance
judgment. In Proceedings of the 2023 ACM SIGIR International Conference on
Theory of Information Retrieval . 39–50.
[20] Naghmeh Farzi and Laura Dietz. 2024. Pencils down! automatic rubric-based
evaluation of retrieve/generate systems. In Proceedings of the 2024 ACM SIGIR
International Conference on Theory of Information Retrieval . 175–184.
[21] David Hawking, Ellen Voorhees, Nick Craswell, Peter Bailey, et al .1999. Overview
of the trec-8 web track. In TREC .
[22] Gabriella Kazai, Emine Yilmaz, Nick Craswell, and S.M.M. Tahaghoghi. 2013. User
Intent and Assessor Disagreement in Web Search Evaluation. In 22nd ACM Inter-
national Conference on Information and Knowledge Management . San Francisco,
California, 699–708.
[23] Alina Leidinger, Robert van Rooij, and Ekaterina Shutova. 2023. The lan-
guage of prompting: What linguistic properties make a prompt successful?
arXiv:2311.01967 [cs.CL] https://arxiv.org/abs/2311.01967
[24] Dawei Li, Bohan Jiang, Liangjie Huang, Alimohammad Beigi, Chengshuai Zhao,
Zhen Tan, Amrita Bhattacharjee, Yuxuan Jiang, Canyu Chen, Tianhao Wu, et al .
2024. From Generation to Judgment: Opportunities and Challenges of LLM-as-a-
judge. arXiv preprint arXiv:2411.16594 (2024).
[25] Sheng Lu, Hendrik Schuff, and Iryna Gurevych. 2024. How are Prompts Different
in Terms of Sensitivity?. In Proceedings of the 2024 Conference of the North Ameri-
can Chapter of the Association for Computational Linguistics: Human Language

SIGIR ’25, July 13–18, 2025, Padua, Italy Negar Arabzadeh and Charles L.A. Clarke
Technologies (Volume 1: Long Papers) , Kevin Duh, Helena Gomez, and Steven
Bethard (Eds.). Association for Computational Linguistics, Mexico City, Mexico,
5833–5856. doi:10.18653/v1/2024.naacl-long.325
[26] Sean MacAvaney and Luca Soldaini. 2023. One-shot labeling for automatic rele-
vance estimation. In Proceedings of the 46th International ACM SIGIR Conference
on Research and Development in Information Retrieval . 2230–2235.
[27] Chuan Meng, Negar Arabzadeh, Arian Askari, Mohammad Aliannejadi, and
Maarten de Rijke. 2024. Query Performance Prediction using Relevance Judg-
ments Generated by Large Language Models. arXiv preprint arXiv:2404.01012
(2024).
[28] Hossein A Rahmani, Clemencia Siro, Mohammad Aliannejadi, Nick Craswell,
Charles LA Clarke, Guglielmo Faggioli, Bhaskar Mitra, Paul Thomas, and Emine
Yilmaz. 2024. Llm4eval: Large language model for evaluation in ir. In Proceedings
of the 47th International ACM SIGIR Conference on Research and Development in
Information Retrieval . 3040–3043.
[29] Hossein A. Rahmani, Clemencia Siro, Mohammad Aliannejadi, Nick Craswell,
Charles L. A. Clarke, Guglielmo Faggioli, Bhaskar Mitra, Paul Thomas, and
Emine Yilmaz. 2024. Report on the 1st Workshop on Large Language
Model for Evaluation in Information Retrieval (LLM4Eval 2024) at SIGIR 2024.
arXiv:2408.05388 [cs.IR] https://arxiv.org/abs/2408.05388
[30] Amirhossein Razavi, Mina Soltangheis, Negar Arabzadeh, Sara Salamat, Morteza
Zihayat, and Ebrahim Bagheri. 2025. Benchmarking Prompt Sensitivity in Large
Language Models. arXiv preprint arXiv:2502.06065 (2025).
[31] Tetsuya Sakai and Zhaohao Zeng. 2020. Good evaluation measures based on
document preferences. In 43rd International ACM SIGIR Conference on Research
and Development in Information Retrieval . 359–368.
[32] Alireza Salemi and Hamed Zamani. 2024. Evaluating retrieval quality in retrieval-
augmented generation. In Proceedings of the 47th International ACM SIGIR Con-
ference on Research and Development in Information Retrieval . 2395–2400.
[33] David P Sander and Laura Dietz. 2021. EXAM: How to Evaluate Retrieve-and-
Generate Systems for Users Who Do Not (Yet) Know What They Want.. InDESIRES . 136–146.
[34] Melanie Sclar, Yejin Choi, Yulia Tsvetkov, and Alane Suhr. 2023. Quantify-
ing Language Models’ Sensitivity to Spurious Features in Prompt Design or:
How I learned to start worrying about prompt formatting. arXiv preprint
arXiv:2310.11324 (2023).
[35] Paul Thomas, Seth Spielman, Nick Craswell, and Bhaskar Mitra. 2023. Large
Language Models Can Accurately Predict Searcher Preferences. arXiv preprint
arXiv:2309.10621 (2023).
[36] Shivani Upadhyay, Ronak Pradeep, Nandan Thakur, Daniel Campos, Nick
Craswell, Ian Soboroff, Hoa Trang Dang, and Jimmy Lin. 2024. A Large-Scale
Study of Relevance Assessments with Large Language Models: An Initial Look.
arXiv:2411.08275 [cs.IR] https://arxiv.org/abs/2411.08275
[37] Shivani Upadhyay, Ronak Pradeep, Nandan Thakur, Nick Craswell, and Jimmy
Lin. 2024. UMBRELA: UMbrela is the (Open-Source Reproduction of the) Bing
RELevance Assessor. arXiv preprint arXiv:2406.06519 (2024).
[38] Ellen M Voorhees. 2000. Report on trec-9. In ACM SIGIR Forum , Vol. 34. ACM
New York, NY, USA, 1–8.
[39] Xiaohui Xie, Jiaxin Mao, Yiqun Liu, Maarten de Rijke, Haitian Chen, Min Zhang,
and Shaoping Ma. 2020. Preference-based evaluation metrics for web image
search. In 43st Annual International ACM SIGIR Conference on Research and De-
velopment in Information Retrieval . Xi’an, China.
[40] Xinyi Yan, Chengxi Luo, Charles L. A. Clarke, Nick Craswell, Ellen M. Voorhees,
and Pablo Castells. 2022. Human Preferences as Dueling Bandits. In Proceedings
of the 45th International ACM SIGIR Conference on Research and Development in
Information Retrieval (SIGIR ’22) . ACM. doi:10.1145/3477495.3531991
[41] Honglei Zhuang, Zhen Qin, Kai Hui, Junru Wu, Le Yan, Xuanhui Wang, and
Michael Berdersky. 2023. Beyond Yes and No: Improving Zero-Shot LLM Rankers
via Scoring Fine-Grained Relevance Labels. arXiv preprint arXiv:2310.14122
(2023).