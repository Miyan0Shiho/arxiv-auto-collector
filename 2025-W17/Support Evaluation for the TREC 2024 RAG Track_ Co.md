# Support Evaluation for the TREC 2024 RAG Track: Comparing Human versus LLM Judges

**Authors**: Nandan Thakur, Ronak Pradeep, Shivani Upadhyay, Daniel Campos, Nick Craswell, Jimmy Lin

**Published**: 2025-04-21 16:20:43

**PDF URL**: [http://arxiv.org/pdf/2504.15205v1](http://arxiv.org/pdf/2504.15205v1)

## Abstract
Retrieval-augmented generation (RAG) enables large language models (LLMs) to
generate answers with citations from source documents containing "ground
truth", thereby reducing system hallucinations. A crucial factor in RAG
evaluation is "support", whether the information in the cited documents
supports the answer. To this end, we conducted a large-scale comparative study
of 45 participant submissions on 36 topics to the TREC 2024 RAG Track,
comparing an automatic LLM judge (GPT-4o) against human judges for support
assessment. We considered two conditions: (1) fully manual assessments from
scratch and (2) manual assessments with post-editing of LLM predictions. Our
results indicate that for 56% of the manual from-scratch assessments, human and
GPT-4o predictions match perfectly (on a three-level scale), increasing to 72%
in the manual with post-editing condition. Furthermore, by carefully analyzing
the disagreements in an unbiased study, we found that an independent human
judge correlates better with GPT-4o than a human judge, suggesting that LLM
judges can be a reliable alternative for support assessment. To conclude, we
provide a qualitative analysis of human and GPT-4o errors to help guide future
iterations of support assessment.

## Full Text


<!-- PDF content starts -->

Support Evaluation for the TREC 2024 RAG Track:
Comparing Human versus LLM Judges
Nandan Thakur1, Ronak Pradeep1, Shivani Upadhyay1,
Daniel Campos2,Nick Craswell3,Jimmy Lin1
1University of Waterloo2Snowflake3Microsoft
https://trec-rag.github.io
Abstract
Retrieval-augmented generation (RAG) enables large language models (LLMs) to
generate answers with citations from source documents containing “ground truth”,
thereby reducing system hallucinations. A crucial factor in RAG evaluation is
“support”—whether the information in the cited documents supports the answer. To
this end, we conducted a large-scale comparative study of 45 participant submis-
sions on 36 topics to the TREC 2024 RAG Track, comparing an automatic LLM
judge (GPT-4o) against human judges for support assessment. We considered two
conditions: (1) fully manual assessments from scratch and (2) manual assessments
with post-editing of LLM predictions. Our results indicate that for 56% of the
manual from-scratch assessments, human and GPT-4o predictions match perfectly
(on a three-level scale), increasing to 72% in the manual with post-editing condi-
tion. Furthermore, by carefully analyzing the disagreements in an unbiased study,
we found that an independent human judge correlates better with GPT-4o than a
human judge, suggesting that LLM judges can be a reliable alternative for support
assessment. To conclude, we provide a qualitative analysis of human and GPT-4o
errors to help guide future iterations of support assessment.
1 Introduction
Retrieval-Augmented Generation (RAG) has recently gained popularity in both academic and indus-
trial settings (e.g., Bing Search [ 19] and popular frameworks like LangChain [ 4]). In RAG, large
language models (LLMs) generate answers to user queries that include citations to source documents
as necessary [ 11,16,12,3]. RAG systems improve factuality and verifiability, reducing hallucinations
observed in “closed-book” LLM generation [14, 16, 10, 18].
Document-level citations for supporting facts in LLM-generated answers are integral to any deployed
RAG system. Therefore, support evaluation assesses whether a RAG answer factually supports the
information present in the cited documents, which is crucial for evaluating the quality of a RAG
system. Prior work on support evaluation in the RAG literature [ 17,5,26,9,29,31,25] relies on an
automatic judge, i.e., an LLM as a proxy judge. However, it is unknown whether an LLM judge can
potentially replace a human judge for support evaluation.
This paper examines results from the TREC 2024 RAG Track, assessing 45 participant systems on 36
information-based queries. A sample query and answer is shown in Table 1. We conducted a large-
scale comparative study between human and LLM judges using resources provided by the National
Institute of Standards and Technology (NIST) to better understand whether support assessment can
be automated. Unique to the TREC setup, we contrast our automatic judgment process using a strong
LLM judge (like GPT-4o) against a manual process under two conditions: (1) manual from scratch ,arXiv:2504.15205v1  [cs.CL]  21 Apr 2025

Topic: how taylor swift’s age affects her relationships
Answer : Taylor Swift’s age has significantly influenced her relationships, often contributing to both the
dynamics and outcomes of her romances []. Swift’s relationships have frequently involved notable age gaps,
which have sometimes led to complications []. For instance, her relationship with John Mayer, who was 11
years her senior, reportedly strained due to the age difference and Mayer’s reputation as a playboy, leading to
Swift feeling taken advantage of [0,3,4]. This relationship inspired her song "Dear John," reflecting her
emotional turmoil [3,4]. Similarly, Swift’s relationship with Jake Gyllenhaal, who was nine years older, also
highlighted the impact of age differences. Gyllenhaal reportedly felt the age gap, and Swift’s inexperience
made it difficult for her to navigate the complexities of the relationship, leading to heartbreak and inspiring
several songs on her "Red" album [8,1,2]. [...] In summary, Taylor Swift’s age has played a crucial role in
shaping her relationships, influencing both the dynamics and outcomes [8,1]. Age differences have often led
to complications, while relationships with peers have been relatively simpler. Over time, her maturity has
allowed her to navigate romantic complexities with greater understanding and resilience. []
Passage [3]: A Timeline Of Taylor Swift’s Age-Inappropriate Romances | Business Insider [...] Taylor
Swift, 21, & John Mayer, 32. The inappropriateness of Swift’s dating habits peaked when she started seeing
11-years her senior and notorious playboy, John Mayer. Neither party confirmed the romance, but they
reportedly dated through February 2010 [...]
Passage [8]: What Really Went Wrong With Jake Gyllenhaal And Taylor Swift? [...] Taylor Swift and Jake
Gyllenhaal dated from October to December 2010. [...] What went wrong with these two “ twin fire signs? ”
It turns out, a lot. Maybe that’s why it’s such a touchy subject for him these days! The age gap, when they
dated, Gyllenhaal was 29 to Swift’s 20, and by the time they split, he was 30 to her 21. “ He said he could
feel the age difference ,” a source told Us Weekly [...]
Table 1: A sample RAG answer segmented into multiple sentences with citations to passages retrieved
from the MS MARCO V2.1 segment collection for a query about Taylor Swift. We highlight the
cited passages, which are provided as a list (that can be empty, i.e., zero citations).
where human annotators perform assessments from scratch and (2) manual with post-editing , where
human annotators are shown GPT-4o predictions during the evaluation process.
In this paper, we focus exclusively on support, i.e., whether the information in an answer sentence is
supported by the cited documents, which we consider as the “ground truth”. This is, of course, only
one aspect of RAG evaluation. For in-depth analyses of other aspects of the TREC 2024 RAG Track,
we refer readers to Upadhyay et al. [28] for relevance assessment and Pradeep et al. [23] for nugget
evaluation.
Our experimental results indicate that GPT-4o and human judgments perfectly match 56% of the time
in the manual from-scratch condition, increasing to 72% in the manual with post-editing condition.
These results show promise in using LLM judges for support assessment in both conditions. We
measured support of a system’s overall answer in terms of two metrics: weighted precision and
weighted recall, where precision penalizes overcitation, and recall penalizes undercitation. We
observe a high correlation at the run level (above 0.79 Kendall τ) between GPT-4o and human judges,
providing evidence that LLMs can potentially replace human judges for support evaluation.
In addition, to better understand the discrepancies between GPT-4o and human judges, we conducted
an unbiased disagreement study with an independent human judge who carefully re-assessed 537
randomly sampled pairs, including both assessment conditions. Our results surprisingly show that the
independent judge agrees more with GPT-4o than the human judge (e.g., Cohen’s κof 0.27 vs. 0.07).
Finally, we discuss annotation errors to help improve future iterations of support evaluation.
2 Background and Related Work
What we refer to as support evaluation represents an attempt to determine whether information in a
RAG answer is factually supported by its cited documents. In our work, we evaluate support at the
sentence level in the answer. We assume an answer rsegmented into nsentences, r={a1,···, an},
where each answer sentence aican contain a maximum of mdocument citations, ai={di,···, dm},
2

Answer Sentence: For instance, her relationship with
John Mayer, who was 11 years her senior, reportedly
strained due to the age difference and Mayer’s reputa-
tion as a playboy, leading to Swift feeling taken advantage
of.
Passage ID [0]: doc_04_1081579649#7_2253255175
Title: Timeline of Taylor Swift’s Relationships
Text: 2009: Taylor Swift, 20, & Taylor Lautner, [...] 2010:
Taylor Swift, 21, & John Mayer, 321 / 12 And then the in-
appropriateness ofSwift’s datinghabits peaked when she
started seeing11-years herseniorandnotoriousplayboy,
John Mayer.
Human Judge: PartialSupport
GPT-4o Judge: PartialSupport
Answer Sentence: This relationship inspired her song
“Dear John,” reflecting her emotional turmoil.
Passage ID [3]: doc_35_202251892#8_427548986
Title: Timeline Of Taylor Swift’s Age-Inappropriate Ro-
mances | Business Insider
Text: [...] 2010: Taylor Swift, 21, & John Mayer, 321 /
12 And then the inappropriateness of Swift’s dating habits
peaked [...] and then theheart brokenyoung Swift penned
thesong ’Dear John’ about thebreak up. Earlier this year,
Mayer admitted that he felt ‘humiliated’ when he heard
the song, but Swift refuses to admit it’s about him, telling
Glamour magazine it was ‘presumptuous’ of him to think
the song was about him.
Human Judge: FullSupport
GPT-4o Judge: FullSupportAnswer Sentence: The age difference was a significant
factor in their breakup, with Gyllenhaal not ready to com-
mit, further exacerbating Swift’s emotional distress. Con-
versely, Swift’s relationships with peers closer to her age,
such as Joe Jonas, were less fraught with such issues.
Passage ID [8]: doc_48_737500982#1_1325021022
Title: What Went Wrong With Jake Gyllenhaal And Tay-
lor Swift?
Text: Taylor Swift and Jake Gyllenhaal dated from Oc-
tober to December 2010. [...] “He said hecould feelthe
agedifference,” a source told Us Weekly. [...] “When
Jake broke her heart, shewassoinexperienced shedidn’t
know how todealwith it...Shewasn’t used toallthehead
games andthelies but now she’s way less naive.”
Human Judge: FullSupport
GPT-4o Judge: PartialSupport
Answer Sentence: As she matured, her understanding
of relationships evolved, making her less naive and more
discerning in her romantic choices.
Passage ID [8]: doc_48_737500982#1_1325021022
Title: What Went Wrong With Jake Gyllenhaal And Tay-
lor Swift?
Text: Taylor Swift and Jake Gyllenhaal dated from Octo-
ber to December 2010. [...] “He said he could feel the age
difference,” a source told Us Weekly. [...] “When Jake
broke her heart, she was so inexperienced she didn’t know
how to deal with it. She wasn’t used to all the head games
and the lies but now she’s way less naive.”
Human Judge: NoSupport
GPT-4o Judge: PartialSupport
Table 2: Examples of support evaluation with GPT-4o and human judges for the Taylor Swift topic
“how taylor swift’s age affects her relationships ”. The fragment of the passage that supports the
answer sentence is highlighted.
each of which are documents drawn from a corpus.1Support is calculated as the function f(ai, dj) =
si,jwhere fcan be a human or LLM judge that generates a scalar value si,j, indicating the extent
that the cited document djprovides support to sentence ai. A few examples of support evaluation
are shown in Table 2. Apart from RAG, support has been primarily explored in the literature for
summarization [15, 13] and natural language explanations [2, 27].
Previous work on support evaluation in RAG used different automatic judges: examples include an
natural language inference (NLI) model [ 10], LLM with prompting [ 9], or even fine-tuned custom
LLMs [ 26] as the automatic judge. Wu et al. [29] evaluated the tug of war between an LLM’s
internal prior over supporting wrong context information. Similar to our formulation, Ming et al. [20]
provided an evaluation benchmark consisting of academic question answering (QA) datasets with
human validation and Liu et al. [17] evaluated the quality of proprietary search engine outputs with
crowdsourced human judges. In contrast, our work is one of the first to conduct a large-scale human
annotation study—encompassing 11K human assessments over multiple RAG systems on 36 topics
containing non-factoid, decompositional, and multi-perspective queries. This study design provides a
rich backdrop for comparing human and GPT-4o judges for support evaluation.
3 Track Description & Assessment Methodology
3.1 TREC 2024 RAG Track
The Text Retrieval Conference (TREC) has led the way in many aspects of evaluation in information
retrieval (IR), natural language processing (NLP), and beyond, for accelerating research within the
1Consistent with parlance in the field, we refer to documents in the generic sense, even though in actuality a
“document” may be a passage (as in our case), a PDF, or even an image.
3

community (both researchers and practitioners). Each year, TREC organizes several tracks, focused
on topics ranging from text or multimodal retrieval [6, 7, 30] to conversational QA [1].
The context for this work is the TREC 2024 RAG Track, which was divided into three tasks: retrieval
(R), augmented generation (AG), and retrieval-augmented generation (RAG). Here, we focus on
the generation part, i.e., participant systems are presented with queries (called topics in TREC
parlance) and candidate passages. These candidate passages are either generated by us (i.e., the track
organizers) and shared with all the participants [ 22] (the AG task) or each participant can directly
perform retrieval from the MS MARCO V2.1 segment collection (the end-to-end RAG task). The
candidate passages provide the context or grounding of RAG for synthesizing the final free-form
answer. We require that answers be segmented into sentences, and that each sentence is associated
with citations to passages from the corpus, as shown in Table 1. Since many teams participated in the
TREC 2024 RAG Track, our human and LLM judges were exposed to multiple answers and cited
documents during support evaluation.
Passage collection. The MS MARCO V2.1 segment collection contains 113,520,750 text passages,
derived from a deduplicated version of the MS MARCO V2 document collection [ 6] by removing
near-duplicate documents using locality-sensitive hashing (LSH) with MinHash and 9-gram shingles.
This reduced the original document count from 11,959,635 to 10,960,555 documents. Passages were
derived from the corresponding document collection using a sliding-window chunking technique
with overlap—specifically, using windows of 10 sentences with a stride of 5 sentences, producing
passages typically between 500–1000 characters. Each passage comprises a title field containing
the title of the passage and a text field containing the body of the passage.
Topic collection. For the TREC 2024 RAG Track topics (queries), we leveraged a fresh scrape of
Bing Search logs containing non-factoid queries that are multifaceted and subjective, warranting RAG
systems to provide long-form answers [ 22,24]. We gathered topics close to the submission period of
the evaluation (around July 2024) to avoid staleness and minimize potential data leakage. Here, we
are not concerned about data contamination in support assessment: since we are retrieving from web
corpora, our passages are likely present in LLM pre-training data. Due to budget constraints with
human annotations, we only ran evaluations using a subset of 36 topics selected from the complete
TREC 2024 RAG Track topic collection.
3.2 Support Assessment
Consistent with previous support evaluations in RAG [ 17,10], we used a three-level grade, with the
following associated descriptions for each support level:
FSFullSupport: All of the information in the answer sentence is factually consistent with and
supported by the cited passage.
PSPartialSupport: Some of the information in the answer sentence is factually consistent with
and supported by the cited passage, but other parts of the sentence are not supported.
NSNoSupport: The cited passage is completely irrelevant and does not support any part of the
answer sentence.
An edge case is a sentence with zero citations: We automatically consider the support assessment to
be “no support”, as the sentence does not cite any retrieved passage.
Next, in order to evaluate the quality of LLM judges in contrast to human judges, we conducted our
support assessment with human judges under two conditions: (1) manual from scratch and (2) manual
with post-editing. We describe both conditions in detail below:
1.Manual from scratch. In this condition, a human judge is provided with the answer sentence
and the cited passage. The judge reads both carefully and evaluates whether the answer sentence
is supported by the cited passage (according to one of the labels above).
2.Manual with post-editing. In this condition, a human judge is provided with the answer
sentence, the cited passage, and the support judgment label given by the LLM judge. The
human judge reads the sentence and passage carefully and provides an assessment using the
LLM judgment label as a reference.
4

In this task, you will evaluate whether each statement is supported by its corresponding citations. Note
that the system responses may appear very fluent and well-formed, but contain slight inaccuracies that are
not easy to discern at first glance. Pay close attention to the text.
You will be provided with a statement and its corresponding passage which the statement cites. It may be
helpful to ask yourself whether it is accurate to say “ according to the citation ... ” with the statement fol-
lowing this phrase. Be sure to check all of the information in the statement. You will be given three options:
• Full Support: All of the information in the statement is supported in the citation.
• Partial Support: Some parts of the information are supported in the citation, but other parts are missing.
• No Support: The citation does not support any part of the statement.
Please provide your response based on the information in the citation. If you are unsure, use your best
judgment. Respond as either “Full Support”, “Partial Support”, or “No Support” with no additional
information.
Statement: { statement }
Citation: { citation }
Figure 1: Prompt used by the GPT-4o judge for support evaluation.
For automatic labeling, we utilized GPT-4o as an automatic judge. We ran inference using the
Microsoft Azure API [ 21], providing a single passage at a time in the prompt2using the answer
sentence and the cited passage. The GPT-4o judge is presented with each sentence and its cited
passage and asked to determine the support label without any explanation (full support, partial support,
or no support). The prompt used is given in Figure 1.
3.3 Computational Cost & Evaluation Tradeoffs
In the TREC 2024 RAG Track, we allowed participants to provide citations for up to 20 passages per
answer sentence. To judge each sentence and its cited passage, our protocol requires a human judge
to read the answer sentence and a relatively long text passage (typically, 500–1000 characters). Thus,
conducting an exhaustive evaluation of all cited passages for every answer sentence across multiple
participants was not feasible given our budget constraints.
Therefore, we had to choose between sparse and dense annotations. Dense annotations would provide
fewer judged topics, but each answer sentence would be evaluated against kcited passages. On the
other hand, sparse annotations would provide higher diversity in judged topics, but at the cost of
judging fewer cited passages for every answer sentence.
We opted for sparse annotations to achieve more judged topics. We fixed both the human and GPT-4o
judge to evaluate only the first cited passage of every answer sentence for all participants. As with all
TREC evaluations, NIST provided the resources to perform human evaluations based on the guidance
of the track organizers (i.e., us). NIST first trained every human judge to understand the task, and
then each human judge evaluated each topic sequentially.
3.4 Support Evaluation Metrics
Support can be evaluated across two dimensions, similar to Liu et al. [ 17]: (1) weighted precision ,
accounting for how many correct passage citations are present in the generated answer, and (2)
weighted recall , accounting for how many sentences in the answer are supported by passage citations
We define both metrics below:
Weighted precision. This metric measures the weighted proportion of citations that support each
answer sentence. We assign a weight to s(ai, dj)of1.0to Full Support (FS), 0.5to Partial Support
(PS), and 0to No Support (NS) for the answer sentence and cited passage. To explain the metric
clearly, let us assume a RAG answer with 3sentences = {a1, a2, a3}, and a corpus Cwith 2passages:
2Separately, we experimented with providing multiple cited passages at once, but we anecdotally observed that
providing one passage at a time performs better with the GPT-4o judge.
5

Condition #Topics #AnnotationsSupport level
FS PS NS
(1a) Manual from scratch (Human) 22 6,742 2,752 1,652 2,338
(1b) Automatic (GPT-4o) 22 6,742 3,110 2,421 1,211
(2a) Manual with post-editing (Human) 14 4,165 1,812 1,076 1,277
(2b) Automatic (GPT-4o) 14 4,165 2,045 1,330 790
Table 3: Descriptive statistics for support judgments for the (1) manual from-scratch condition and
(2) manual with post-editing condition for 45 participant submissions on 36 topics.
{p1, p2}. Now, let’s assume that passage p1partially supports a1, passage p2fully supports a2, and
a3has zero citations. We compute weighted precision as follows:
Weighted Precision =s(a1, p1) +s(a2, p2)
count ({a1, p1},{a2, p2})=0.5 + 1
2= 0.75
Weighted recall. This metric measures the weighted proportion of answer sentences that are
supported by their cited passages. We assign the same weights as defined above in weighted precision.
For the above example, we compute weighted recall as follows:
Weighted recall =s(a1, p1) +s(a2, p2)
count ({a1, a2, a3})=0.5 + 1
3= 0.5
Precision penalizes overcitation in the answer text that is often distracting and unnecessary for
the user. On the other hand, recall penalizes answers with undercitation , i.e., sentences with zero
citations. Therefore, if a sentence has zero citations, it reduces the recall score but keeps precision
unchanged. As described above, we evaluated only the first cited passage per answer sentence for
support. Therefore, the weighted recall and precision scores are identical if all answer sentences have
at least one citation. We save for future work how to best evaluate multiple passage citations for every
sentence in the answer, as it would require a much larger annotation budget.
4 Experimental Results
For the TREC 2024 RAG Track, NIST received 93 runs from 20 groups for the end-to-end RAG
task and 53 runs from 11 groups for the AG task. Given resource constraints, we evaluated only
the two highest-priority submissions from each group across the RAG and AG tasks. As shown in
Table 3, this translates into 31 runs from 18 groups for RAG and 14 runs from 9 groups for AG,
totaling a maximum of 45 participant submissions for each topic. The human judges were able to
complete judgments for 36 topics, sparsely annotated: 6,742 annotations on 22 topics in the manual
from-scratch condition and 4,165 annotations on 14 topics in the manual with post-editing condition.
4.1 Weighted Precision and Recall
Table 5 and Table 6 show the average weighted precision and recall scores of all participant runs for
both the AG and RAG tasks in the manual from-scratch condition on the 22 topics evaluated by human
and GPT-4o judges, respectively. Table 7 and Table 8 show the average weighted precision and recall
scores for all participants in the manual with post-editing condition on the 14 topics evaluated by
human and GPT-4o judges, respectively. We sort the runs in terms of the average weighted precision
score in descending order.
In Figure 2, we show scatter plots of weighted precision and recall scores obtained by all participant
submissions. Run-level scores (denoted by ×) are strongly correlated (all above 0.79 Kendall’s τ)
between GPT-4o and human annotations. Per-topic averages (denoted by △) vary on both axes, where
certain topics achieve a higher weighted precision and recall score than humans over GPT-4o, and
vice versa. Individual participant scores (denoted by #or□) show a high variance in both weighted
precision and recall scores. This is likely due to the mismatch of human annotators preferring “no
support”, whereas GPT-4o prefers “partial support”. Overall, we observe the majority of scores in the
6

0.0 0.2 0.4 0.6 0.8 1.0
Automatic (GPT-4o Judge)0.00.20.40.60.81.0Manual (Human Judge)Kendall's  (all topics/runs): 0.47
Kendall's  (per-topic avg): 0.596
Kendall's  (run-level): 0.884
Weighted Precision / Manual from Scratch
individual
per-topic avg
run-level
0.0 0.2 0.4 0.6 0.8 1.0
Automatic (GPT-4o Judge)0.00.20.40.60.81.0Manual (Human Judge)Kendall's  (all topics/runs): 0.539
Kendall's  (per-topic avg): 0.644
Kendall's  (run-level): 0.892
Weighted Recall / Manual from Scratch
individual
per-topic avg
run-level
0.0 0.2 0.4 0.6 0.8 1.0
Automatic (GPT-4o Judge)0.00.20.40.60.81.0Manual (Human Judge)Kendall's  (all topics/runs): 0.611
Kendall's  (per-topic avg): 0.629
Kendall's  (run-level): 0.792
Weighted Precision / Manual with Post-editing
individual
per-topic avg
run-level
0.0 0.2 0.4 0.6 0.8 1.0
Automatic (GPT-4o Judge)0.00.20.40.60.81.0Manual (Human Judge)Kendall's  (all topics/runs): 0.671
Kendall's  (per-topic avg): 0.695
Kendall's  (run-level): 0.822
Weighted Recall / Manual with Post-editing
individual
per-topic avg
run-levelFigure 2: Correlations between scores from human and GPT-4o judges for the manual from-scratch
condition (top) and the manual with post-editing condition (bottom), measuring weighted precision
and recall. Red markers show run-level scores, yellow triangles show per-topic averages, and blue
dots or green boxes show all individual topic/run combinations. Each plot is annotated with rank
correlations showing Kendall’s τ.
bottom right triangle, indicating that humans take a more conservative approach and provide lower
levels of support overall than GPT-4o, leading to lower weighted precision and recall scores.
4.2 Confusion Matrices
Next, to better understand how often the GPT-4o judge agrees with the human judges, we plot the
confusion matrices in Figure 3. We compare predictions by human judges with GPT-4o on two
conditions: manual from scratch and manual with post-editing.
Manual from-scratch condition. For 56% (13.7% + 11.9% + 30.4%), GPT-4o and the human
judge perfectly agreed on their support judgment on 22 topics. Both “full support” and “no support”
categories have higher percentages (30.4% and 13.7%), showing that humans and GPT-4o as judges
agreed more on both ends of the spectrum. For 15.1%, the GPT-4o judge considered an annotation as
“partial support”, which the human judge annotated as “no support”. Finally, an important observation
is that the GPT-4o judge is more likely to provide a higher support label than the human judge (the
upper right triangle has a higher combined percentage over the lower left triangle).
Manual with post-editing condition. From the previous condition, we see the increase in perfect
agreement rise to 72.1% (15.9% + 18.7% + 37.5%) on 14 topics that were annotated with post-editing
GPT-4o labels. This shows that sentences and cited passages with “partial support” that led to
disagreements in the manual from-scratch condition are reduced. In this condition, human judges are
more likely to agree with the GPT-4o judge unless it is an obvious mistake, i.e., when the GPT-4o
7

No Support Partial Support Full Support
GPT-4o JudgeNo Support Partial Support Full SupportHuman Judge13.7% 15.1% 5.9%
2.8% 11.9% 9.8%
1.5% 8.9% 30.4%Manual from Scratch: 22 Topics
0.050.100.150.200.250.30
No Support Partial Support Full Support
GPT-4o JudgeNo Support Partial Support Full SupportHuman Judge15.9% 8.5% 6.3%
1.8% 18.7% 5.4%
1.3% 4.7% 37.5%Manual with Post-editing: 14 Topics
0.050.100.150.200.250.300.35Figure 3: Confusion matrices comparing predictions from human and GPT-4o judges for the manual
from-scratch condition (left) and the manual with post-editing condition (right).
Cohen’s KappaFrom scratch With post-editing
GPT-4o Human GPT-4o Human
Independent human 0.29 −0.03 0.27 0.07
LLAMA-3.1 (405B) 0.60 −0.20 0.46 −0.06
Figure 4: Inter-annotator agreement score (Co-
hen’s κ) for our unbiased study on disagreements
between GPT-4o and human annotators.
Full Support Partial Support No Support0100200Independent Human GPT-4o LLAMA-3.1Figure 5: Support label prediction by different
judges for each support category (FS, PS, NS)
in the disagreement analysis on 537 sentence–
passage pairs.
judge considers an annotation to be “full support” and the human judge considers it to be “no support”
(increased now to 6.3% from 5.9% in the manual from-scratch condition).
5 Annotator Disagreements
In the experiments reported in Section 4, we observed frequent disagreements between the human and
GPT-4o judge. To further study this, we performed unbiased annotations from scratch by carefully
re-assessing the support judgment of randomly sampled disagreements between the human and
GPT-4o judge, with an independent human judge and another LLM judge using LLAMA-3.1 405B
[8] (with the same prompt as GPT-4o in Figure 1). We randomly sampled 15 disagreement pairs per
topic, re-evaluating 537 sentences and their first cited passages, including both assessment conditions:
(1) manual from scratch and (2) manual with post-editing.
Results. As shown in Figure 4, we interestingly find the independent human judge to be better
correlated with GPT-4o than the human judge provided by NIST (Cohen’s κof 0.29 and 0.27 versus
−0.03 and 0.07) in the manual from-scratch condition. The independent judge fully matched 31% of
the time with the human judge and 51% of the time with the GPT-4o judge. Similarly, in the manual
with post-editing condition, the independent judge fully matched 37% of the time with the human
judge and 52% of the time with the GPT-4o judge. LLAMA-3.1 405B had a stronger correlation with
another LLM (GPT-4o) over human judges (Cohen’s κof 0.60 and 0.46 versus −0.20 and −0.06),
demonstrating the high likelihood of different LLMs providing similar prediction labels.
From the label distributions in the disagreement analysis shown in Figure 5, we observed that
both LLMs (LLAMA-3.1 405B and GPT-4o) labeled about 49–51% of sentences and their cited
passage as “partial support”, whereas the human judge labeled 47% of the sentences as “no support”.
The independent judge labeled 50% of the sentences as “full support”. We keep as future work
explorations of consistency: why LLM judges labeled only a few sentences as “no support”, and
similarly, why human judges labeled a majority of sentences as “no support”.
Qualitative analysis. We further assessed examples qualitatively to understand failure cases, for
example, when a human or GPT-4o judge makes mistakes during support evaluation. In Table 4, we
8

Answer Sentence: Swift started dating 19-year-old One
Direction frontman, Harry Styles, barely after her breakup
from Conor Kennedy.
Passage ID [0]: doc_04_1081579649#7_2253255175
Title: A Timeline Of Taylor Swift’s Age-Inappropriate
Romances | Business Insider
Text: Today, Taylor Swift turns 23, but you would never
know how old the singer actually is based on her love
life. From 18-year-old Conor Kennedy to 35-year-old
John Mayer, Swift is no ageist when it comes to who she
will date. Young or old, British or American, Swift really
doesn’t discriminate in her love life. But at the age of 23,
Swift has already had at least 10 high-profile relationships
(some more flings, really) that have all ended the same
way — in song. In honour of her 23rd year, let’s relive
Swift’s vast array of boyfriends, shall we? [...]
Human Judge: FullSupport×
GPT-4o Judge: NoSupport✓
Answer Sentence: Her first Hollywood romance with Joe
Jonas was age-appropriate, as both were 19 at the time.
Passage ID [3]: doc_04_1081579649#2_2253244363
Title: Timeline of Taylor Swift’s Relationships
Text: [...]2008: TaylorSwift, 19&JoeJonas, 19.Swift’s
firstHollywood romance wasperfectly age-appropriate.
While Swift was just emerging onto the pop scene, Jonas,
along with his singing and dancing brothers, were at the
top of their game. [...]
Human Judge: NoSupport×
GPT-4o Judge: FullSupport✓Answer Sentence: Swift has dated younger men, such as
17-year-old Taylor Lautner when she was 20, which was
controversial due to the age difference.
Passage ID [8]: doc_35_202251892#7_427547583
Title: A Timeline Of Taylor Swift’s Age-Inappropriate
Romances | Business Insider
Text: 2009: TaylorSwift, 20,&TaylorLautner,17.And
here’s where shestarted toslide. Tech nically, thisrelation-
ship was possiblyillegalinmany states as20-year-old
Swift was datingaminor. [...] In October 2010, Swift
revealed her crush: ‘Taylor Lautner. It’s always going to
be Taylor.’ It’s believed that the song ‘Back To December’
is Swift’s apology to Lautner for their breakup. [...]
Human Judge: FullSupport✓
GPT-4o Judge: PartialSupport×
Answer Sentence: Swift’s youth and inexperience have
been cited as factors in her relationships, with some part-
ners reportedly taking advantage of her naivety.
Passage ID [8]: doc_48_481103263#4_840591332
Title: Taylor Swift Boyfriends: Taylor Swift’s Relation-
ships | New Idea Magazine
Text: [...] John Mayer (December 2009-February 2010)
The11-year agegapbetween TaylorandJohn reportedly
strained their relationship, with Taylorfeelingthather
naivety wastaken advantage of. She references their ugly
breakup in her song “Dear John”. Songs: Dear John [...]
Human Judge: FullSupport✓
GPT-4o Judge: PartialSupport×
Table 4: Examples of annotation mistakes by either GPT-4o or the human judge for the topic “ how
taylor swift’s age affects her relationships ” taken from the disagreement analysis. The fragment of
the passage that supports the answer sentence is highlighted.
show a few examples of annotation mistakes on the Taylor Swift topic found in either the human or
GPT-4o judgment. Overall, we summarize a few of the following errors made by GPT-4o:
•GPT-4o can confuse words or phrases with similar meanings; for example, it is unable to
distinguish between police and security specialists.
•GPT-4o can miss out on evaluating the whole sentence (especially information present at the end
of the sentence), biasing towards the “full support” label.
•GPT-4o can label “partial support” if the theme in the answer sentence is similar, but the passage
does not support any text present in the answer sentence, i.e., “no support”.
On the other hand, human judges make mistakes due to not reading the passages carefully. In some
cases, answer sentences that were directly stated in the middle of or at the end of a passage, or
mentioned in parts of the passage, were surprisingly unnoticed by a human judge. This causes the
human judge to label such cases as “no support” instead of “full support”. Lastly, we observe that
a human judge occasionally labels an answer sentence as “full support” even though the passage
doesn’t provide any support information. We suspect that this could be due to an inherent bias relying
on the human judge’s memory or understanding of the topic, instead of strictly relying on the actual
passage text.
6 Conclusion
In this work, we evaluated support in RAG answers by analyzing 45 submissions across 36 topics
from the TREC 2024 RAG Track in a large-scale comparative study involving both humans and LLMs
as judges. We critiqued and evaluated strong LLM judges, like GPT-4o, against human annotators for
support assessment.
9

Our results show a high agreement between GPT-4o and human judgments, with a perfect match
between judgments occurring 56% of the time in the manual from-scratch condition, increasing to
72% in the manual with post-editing condition. We observe that disagreements between humans and
LLMs mainly occur for sentence–passage pairs indicating partial support, i.e., in the middle of the
support evaluation spectrum.
To better understand these disagreements, we conducted an unbiased evaluation by carefully re-
assessing judgments with an independent human judge and a different LLM. Interestingly, in cases
of disagreements, both the independent human judge and the LLAMA-3.1 judge agreed more with
the GPT-4o judge than with the human judge, providing evidence for widely divergent opinions
and perhaps the veracity of using LLMs for support evaluation. Further research could explore
the nuances of disagreements between human and LLM judges and investigate limitations of both
humans and LLMs to improve future iterations of support assessment.
Acknowledgments
This work would not have been possible without the annotator team at NIST. We are grateful for
their contributions. This research was supported in part by the Natural Sciences and Engineering
Research Council (NSERC) of Canada. Additional funding is provided by Snowflake, Microsoft
via the Accelerating Foundation Models Research program, and an Institute of Information &
Communications Technology Planning & Evaluation (IITP) grant funded by the Korean Government
(MSIT) (No. RS-2024-00457882, National AI Research Lab Project). Thanks to Corby Rosset for
providing the test queries, based on the methodology developed in Researchy Questions [24].
References
[1]Mohammad Aliannejadi, Zahra Abbasiantaeb, Shubham Chatterjee, Jeffrey Dalton, and Leif
Azzopardi. 2024. TREC iKAT 2023: A Test Collection for Evaluating Conversational and
Interactive Knowledge Assistants. In Proceedings of the 47th International ACM SIGIR
Conference on Research and Development in Information Retrieval , SIGIR ’24, page 819–829,
New York, NY , USA. Association for Computing Machinery.
[2]Pepa Atanasova, Oana-Maria Camburu, Christina Lioma, Thomas Lukasiewicz, Jakob Grue
Simonsen, and Isabelle Augenstein. 2023. Faithfulness Tests for Natural Language Explanations.
InProceedings of the 61st Annual Meeting of the Association for Computational Linguistics
(Volume 2: Short Papers), ACL 2023, Toronto, Canada, July 9-14, 2023 , pages 283–294.
Association for Computational Linguistics.
[3]Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie
Millican, George van den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark,
Diego de Las Casas, Aurelia Guy, Jacob Menick, Roman Ring, Tom Hennigan, Saffron Huang,
Loren Maggiore, Chris Jones, Albin Cassirer, Andy Brock, Michela Paganini, Geoffrey Irving,
Oriol Vinyals, Simon Osindero, Karen Simonyan, Jack W. Rae, Erich Elsen, and Laurent Sifre.
2022. Improving Language Models by Retrieving from Trillions of Tokens. In International
Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA ,
volume 162 of Proceedings of Machine Learning Research , pages 2206–2240. PMLR.
[4] Harrison Chase. 2022. LangChain.
[5]Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun. 2024. Benchmarking Large Language
Models in Retrieval-Augmented Generation. Proceedings of the AAAI Conference on Artificial
Intelligence , 38(16):17754–17762.
[6]Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Daniel Campos, Jimmy Lin, Ellen M. V oorhees,
and Ian Soboroff. 2022. Overview of the TREC 2022 Deep Learning Track. In Proceedings of
the Thirty-First Text REtrieval Conference (TREC 2022) , Gaithersburg, Maryland.
[7]Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Hossein A. Rahmani, Daniel Campos, Jimmy Lin,
Ellen M. V oorhees, and Ian Soboroff. 2023. Overview of the TREC 2023 Deep Learning Track.
InProceedings of the Thirty-Second Text REtrieval Conference (TREC 2023) , Gaithersburg,
Maryland.
10

[8]Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle,
Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, Anirudh Goyal, Anthony
Hartshorn, and 520 others. 2024. The Llama 3 Herd of Models. CoRR , abs/2407.21783.
[9]Shahul Es, Jithin James, Luis Espinosa Anke, and Steven Schockaert. 2024. RAGAs: Automated
Evaluation of Retrieval Augmented Generation. In Proceedings of the 18th Conference of the
European Chapter of the Association for Computational Linguistics: System Demonstrations ,
pages 150–158, St. Julians, Malta. Association for Computational Linguistics.
[10] Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen. 2023. Enabling Large Language
Models to Generate Text with Citations. In Proceedings of the 2023 Conference on Empirical
Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023 ,
pages 6465–6488. Association for Computational Linguistics.
[11] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. 2020. Retrieval
Augmented Language Model Pre-Training. In Proceedings of the 37th International Conference
on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event , volume 119 of Proceedings
of Machine Learning Research , pages 3929–3938. PMLR.
[12] Gautier Izacard and Edouard Grave. 2021. Leveraging Passage Retrieval with Generative
Models for Open Domain Question Answering. In Proceedings of the 16th Conference of the
European Chapter of the Association for Computational Linguistics: Main Volume, EACL 2021,
Online, April 19 - 23, 2021 , pages 874–880. Association for Computational Linguistics.
[13] Qi Jia, Siyu Ren, Yizhu Liu, and Kenny Q. Zhu. 2023. Zero-shot Faithfulness Evaluation for
Text Summarization with Foundation Language Model. In Proceedings of the 2023 Conference
on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December
6-10, 2023 , pages 11017–11031. Association for Computational Linguistics.
[14] Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, and Mike Lewis. 2020.
Generalization through Memorization: Nearest Neighbor Language Models. In 8th International
Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020 .
OpenReview.net.
[15] Philippe Laban, Wojciech Kryscinski, Divyansh Agarwal, Alexander R. Fabbri, Caiming Xiong,
Shafiq Joty, and Chien-Sheng Wu. 2023. SummEdits: Measuring LLM Ability at Factual
Reasoning Through The Lens of Summarization. In Proceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10,
2023 , pages 9662–9676. Association for Computational Linguistics.
[16] Patrick S. H. Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and
Douwe Kiela. 2020. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.
InAdvances in Neural Information Processing Systems 33: Annual Conference on Neural
Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual .
[17] Nelson Liu, Tianyi Zhang, and Percy Liang. 2023. Evaluating Verifiability in Generative Search
Engines. In Findings of the Association for Computational Linguistics: EMNLP 2023 , pages
7001–7025, Singapore. Association for Computational Linguistics.
[18] Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni,
and Percy Liang. 2024. Lost in the Middle: How Language Models Use Long Contexts.
Transactions of the Association for Computational Linguistics , 12:157–173.
[19] Microsoft. 2023. Reinventing search with a new AI-powered Microsoft Bing and Edge, your
copilot for the web.
[20] Yifei Ming, Senthil Purushwalkam, Shrey Pandit, Zixuan Ke, Xuan-Phi Nguyen, Caiming
Xiong, and Shafiq Joty. 2024. FaithEval: Can Your Language Model Stay Faithful to Context,
Even If “The Moon is Made of Marshmallows”. CoRR , abs/2410.03727.
[21] OpenAI. 2024. Hello GPT-4o.
11

[22] Ronak Pradeep, Nandan Thakur, Sahel Sharifymoghaddam, Eric Zhang, Ryan Nguyen, Daniel
Campos, Nick Craswell, and Jimmy Lin. 2024. Ragnarök: A Reusable RAG Framework and
Baselines for TREC 2024 Retrieval-Augmented Generation Track. arXiv:2406.16828 .
[23] Ronak Pradeep, Nandan Thakur, Shivani Upadhyay, Daniel Campos, Nick Craswell, and
Jimmy Lin. 2024. Initial Nugget Evaluation Results for the TREC 2024 RAG Track with the
AutoNuggetizer Framework. CoRR , abs/2411.09607.
[24] Corby Rosset, Ho-Lam Chung, Guanghui Qin, Ethan C. Chau, Zhuo Feng, Ahmed Awadallah,
Jennifer Neville, and Nikhil Rao. 2024. Researchy Questions: A Dataset of Multi-Perspective,
Decompositional Questions for LLM Web Agents. CoRR , abs/2402.17896.
[25] Dongyu Ru, Lin Qiu, Xiangkun Hu, Tianhang Zhang, Peng Shi, Shuaichen Chang, Cheng
Jiayang, Cunxiang Wang, Shichao Sun, Huanyu Li, Zizhao Zhang, Binjie Wang, Jiarong Jiang,
Tong He, Zhiguo Wang, Pengfei Liu, Yue Zhang, and Zheng Zhang. 2024. RAGChecker:
A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation. In Advances
in Neural Information Processing Systems 38: Annual Conference on Neural Information
Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024 .
[26] Jon Saad-Falcon, Omar Khattab, Christopher Potts, and Matei Zaharia. 2024. ARES: An Auto-
mated Evaluation Framework for Retrieval-Augmented Generation Systems. In Proceedings
of the 2024 Conference of the North American Chapter of the Association for Computational
Linguistics: Human Language Technologies (Volume 1: Long Papers) , pages 338–354, Mexico
City, Mexico. Association for Computational Linguistics.
[27] Noah Y . Siegel, Oana-Maria Camburu, Nicolas Heess, and María Pérez-Ortiz. 2024. The
Probabilities Also Matter: A More Faithful Metric for Faithfulness of Free-Text Explanations in
Large Language Models. In Proceedings of the 62nd Annual Meeting of the Association for
Computational Linguistics, ACL 2024 - Short Papers, Bangkok, Thailand, August 11-16, 2024 ,
pages 530–546. Association for Computational Linguistics.
[28] Shivani Upadhyay, Ronak Pradeep, Nandan Thakur, Daniel Campos, Nick Craswell, Ian Sobo-
roff, Hoa Trang Dang, and Jimmy Lin. 2024. A Large-Scale Study of Relevance Assessments
with Large Language Models: An Initial Look. CoRR , abs/2411.08275.
[29] Kevin Wu, Eric Wu, and James Zou. 2024. How faithful are RAG models? Quantifying the
tug-of-war between RAG and LLMs’ internal prior. CoRR , abs/2404.10198.
[30] Jheng-Hong Yang, Carlos Lassance, Rafael Sampaio De Rezende, Krishna Srinivasan, Miriam
Redi, Stéphane Clinchant, and Jimmy Lin. 2023. AToMiC: An Image/Text Retrieval Test
Collection to Support Multimedia Content Creation. In Proceedings of the 46th International
ACM SIGIR Conference on Research and Development in Information Retrieval , SIGIR ’23,
page 2975–2984, New York, NY , USA. Association for Computing Machinery.
[31] Hao Yu, Aoran Gan, Kai Zhang, Shiwei Tong, Qi Liu, and Zhaofeng Liu. 2024. Evaluation of
Retrieval-Augmented Generation: A Survey. CoRR , abs/2405.07437.
12

Run ID Group Task Weighted Precision Weighted Recall #Sentences
ag_rag_gpt35_expansion_rrf_20 IITD-IRL RAG 0.793 0.793 4.82
Enhanced_Iterative_Fact_Refinement_and_Prioritization TREMA-UNH RAG 0.777 0.777 13.45
UWCrag WaterlooClarke AG 0.717 0.717 7.91
Ranked_Iterative_Fact_Extraction_and_Refinement TREMA-UNH RAG 0.709 0.709 13.68
ldilab_gpt_4o ldisnu RAG 0.698 0.445 12.86
zeph_test_rag_rrf_expand_query IITD-IRL AG 0.681 0.681 4.64
dilab_repllama_listt5_pass3_gpt4o ldisnu AG 0.669 0.408 13.45
baseline_frag_rag24.test_gpt-4o_top20 coordinators AG 0.654 0.401 14.18
cir_gpt-4o-mini_no_reranking_50_0.5_100_301_p1 CIR RAG 0.646 0.646 7.05
neurag neu AG 0.644 0.397 15.82
baseline_frag_rag24.test_command-r-plus_top20 coordinators AG 0.642 0.583 11.68
iiia_dedup_p1_straight_ag IIIA-UNIPD RAG 0.635 0.572 3.77
listgalore_gpt4o_ragnarokv4_top20 h2oloo AG 0.610 0.487 12.14
listgalore_l31-70b_ragnarokv4_top20 h2oloo AG 0.605 0.532 9.64
cohere+post_processing KML RAG 0.582 0.542 18.0
neuragfix neu AG 0.569 0.359 15.82
baseline_rag24.test_l31_70b_instruct_top20 coordinators RAG 0.568 0.556 7.5
ielab-b70bf-70bqp-70bafs ielab AG 0.565 0.445 3.14
iiia_standard_p1_straight_ag IIIA-UNIPD RAG 0.562 0.483 6.23
FT-llama3 uog-tht AG 0.553 0.553 4.64
UDInfolab.RAG.Query InfoLab AG 0.534 0.534 6.86
webis-rag-run0-taskrag webis AG 0.527 0.527 6.27
baseline_top_5 uis-iai RAG 0.522 0.522 3.23
agtask-bm25-colbert_faiss-gpt4o-llama70b softbank-meisei RAG 0.516 0.516 8.27
UDInfolab.RAG.AnsAI InfoLab AG 0.505 0.505 7.18
cir_gpt-4o-mini_Cosine_50_0.5_100_301_p1 CIR RAG 0.495 0.495 7.05
buw buw AG 0.481 0.481 8.14
webis-rag-run1-taskrag webis AG 0.464 0.464 6.18
oneshot_post_sentenced buw AG 0.461 0.356 9.09
rag_bm25-colbert_faiss-gpt4o-llama70b softbank-meisei AG 0.433 0.433 7.27
ruc001 Ruc01 AG 0.425 0.348 14.77
gpt_mini KML RAG 0.368 0.368 7.45
ginger_top_5 uis-iai RAG 0.367 0.367 4.32
LAS-splade-mxbai-mmr8-RAG ncsu-las AG 0.308 0.226 12.82
UWCgarag WaterlooClarke AG 0.305 0.289 9.95
iiresearch-bm25-top10-llama3-8b-instruct ii_research AG 0.293 0.293 4.05
BEST_cot_gpt3.5 citi AG 0.291 0.256 6.23
ICL-mistral uog-tht AG 0.267 0.201 5.82
ielab-b70bf-70bqfs-ad_hoc ielab AG 0.246 0.246 4.59
SECOND_cot_gpt3.5 citi AG 0.246 0.192 5.36
ISIR-IRIT-zephyr_query_gen IRIT AG 0.205 0.115 2.59
LAS-splade-mxbai-rrf-mmr8 ncsu-las AG 0.167 0.116 13.41
ISIR-IRIT-zephyr_p2 IRIT AG 0.131 0.094 1.59
qrant_bge_gemini SGU AG 0.130 0.130 6.45
webis-manual webis AG 0.079 0.037 1.68
Table 5: Weighted precision and recall scores for the top two runs from each group in TREC 2024
RAG Track under the manual from-scratch condition on 22 topics evaluated by human judges
provided by NIST. #Sentences denotes the average number of sentences in the participant’s submitted
answer.
13

Run ID Group Task Weighted Precision Weighted Recall #Sentences
ag_rag_gpt35_expansion_rrf_20 IITD-IRL RAG 0.868 0.868 4.82
Ranked_Iterative_Fact_Extraction_and_Refinement TREMA-UNH RAG 0.861 0.861 13.68
Enhanced_Iterative_Fact_Refinement_and_Prioritization TREMA-UNH RAG 0.846 0.846 13.45
UWCrag WaterlooClarke AG 0.841 0.841 7.91
zeph_test_rag_rrf_expand_query IITD-IRL AG 0.807 0.807 4.64
ldilab_gpt_4o ldisnu RAG 0.791 0.514 12.86
dilab_repllama_listt5_pass3_gpt4o ldisnu AG 0.784 0.491 13.45
baseline_frag_rag24.test_command-r-plus_top20 coordinators AG 0.778 0.718 11.68
iiia_dedup_p1_straight_ag IIIA-UNIPD RAG 0.767 0.715 3.77
cir_gpt-4o-mini_no_reranking_50_0.5_100_301_p1 CIR RAG 0.757 0.757 7.05
baseline_frag_rag24.test_gpt-4o_top20 coordinators AG 0.741 0.467 14.18
neuragfix neu AG 0.737 0.459 15.82
neurag neu AG 0.735 0.458 15.82
ielab-b70bf-70bqp-70bafs ielab AG 0.733 0.591 3.18
baseline_rag24.test_l31_70b_instruct_top20 coordinators RAG 0.731 0.719 7.50
cohere+post_processing KML RAG 0.712 0.671 18.00
listgalore_l31-70b_ragnarokv4_top20 h2oloo AG 0.711 0.622 9.64
iiia_standard_p1_straight_ag IIIA-UNIPD RAG 0.702 0.610 6.23
webis-manual webis AG 0.702 0.415 14.14
listgalore_gpt4o_ragnarokv4_top20 h2oloo AG 0.699 0.563 12.14
cir_gpt-4o-mini_Cosine_50_0.5_100_301_p1 CIR RAG 0.677 0.677 7.05
UDInfolab.RAG.AnsAI InfoLab AG 0.659 0.659 7.18
FT-llama3 uog-tht AG 0.640 0.640 4.64
baseline_top_5 uis-iai RAG 0.639 0.639 3.23
agtask-bm25-colbert_faiss-gpt4o-llama70b softbank-meisei RAG 0.633 0.633 8.27
UDInfolab.RAG.Query InfoLab AG 0.623 0.623 6.86
webis-rag-run0-taskrag webis AG 0.616 0.616 6.27
buw buw AG 0.597 0.597 8.14
oneshot_post_sentenced buw AG 0.581 0.445 9.09
ruc001 Ruc01 AG 0.577 0.479 14.77
webis-rag-run1-taskrag webis AG 0.531 0.531 6.18
gpt_mini KML RAG 0.494 0.494 7.45
rag_bm25-colbert_faiss-gpt4o-llama70b softbank-meisei AG 0.475 0.475 7.27
ICL-mistral uog-tht AG 0.454 0.345 5.82
UWCgarag WaterlooClarke AG 0.443 0.425 9.95
iiresearch-bm25-top10-llama3-8b-instruct ii_research AG 0.433 0.392 4.14
ginger_top_5 uis-iai RAG 0.419 0.419 4.32
LAS-splade-mxbai-mmr8-RAG ncsu-las AG 0.376 0.270 12.82
BEST_cot_gpt3.5 citi AG 0.329 0.274 6.45
SECOND_cot_gpt3.5 citi AG 0.319 0.229 5.50
ISIR-IRIT-zephyr_query_gen IRIT AG 0.273 0.132 6.95
ielab-b70bf-70bqfs-ad_hoc ielab AG 0.253 0.253 4.59
LAS-splade-mxbai-rrf-mmr8 ncsu-las AG 0.237 0.170 13.41
ISIR-IRIT-zephyr_p2 IRIT AG 0.159 0.112 6.73
qrant_bge_gemini SGU AG 0.157 0.157 6.45
Table 6: Weighted precision and recall scores for the top two runs from each group in TREC 2024
RAG Track under the manual from-scratch condition on 22 topics evaluated by the GPT-4o judge .
#Sentences denotes the average number of sentences in the participant’s submitted answer.
14

Run ID Group Task Weighted Precision Weighted Recall #Sentences
ag_rag_gpt35_expansion_rrf_20 IITD-IRL RAG 0.793 0.793 4.79
Enhanced_Iterative_Fact_Refinement_and_Prioritization TREMA-UNH RAG 0.773 0.773 14.5
Ranked_Iterative_Fact_Extraction_and_Refinement TREMA-UNH RAG 0.772 0.772 14.07
baseline_frag_rag24.test_command-r-plus_top20 coordinators AG 0.760 0.683 12.29
neurag neu AG 0.735 0.442 15.71
listgalore_gpt4o_ragnarokv4_top20 h2oloo AG 0.734 0.575 12.21
baseline_frag_rag24.test_gpt-4o_top20 coordinators AG 0.726 0.404 14.64
UWCrag WaterlooClarke AG 0.724 0.724 6.64
zeph_test_rag_rrf_expand_query IITD-IRL AG 0.709 0.709 4.07
dilab_repllama_listt5_pass3_gpt4o ldisnu AG 0.709 0.418 12.0
listgalore_l31-70b_ragnarokv4_top20 h2oloo AG 0.679 0.593 9.57
baseline_rag24.test_l31_70b_instruct_top20 coordinators RAG 0.668 0.660 7.5
neuragfix neu AG 0.666 0.398 15.71
iiia_standard_p1_straight_ag IIIA-UNIPD RAG 0.662 0.602 4.93
ielab-b70bf-70bqp-70bafs ielab AG 0.658 0.571 2.86
UDInfolab.RAG.Query InfoLab AG 0.654 0.654 7.71
iiia_dedup_p1_straight_ag IIIA-UNIPD RAG 0.650 0.634 4.43
ldilab_gpt_4o ldisnu RAG 0.646 0.363 12.79
cir_gpt-4o-mini_no_reranking_50_0.5_100_301_p1 CIR RAG 0.644 0.644 6.86
UDInfolab.RAG.AnsAI InfoLab AG 0.639 0.639 7.43
webis-rag-run0-taskrag webis AG 0.611 0.611 5.21
baseline_top_5 uis-iai RAG 0.571 0.571 3.07
FT-llama3 uog-tht AG 0.557 0.557 3.29
cohere+post_processing KML RAG 0.556 0.556 18.79
cir_gpt-4o-mini_Cosine_50_0.5_100_301_p1 CIR RAG 0.538 0.538 7.29
agtask-bm25-colbert_faiss-gpt4o-llama70b softbank-meisei RAG 0.526 0.526 7.43
ruc001 Ruc01 AG 0.516 0.458 11.21
oneshot_post_sentenced buw AG 0.460 0.357 12.0
buw buw AG 0.459 0.459 9.5
gpt_mini KML RAG 0.417 0.417 8.36
ginger_top_5 uis-iai RAG 0.413 0.413 3.79
rag_bm25-colbert_faiss-gpt4o-llama70b softbank-meisei AG 0.398 0.398 7.29
webis-rag-run1-taskrag webis AG 0.356 0.356 5.57
ielab-b70bf-70bqfs-ad_hoc ielab AG 0.352 0.350 4.71
LAS-splade-mxbai-mmr8-RAG ncsu-las AG 0.322 0.241 11.71
iiresearch-bm25-top10-llama3-8b-instruct ii_research AG 0.321 0.321 1.86
SECOND_cot_gpt3.5 citi AG 0.287 0.224 5.14
ICL-mistral uog-tht AG 0.250 0.196 4.29
UWCgarag WaterlooClarke AG 0.237 0.201 9.29
BEST_cot_gpt3.5 citi AG 0.216 0.170 3.71
LAS-splade-mxbai-rrf-mmr8 ncsu-las AG 0.190 0.144 12.21
qrant_bge_gemini SGU AG 0.147 0.147 5.79
ISIR-IRIT-zephyr_p2 IRIT AG 0.143 0.047 2.86
ISIR-IRIT-zephyr_query_gen IRIT AG 0.143 0.055 1.79
webis-manual webis AG 0.106 0.075 1.71
Table 7: Weighted precision and recall scores for the top two runs from each group in TREC 2024
RAG Track under the manual with post-editing condition on 14 topics evaluated by human judges
provided by NIST. #Sentences denotes the average number of sentences in the participant’s submitted
answer.
15

Run ID Group Task Weighted Precision Weighted Recall #Sentences
Ranked_Iterative_Fact_Extraction_and_Refinement TREMA-UNH RAG 0.900 0.900 14.07
Enhanced_Iterative_Fact_Refinement_and_Prioritization TREMA-UNH RAG 0.885 0.885 14.5
zeph_test_rag_rrf_expand_query IITD-IRL AG 0.852 0.852 4.07
ldilab_gpt_4o ldisnu RAG 0.842 0.482 12.79
ag_rag_gpt35_expansion_rrf_20 IITD-IRL RAG 0.838 0.838 4.79
dilab_repllama_listt5_pass3_gpt4o ldisnu AG 0.831 0.474 12.0
UWCrag WaterlooClarke AG 0.817 0.817 6.64
listgalore_l31-70b_ragnarokv4_top20 h2oloo AG 0.807 0.687 9.57
baseline_frag_rag24.test_command-r-plus_top20 coordinators AG 0.805 0.724 12.29
iiia_dedup_p1_straight_ag IIIA-UNIPD RAG 0.783 0.740 4.43
baseline_frag_rag24.test_gpt-4o_top20 coordinators AG 0.782 0.427 14.64
neuragfix neu AG 0.775 0.464 15.71
listgalore_gpt4o_ragnarokv4_top20 h2oloo AG 0.775 0.606 12.21
cir_gpt-4o-mini_Cosine_50_0.5_100_301_p1 CIR RAG 0.773 0.773 7.29
webis-manual webis AG 0.770 0.436 12.64
neurag neu AG 0.762 0.455 15.71
cir_gpt-4o-mini_no_reranking_50_0.5_100_301_p1 CIR RAG 0.750 0.750 6.86
UDInfolab.RAG.Query InfoLab AG 0.731 0.731 7.71
baseline_rag24.test_l31_70b_instruct_top20 coordinators RAG 0.728 0.719 7.5
ielab-b70bf-70bqp-70bafs ielab AG 0.723 0.620 2.93
UDInfolab.RAG.AnsAI InfoLab AG 0.717 0.717 7.43
iiia_standard_p1_straight_ag IIIA-UNIPD RAG 0.690 0.621 4.93
agtask-bm25-colbert_faiss-gpt4o-llama70b softbank-meisei RAG 0.687 0.687 7.43
webis-rag-run0-taskrag webis AG 0.682 0.682 5.21
cohere+post_processing KML RAG 0.682 0.682 18.79
baseline_top_5 uis-iai RAG 0.601 0.601 3.07
FT-llama3 uog-tht AG 0.574 0.574 3.29
oneshot_post_sentenced buw AG 0.571 0.442 12.0
buw buw AG 0.558 0.558 9.5
ruc001 Ruc01 AG 0.548 0.482 11.29
webis-rag-run1-taskrag webis AG 0.487 0.487 5.57
gpt_mini KML RAG 0.471 0.440 8.36
rag_bm25-colbert_faiss-gpt4o-llama70b softbank-meisei AG 0.435 0.435 7.29
ginger_top_5 uis-iai RAG 0.408 0.408 3.79
UWCgarag WaterlooClarke AG 0.387 0.324 9.29
SECOND_cot_gpt3.5 citi AG 0.369 0.297 5.5
ICL-mistral uog-tht AG 0.357 0.268 4.29
LAS-splade-mxbai-mmr8-RAG ncsu-las AG 0.347 0.261 11.71
ielab-b70bf-70bqfs-ad_hoc ielab AG 0.297 0.295 4.71
iiresearch-bm25-top10-llama3-8b-instruct ii_research AG 0.296 0.296 1.86
ISIR-IRIT-zephyr_p2 IRIT AG 0.243 0.074 6.36
BEST_cot_gpt3.5 citi AG 0.227 0.161 5.5
LAS-splade-mxbai-rrf-mmr8 ncsu-las AG 0.195 0.150 12.21
ISIR-IRIT-zephyr_query_gen IRIT AG 0.179 0.053 5.36
qrant_bge_gemini SGU AG 0.149 0.149 5.79
Table 8: Weighted precision and recall scores for the top two runs from each group in TREC 2024
RAG Track under the manual with post-editing condition on 14 topics evaluated by the GPT-4o
judge . #Sentences denotes the average number of sentences in the participant’s submitted answer.
16