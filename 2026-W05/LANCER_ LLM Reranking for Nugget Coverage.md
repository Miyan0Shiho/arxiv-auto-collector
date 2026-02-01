# LANCER: LLM Reranking for Nugget Coverage

**Authors**: Jia-Huei Ju, François G. Landry, Eugene Yang, Suzan Verberne, Andrew Yates

**Published**: 2026-01-29 17:16:17

**PDF URL**: [https://arxiv.org/pdf/2601.22008v1](https://arxiv.org/pdf/2601.22008v1)

## Abstract
Unlike short-form retrieval-augmented generation (RAG), such as factoid question answering, long-form RAG requires retrieval to provide documents covering a wide range of relevant information. Automated report generation exemplifies this setting: it requires not only relevant information but also a more elaborate response with comprehensive information. Yet, existing retrieval methods are primarily optimized for relevance ranking rather than information coverage. To address this limitation, we propose LANCER, an LLM-based reranking method for nugget coverage. LANCER predicts what sub-questions should be answered to satisfy an information need, predicts which documents answer these sub-questions, and reranks documents in order to provide a ranked list covering as many information nuggets as possible. Our empirical results show that LANCER enhances the quality of retrieval as measured by nugget coverage metrics, achieving higher $α$-nDCG and information coverage than other LLM-based reranking methods. Our oracle analysis further reveals that sub-question generation plays an essential role.

## Full Text


<!-- PDF content starts -->

LANCER: LLM Reranking for Nugget Coverage
Jia-Huei Ju1, François G. Landry2, Eugene Yang3,
Suzan Verberne4, and Andrew Yates3
1University of Amsterdam, The Netherlands
j.ju@uva.nl
2Université de Moncton, Canada
efl7126@umoncton.ca
3Johns Hopkins University, USA
{eugene.yang,andrew.yates}@jhu.edu
4Leiden University, The Netherlands
s.verberne@liacs.leidenuniv.nl
Abstract.Unlikeshort-formretrieval-augmentedgeneration(RAG),such
as factoid question answering, long-form RAG requires retrieval to pro-
vide documents covering a wide range of relevant information. Auto-
mated report generation exemplifies this setting: it requires not only
relevant information but also a more elaborate response with comprehen-
sive information. Yet, existing retrieval methods are primarily optimized
for relevance ranking rather than information coverage. To address this
limitation, we propose LANCER,5an LLM-based reranking method for
nugget coverage. LANCER predicts what sub-questions should be an-
swered to satisfy an information need, predicts which documents answer
these sub-questions, and reranks documents in order to provide a ranked
list covering as many information nuggets as possible. Our empirical re-
sults show that LANCER enhances the quality of retrieval as measured
by nugget coverage metrics, achieving higherα-nDCG and information
coverage than other LLM-based reranking methods. Our oracle analysis
further reveals that sub-question generation plays an essential role.
Keywords:LLM Reranking·RAG·Nugget Coverage·Report Generation·
Multi-facet Retrieval
1 Introduction
Long-form RAG has introduced a new frontier for information-seeking. Com-
pared to the traditional search paradigm, users can now ask LLMs to organize
information from retrieved documents. In this specialized generation setting,
retrieval becomes crucial, because it determines the finite scope of information
available for the generator to incorporate [41]. For instance, the TREC NeuCLIR
track’s report generation task involves open-ended, multi-faceted information
needs that are addressed by generating a comprehensive report describing and
5https://github.com/DylanJoo/LANCERarXiv:2601.22008v1  [cs.IR]  29 Jan 2026

2 Ju. et al.
citing relevant information in the corpus [7]. This is a challenging task that re-
quires the retrieval component to retrieve documents covering all facets of the
information need, so that they can be cited in the report [14].
With these emerging use cases, there has been renewed interest in nugget-
based evaluation approaches that consider what fine-grained information is pro-
vided by documents rather than considering only document-level relevance [8,
33, 35, 44]. These approaches naturally align with the notion of information
coverage [15, 32]:how well does a retrieved context cover the required relevant
facts?Therefore, nugget coverage has become a key retrieval-sensitive criterion
in long-form report generation tasks [27]. In practice, however, the retrieved
context often includes irrelevant and redundant information, limiting the infor-
mation that the generator can use and wasting some of the generator’s limited
input context [18].
Existing retrieval approaches are not particularly designed for coverage [18].
Neural retrieval and reranking models are typically trained to predict relevance
rather than to consider nugget coverage of the retrieved context. While listwise
rerankers can consider interactions between documents in principle, this direc-
tion is underexplored and all state-of-the-art listwise rerankers are optimized
for relevance ranking (i.e., they are trained to find relevant documents, not to
find a set of documents that covers all aspects of an information need) [12, 34].
Other approaches like bi-encoders and pointwise rerankers predict the relevance
of each document independently, preventing them from considering nugget cov-
erage across a set of documents. Optimizing nugget coverage is closely related
to diversification, which has been studied in the past, but is not the goal of any
state-of-the-art ranking methods. For example, ranking for diversification was
explored using pre-neural methods [3, 38], whereas generating query intents for
diversification was considered with early transformer methods [26]. Motivated
by these limitations, we propose a reranking method aimed at improving nugget
coverage and explore its performance on collections with fine-grained nugget
judgments.
We introduce LANCER, anLLM rerAnking method forNuggetCovERage,
which aims to rerank documents in order to improve their nugget coverage at
a shallow cutoff. As illustrated in Figure 1, LANCER has three stages: (i) syn-
thetic sub-question generation, (ii) document answerability judgment, and (iii)
coverage-based aggregation. LANCER uses an LLM to generate sub-questions
thatshouldbeansweredinordertosatisfyaninformationneed,predictswhether
the documents from first-stage retrieval answer these sub-questions, and then
uses these predictions to produce a reranked list that aims to cover as many
information nuggets as possible.
In our empirical evaluation on two datasets with nugget-level judgments,
LANCER improves the coverage of the retrieved documents and can outperform
other LLM-based reranking methods optimized for relevance [42, 53]. Moreover,
LANCER offers the advantage of transparency; the synthetic sub-questions and
their answerability scores provide an explicit trace of what facets of information
have been collected or missed. In addition, providing LANCER with oracle sub-

LANCER: LLM Reranking for Nugget Coverage 3
questions substantially increases performance further, demonstrating that opti-
mizing for coverage can yield significant benefits and highlighting the quality of
sub-questions as one of the important areas to improve in the future. We also
study the impact of the parameters under different settings, providing insights
into the sub-question generation and coverage-based aggregation strategies.
2 Related Work
Initial RAG studies [17, 22] have shown that retrieval can supply relevant in-
formation as a source of complementary knowledge for language models [39].
Subsequent works have further applied it on a wide range of real-world appli-
cations, e.g. [19, 41]. Among them, automated report generation has unique
demands for retrieval: it requires the retrieved context to be not only relevant
but to comprehensively identify relevant documents, so the generated report can
provide all relevant information in the corpus. This distinction diverges from the
traditional relevance-based retrieval for short-form QA tasks, where information
needs are clear and narrow.
To support the development of long-form RAG systems, many recent studies
have revisited nugget-based evaluation [33]. A nugget represents a standalone
fact, which was first introduced for evaluating definition question answering [45]
with nugget recall being the primary metric [44]. The concept has been further
extended to measure coverage in summarization tasks [11, 15, 32]. Together,
nugget and coverage collectively align with the goal of long-form RAG report
generation [27], imposing additional coverage-based criteria on retrieval and the
generated report [7, 18].
However, most existing first-stage retrieval methods, instead of optimizing
for coverage, are optimized solely for document relevance [43], favoring rele-
vant documents with common nuggets [18]. As zero-shot re-rankers, LLMs have
shown their adaptability across different ranking paradigms, including point-
wise [30, 37], pairwise [36], listwise [25, 42], and setwise [53]. Yet each has its
own drawbacks. Pointwise treats documents independently and omits relation-
shipsamongredundantdocuments.Whiletheothersoftenfocusontherelevance
aspect, lacking consideration of covering more nuggets for the downstream gen-
eration.
Though coverage-based retrieval methods remain underexplored, in a similar
vein, many studies have proposed to diversify the retrieved results [4, 13, 38, 40],
aiming at tackling the trade-off between diversity and relevance [3, 5]. Recent
studies use LLMs to generate sub-queries [23, 50] for increasing recall or in-
tents [26] for diversification. The research most closely related to ours is done
by Guo et al. [16], which improves pointwise reranking with multiple criteria.
Our work complements them by explicitly identifying nuggets and optimizing
coverage for long-form RAG.

4 Ju. et al.
Synthetic Sub-Questions <latexit sha1_base64="E/adLWm4mrHabkAXOekC3xWo2DA=">AAAB/XicbVDLSgMxFL3js9ZXfezcBIvgopSZUtRl0Y3LCvYB7TBk0kwbmslMk4xQS/FX3LhQxK3/4c6/MdPOQlsPBA7n3MO9OX7MmdK2/W2trK6tb2zmtvLbO7t7+4WDw6aKEklog0Q8km0fK8qZoA3NNKftWFIc+py2/OFN6rceqFQsEvd6HFM3xH3BAkawNpJXOB55TgmNvEoJdXuRVik3ctEu2zOgZeJkpAgZ6l7hy4RJElKhCcdKdRw71u4ES80Ip9N8N1E0xmSI+7RjqMAhVe5kdv0UnRmlh4JImic0mqm/ExMcKjUOfTMZYj1Qi14q/ud1Eh1cuRMm4kRTQeaLgoQjHaG0CtRjkhLNx4ZgIpm5FZEBlphoU1jelOAsfnmZNCtl56JcvasWa9dZHTk4gVM4BwcuoQa3UIcGEHiEZ3iFN+vJerHerY/56IqVZY7gD6zPHxQkk7c=</latexit>q1,q2,...,qn
<latexit sha1_base64="E/adLWm4mrHabkAXOekC3xWo2DA=">AAAB/XicbVDLSgMxFL3js9ZXfezcBIvgopSZUtRl0Y3LCvYB7TBk0kwbmslMk4xQS/FX3LhQxK3/4c6/MdPOQlsPBA7n3MO9OX7MmdK2/W2trK6tb2zmtvLbO7t7+4WDw6aKEklog0Q8km0fK8qZoA3NNKftWFIc+py2/OFN6rceqFQsEvd6HFM3xH3BAkawNpJXOB55TgmNvEoJdXuRVik3ctEu2zOgZeJkpAgZ6l7hy4RJElKhCcdKdRw71u4ES80Ip9N8N1E0xmSI+7RjqMAhVe5kdv0UnRmlh4JImic0mqm/ExMcKjUOfTMZYj1Qi14q/ud1Eh1cuRMm4kRTQeaLgoQjHaG0CtRjkhLNx4ZgIpm5FZEBlphoU1jelOAsfnmZNCtl56JcvasWa9dZHTk4gVM4BwcuoQa3UIcGEHiEZ3iFN+vJerHerY/56IqVZY7gD6zPHxQkk7c=</latexit>q1,q2,...,qnFirst-stage Retrieved Documents <latexit sha1_base64="Ub2fGO+I0YsIJTSVbkMrvAdocIQ=">AAAB9HicbVBNS8NAEJ3Ur1q/qh69LBbBU0lEqjeLvXisYD+gCWWz2bRLN5u4uymU0N/hxYMiXv0x3vwp3tymPWjrg4HHezPMzPMTzpS27S+rsLa+sblV3C7t7O7tH5QPj9oqTiWhLRLzWHZ9rChngrY005x2E0lx5HPa8UeNmd8ZU6lYLB70JKFehAeChYxgbSQvQC4TyPWxzBrTfrliV+0caJU4C1K5+Q5zNPvlTzeISRpRoQnHSvUcO9FehqVmhNNpyU0VTTAZ4QHtGSpwRJWX5UdP0ZlRAhTG0pTQKFd/T2Q4UmoS+aYzwnqolr2Z+J/XS3V47WVMJKmmgswXhSlHOkazBFDAJCWaTwzBRDJzKyJDLDHRJqeSCcFZfnmVtC+qTq16eW9X6rcwRxFO4BTOwYErqMMdNKEFBB7hCV7g1Rpbz9ab9T5vLViLmWP4A+vjB+stlUA=</latexit>d2¯C
<latexit sha1_base64="Ub2fGO+I0YsIJTSVbkMrvAdocIQ=">AAAB9HicbVBNS8NAEJ3Ur1q/qh69LBbBU0lEqjeLvXisYD+gCWWz2bRLN5u4uymU0N/hxYMiXv0x3vwp3tymPWjrg4HHezPMzPMTzpS27S+rsLa+sblV3C7t7O7tH5QPj9oqTiWhLRLzWHZ9rChngrY005x2E0lx5HPa8UeNmd8ZU6lYLB70JKFehAeChYxgbSQvQC4TyPWxzBrTfrliV+0caJU4C1K5+Q5zNPvlTzeISRpRoQnHSvUcO9FehqVmhNNpyU0VTTAZ4QHtGSpwRJWX5UdP0ZlRAhTG0pTQKFd/T2Q4UmoS+aYzwnqolr2Z+J/XS3V47WVMJKmmgswXhSlHOkazBFDAJCWaTwzBRDJzKyJDLDHRJqeSCcFZfnmVtC+qTq16eW9X6rcwRxFO4BTOwYErqMMdNKEFBB7hCV7g1Rpbz9ab9T5vLViLmWP4A+vjB+stlUA=</latexit>d2¯C<latexit sha1_base64="tB9SLyBQiR75W0VX0JoF9OB9yuI=">AAAB6nicbVC7SgNBFL3rM8ZXVLCxGQyCVdgVUcsQG8sEzQOSJczOziZDZmeWmVkhLPkEGwtFbG39C7/AzsZvcfIoNPHAhcM593LvPUHCmTau++UsLa+srq3nNvKbW9s7u4W9/YaWqSK0TiSXqhVgTTkTtG6Y4bSVKIrjgNNmMLge+817qjST4s4ME+rHuCdYxAg2VroNu163UHRL7gRokXgzUiwf1r7Ze+Wj2i18dkJJ0pgKQzjWuu25ifEzrAwjnI7ynVTTBJMB7tG2pQLHVPvZ5NQROrFKiCKpbAmDJurviQzHWg/jwHbG2PT1vDcW//PaqYmu/IyJJDVUkOmiKOXISDT+G4VMUWL40BJMFLO3ItLHChNj08nbELz5lxdJ46zkXZTOazaNCkyRgyM4hlPw4BLKcANVqAOBHjzAEzw73Hl0XpzXaeuSM5s5gD9w3n4Azr+RMg==</latexit>d1
<latexit sha1_base64="tB9SLyBQiR75W0VX0JoF9OB9yuI=">AAAB6nicbVC7SgNBFL3rM8ZXVLCxGQyCVdgVUcsQG8sEzQOSJczOziZDZmeWmVkhLPkEGwtFbG39C7/AzsZvcfIoNPHAhcM593LvPUHCmTau++UsLa+srq3nNvKbW9s7u4W9/YaWqSK0TiSXqhVgTTkTtG6Y4bSVKIrjgNNmMLge+817qjST4s4ME+rHuCdYxAg2VroNu163UHRL7gRokXgzUiwf1r7Ze+Wj2i18dkJJ0pgKQzjWuu25ifEzrAwjnI7ynVTTBJMB7tG2pQLHVPvZ5NQROrFKiCKpbAmDJurviQzHWg/jwHbG2PT1vDcW//PaqYmu/IyJJDVUkOmiKOXISDT+G4VMUWL40BJMFLO3ItLHChNj08nbELz5lxdJ46zkXZTOazaNCkyRgyM4hlPw4BLKcANVqAOBHjzAEzw73Hl0XpzXaeuSM5s5gD9w3n4Azr+RMg==</latexit>d1…<latexit sha1_base64="QMv2/gWzBNfvB+bY3jnzWqEPEzk=">AAAB6nicbVBNS8NAEJ3Ur1q/qh69LBbBU0lKUY9FLx4r2lpoQ9lspu3SzSbsboQS+hO8eFDEq7/Im//GbZuDtj4YeLw3w8y8IBFcG9f9dgpr6xubW8Xt0s7u3v5B+fCoreNUMWyxWMSqE1CNgktsGW4EdhKFNAoEPgbjm5n/+IRK81g+mEmCfkSHkg84o8ZK92G/1i9X3Ko7B1klXk4qkKPZL3/1wpilEUrDBNW667mJ8TOqDGcCp6VeqjGhbEyH2LVU0gi1n81PnZIzq4RkECtb0pC5+nsio5HWkyiwnRE1I73szcT/vG5qBld+xmWSGpRssWiQCmJiMvubhFwhM2JiCWWK21sJG1FFmbHplGwI3vLLq6Rdq3oX1fpdvdK4zuMowgmcwjl4cAkNuIUmtIDBEJ7hFd4c4bw4787HorXg5DPH8AfO5w/xm42W</latexit>d2
<latexit sha1_base64="QMv2/gWzBNfvB+bY3jnzWqEPEzk=">AAAB6nicbVBNS8NAEJ3Ur1q/qh69LBbBU0lKUY9FLx4r2lpoQ9lspu3SzSbsboQS+hO8eFDEq7/Im//GbZuDtj4YeLw3w8y8IBFcG9f9dgpr6xubW8Xt0s7u3v5B+fCoreNUMWyxWMSqE1CNgktsGW4EdhKFNAoEPgbjm5n/+IRK81g+mEmCfkSHkg84o8ZK92G/1i9X3Ko7B1klXk4qkKPZL3/1wpilEUrDBNW667mJ8TOqDGcCp6VeqjGhbEyH2LVU0gi1n81PnZIzq4RkECtb0pC5+nsio5HWkyiwnRE1I73szcT/vG5qBld+xmWSGpRssWiQCmJiMvubhFwhM2JiCWWK21sJG1FFmbHplGwI3vLLq6Rdq3oX1fpdvdK4zuMowgmcwjl4cAkNuIUmtIDBEJ7hFd4c4bw4787HorXg5DPH8AfO5w/xm42W</latexit>d2
<latexit sha1_base64="SFrsaEMBH4MF4qMQey9Al3GoplA=">AAAB7nicbVBNS8NAEJ3Ur1q/qh69LBbBU0lE1GPRi8cK9gPaUDabSbt0swm7G6GE/ggvHhTx6u/x5r9x2+agrQ8GHu/NMDMvSAXXxnW/ndLa+sbmVnm7srO7t39QPTxq6yRTDFssEYnqBlSj4BJbhhuB3VQhjQOBnWB8N/M7T6g0T+SjmaTox3QoecQZNVbqhIPcc93poFpz6+4cZJV4BalBgeag+tUPE5bFKA0TVOue56bGz6kynAmcVvqZxpSyMR1iz1JJY9R+Pj93Ss6sEpIoUbakIXP190ROY60ncWA7Y2pGetmbif95vcxEN37OZZoZlGyxKMoEMQmZ/U5CrpAZMbGEMsXtrYSNqKLM2IQqNgRv+eVV0r6oe1f1y4fLWuO2iKMMJ3AK5+DBNTTgHprQAgZjeIZXeHNS58V5dz4WrSWnmDmGP3A+fwCTa48V</latexit>d100
<latexit sha1_base64="SFrsaEMBH4MF4qMQey9Al3GoplA=">AAAB7nicbVBNS8NAEJ3Ur1q/qh69LBbBU0lE1GPRi8cK9gPaUDabSbt0swm7G6GE/ggvHhTx6u/x5r9x2+agrQ8GHu/NMDMvSAXXxnW/ndLa+sbmVnm7srO7t39QPTxq6yRTDFssEYnqBlSj4BJbhhuB3VQhjQOBnWB8N/M7T6g0T+SjmaTox3QoecQZNVbqhIPcc93poFpz6+4cZJV4BalBgeag+tUPE5bFKA0TVOue56bGz6kynAmcVvqZxpSyMR1iz1JJY9R+Pj93Ss6sEpIoUbakIXP190ROY60ncWA7Y2pGetmbif95vcxEN37OZZoZlGyxKMoEMQmZ/U5CrpAZMbGEMsXtrYSNqKLM2IQqNgRv+eVV0r6oe1f1y4fLWuO2iKMMJ3AK5+DBNTTgHprQAgZjeIZXeHNS58V5dz4WrSWnmDmGP3A+fwCTa48V</latexit>d100…
<latexit sha1_base64="3YNU6oG9cIYL6hGxK+R3NsAPyoQ=">AAAB83icbVDLSgNBEOyNrxhfUY9eBoPgKeyKqMegF48RzAOSJcxOZpMhs7PjPMSw5De8eFDEqz/jzb9xkuxBEwsaiqpuursiyZk2vv/tFVZW19Y3ipulre2d3b3y/kFTp1YR2iApT1U7wppyJmjDMMNpWyqKk4jTVjS6mfqtR6o0S8W9GUsaJnggWMwINk7qPqFuKrnV6KEX9MoVv+rPgJZJkJMK5Kj3yl/dfkpsQoUhHGvdCXxpwgwrwwink1LXaioxGeEB7TgqcEJ1mM1unqATp/RRnCpXwqCZ+nsiw4nW4yRynQk2Q73oTcX/vI418VWYMSGtoYLMF8WWI5OiaQCozxQlho8dwUQxdysiQ6wwMS6mkgshWHx5mTTPqsFF9fzuvFK7zuMowhEcwykEcAk1uIU6NICAhGd4hTfPei/eu/cxby14+cwh/IH3+QNfg5FD</latexit>x q1
<latexit sha1_base64="3YNU6oG9cIYL6hGxK+R3NsAPyoQ=">AAAB83icbVDLSgNBEOyNrxhfUY9eBoPgKeyKqMegF48RzAOSJcxOZpMhs7PjPMSw5De8eFDEqz/jzb9xkuxBEwsaiqpuursiyZk2vv/tFVZW19Y3ipulre2d3b3y/kFTp1YR2iApT1U7wppyJmjDMMNpWyqKk4jTVjS6mfqtR6o0S8W9GUsaJnggWMwINk7qPqFuKrnV6KEX9MoVv+rPgJZJkJMK5Kj3yl/dfkpsQoUhHGvdCXxpwgwrwwink1LXaioxGeEB7TgqcEJ1mM1unqATp/RRnCpXwqCZ+nsiw4nW4yRynQk2Q73oTcX/vI418VWYMSGtoYLMF8WWI5OiaQCozxQlho8dwUQxdysiQ6wwMS6mkgshWHx5mTTPqsFF9fzuvFK7zuMowhEcwykEcAk1uIU6NICAhGd4hTfPei/eu/cxby14+cwh/IH3+QNfg5FD</latexit>x q1<latexit sha1_base64="yDi/+EQFnf4vWCB5hkna9GhwPc0=">AAAB83icbVDLSgMxFM3UV62vqks3wSK4KjOlqMuiG5cV7AM6Q8mkmTY0k8Q8xDL0N9y4UMStP+POvzFtZ6GtBy4czrmXe++JJaPa+P63V1hb39jcKm6Xdnb39g/Kh0dtLazCpIUFE6obI00Y5aRlqGGkKxVBacxIJx7fzPzOI1GaCn5vJpJEKRpymlCMjJPCJxgKyayGD/1av1zxq/4ccJUEOamAHM1++SscCGxTwg1mSOte4EsTZUgZihmZlkKriUR4jIak5yhHKdFRNr95Cs+cMoCJUK64gXP190SGUq0naew6U2RGetmbif95PWuSqyijXFpDOF4sSiyDRsBZAHBAFcGGTRxBWFF3K8QjpBA2LqaSCyFYfnmVtGvV4KJav6tXGtd5HEVwAk7BOQjAJWiAW9AELYCBBM/gFbx51nvx3r2PRWvBy2eOwR94nz9hB5FE</latexit>x q2
<latexit sha1_base64="yDi/+EQFnf4vWCB5hkna9GhwPc0=">AAAB83icbVDLSgMxFM3UV62vqks3wSK4KjOlqMuiG5cV7AM6Q8mkmTY0k8Q8xDL0N9y4UMStP+POvzFtZ6GtBy4czrmXe++JJaPa+P63V1hb39jcKm6Xdnb39g/Kh0dtLazCpIUFE6obI00Y5aRlqGGkKxVBacxIJx7fzPzOI1GaCn5vJpJEKRpymlCMjJPCJxgKyayGD/1av1zxq/4ccJUEOamAHM1++SscCGxTwg1mSOte4EsTZUgZihmZlkKriUR4jIak5yhHKdFRNr95Cs+cMoCJUK64gXP190SGUq0naew6U2RGetmbif95PWuSqyijXFpDOF4sSiyDRsBZAHBAFcGGTRxBWFF3K8QjpBA2LqaSCyFYfnmVtGvV4KJav6tXGtd5HEVwAk7BOQjAJWiAW9AELYCBBM/gFbx51nvx3r2PRWvBy2eOwR94nz9hB5FE</latexit>x q2<latexit sha1_base64="ur92iNDlm/1+c8Ha8P88wC8e/Ls=">AAAB83icbVDLSgNBEOyNrxhfUY9eBoPgKeyKqMegF48RzAOSJcxOZpMhs7PjPMSw5De8eFDEqz/jzb9xkuxBEwsaiqpuursiyZk2vv/tFVZW19Y3ipulre2d3b3y/kFTp1YR2iApT1U7wppyJmjDMMNpWyqKk4jTVjS6mfqtR6o0S8W9GUsaJnggWMwINk7qPqFuKrnV6KEneuWKX/VnQMskyEkFctR75a9uPyU2ocIQjrXuBL40YYaVYYTTSalrNZWYjPCAdhwVOKE6zGY3T9CJU/ooTpUrYdBM/T2R4UTrcRK5zgSboV70puJ/Xsea+CrMmJDWUEHmi2LLkUnRNADUZ4oSw8eOYKKYuxWRIVaYGBdTyYUQLL68TJpn1eCien53Xqld53EU4QiO4RQCuIQa3EIdGkBAwjO8wptnvRfv3fuYtxa8fOYQ/sD7/AG795GA</latexit>x qn
<latexit sha1_base64="ur92iNDlm/1+c8Ha8P88wC8e/Ls=">AAAB83icbVDLSgNBEOyNrxhfUY9eBoPgKeyKqMegF48RzAOSJcxOZpMhs7PjPMSw5De8eFDEqz/jzb9xkuxBEwsaiqpuursiyZk2vv/tFVZW19Y3ipulre2d3b3y/kFTp1YR2iApT1U7wppyJmjDMMNpWyqKk4jTVjS6mfqtR6o0S8W9GUsaJnggWMwINk7qPqFuKrnV6KEneuWKX/VnQMskyEkFctR75a9uPyU2ocIQjrXuBL40YYaVYYTTSalrNZWYjPCAdhwVOKE6zGY3T9CJU/ooTpUrYdBM/T2R4UTrcRK5zgSboV70puJ/Xsea+CrMmJDWUEHmi2LLkUnRNADUZ4oSw8eOYKKYuxWRIVaYGBdTyYUQLL68TJpn1eCien53Xqld53EU4QiO4RQCuIQa3EIdGkBAwjO8wptnvRfv3fuYtxa8fOYQ/sD7/AG795GA</latexit>x qn041…………424…53……0Coverage-based Aggregation
<latexit sha1_base64="EH/atNw4vIWFMTE098MHB5IiIiQ=">AAAB7XicbVBNSwMxEJ2tX7V+VT16CRahXsquFPVY9OKxgv2AdinZbLaNzSZLkhVL7X/w4kERr/4fb/4b03YP2vpg4PHeDDPzgoQzbVz328mtrK6tb+Q3C1vbO7t7xf2DppapIrRBJJeqHWBNORO0YZjhtJ0oiuOA01YwvJ76rQeqNJPizowS6se4L1jECDZWaupy+PR42iuW3Io7A1omXkZKkKHeK351Q0nSmApDONa647mJ8cdYGUY4nRS6qaYJJkPcpx1LBY6p9sezayfoxCohiqSyJQyaqb8nxjjWehQHtjPGZqAXvan4n9dJTXTpj5lIUkMFmS+KUo6MRNPXUcgUJYaPLMFEMXsrIgOsMDE2oIINwVt8eZk0zyreeaV6Wy3VrrI48nAEx1AGDy6gBjdQhwYQuIdneIU3RzovzrvzMW/NOdnMIfyB8/kDI/aO2w==</latexit>s(d|x)
<latexit sha1_base64="EH/atNw4vIWFMTE098MHB5IiIiQ=">AAAB7XicbVBNSwMxEJ2tX7V+VT16CRahXsquFPVY9OKxgv2AdinZbLaNzSZLkhVL7X/w4kERr/4fb/4b03YP2vpg4PHeDDPzgoQzbVz328mtrK6tb+Q3C1vbO7t7xf2DppapIrRBJJeqHWBNORO0YZjhtJ0oiuOA01YwvJ76rQeqNJPizowS6se4L1jECDZWaupy+PR42iuW3Io7A1omXkZKkKHeK351Q0nSmApDONa647mJ8cdYGUY4nRS6qaYJJkPcpx1LBY6p9sezayfoxCohiqSyJQyaqb8nxjjWehQHtjPGZqAXvan4n9dJTXTpj5lIUkMFmS+KUo6MRNPXUcgUJYaPLMFEMXsrIgOsMDE2oIINwVt8eZk0zyreeaV6Wy3VrrI48nAEx1AGDy6gBjdQhwYQuIdneIU3RzovzrvzMW/NOdnMIfyB8/kDI/aO2w==</latexit>s(d|x)…1359Report Request <latexit sha1_base64="b2qZ/hsYHIgcRLd+KldYES74moA=">AAAB6HicbVDLSgNBEOyNrxhfUY9eBoPgKexKUI8BPXhMwDwgWcLspDcZMzu7zMyKIeQLvHhQxKuf5M2/cZLsQRMLGoqqbrq7gkRwbVz328mtrW9sbuW3Czu7e/sHxcOjpo5TxbDBYhGrdkA1Ci6xYbgR2E4U0igQ2ApGNzO/9YhK81jem3GCfkQHkoecUWOl+lOvWHLL7hxklXgZKUGGWq/41e3HLI1QGiao1h3PTYw/ocpwJnBa6KYaE8pGdIAdSyWNUPuT+aFTcmaVPgljZUsaMld/T0xopPU4CmxnRM1QL3sz8T+vk5rw2p9wmaQGJVssClNBTExmX5M+V8iMGFtCmeL2VsKGVFFmbDYFG4K3/PIqaV6UvctypV4pVW+zOPJwAqdwDh5cQRXuoAYNYIDwDK/w5jw4L86787FozTnZzDH8gfP5A+nBjQc=</latexit>x<latexit sha1_base64="b2qZ/hsYHIgcRLd+KldYES74moA=">AAAB6HicbVDLSgNBEOyNrxhfUY9eBoPgKexKUI8BPXhMwDwgWcLspDcZMzu7zMyKIeQLvHhQxKuf5M2/cZLsQRMLGoqqbrq7gkRwbVz328mtrW9sbuW3Czu7e/sHxcOjpo5TxbDBYhGrdkA1Ci6xYbgR2E4U0igQ2ApGNzO/9YhK81jem3GCfkQHkoecUWOl+lOvWHLL7hxklXgZKUGGWq/41e3HLI1QGiao1h3PTYw/ocpwJnBa6KYaE8pGdIAdSyWNUPuT+aFTcmaVPgljZUsaMld/T0xopPU4CmxnRM1QL3sz8T+vk5rw2p9wmaQGJVssClNBTExmX5M+V8iMGFtCmeL2VsKGVFFmbDYFG4K3/PIqaV6UvctypV4pVW+zOPJwAqdwDh5cQRXuoAYNYIDwDK/w5jw4L86787FozTnZzDH8gfP5A+nBjQc=</latexit>xAnswerability  Judgements<latexit sha1_base64="f2BFVsr+40flQsthVAiXIUtBKgU=">AAAB83icbVDJSgNBEK2JW0xcoh69NEYhgoQZEfUY9OIxglkgM4Senp6kTc9id08gDPkNLx4U8erVH/APvPkherazHDTxQcHjvSqq6rkxZ1KZ5qeRWVhcWl7Jrubya+sbm4Wt7bqMEkFojUQ8Ek0XS8pZSGuKKU6bsaA4cDltuL3Lkd/oUyFZFN6oQUydAHdC5jOClZZsuypZyTtCd+3bw3ahaJbNMdA8saakWNn/envv57+r7cKH7UUkCWioCMdStiwzVk6KhWKE02HOTiSNMenhDm1pGuKASicd3zxEB1rxkB8JXaFCY/X3RIoDKQeBqzsDrLpy1huJ/3mtRPnnTsrCOFE0JJNFfsKRitAoAOQxQYniA00wEUzfikgXC0yUjimnQ7BmX54n9eOydVo+udZpXMAEWdiFPSiBBWdQgSuoQg0IxHAPj/BkJMaD8Wy8TFozxnRmB/7AeP0BVPSVAw==</latexit> (d, qj)
<latexit sha1_base64="f2BFVsr+40flQsthVAiXIUtBKgU=">AAAB83icbVDJSgNBEK2JW0xcoh69NEYhgoQZEfUY9OIxglkgM4Senp6kTc9id08gDPkNLx4U8erVH/APvPkherazHDTxQcHjvSqq6rkxZ1KZ5qeRWVhcWl7Jrubya+sbm4Wt7bqMEkFojUQ8Ek0XS8pZSGuKKU6bsaA4cDltuL3Lkd/oUyFZFN6oQUydAHdC5jOClZZsuypZyTtCd+3bw3ahaJbNMdA8saakWNn/envv57+r7cKH7UUkCWioCMdStiwzVk6KhWKE02HOTiSNMenhDm1pGuKASicd3zxEB1rxkB8JXaFCY/X3RIoDKQeBqzsDrLpy1huJ/3mtRPnnTsrCOFE0JJNFfsKRitAoAOQxQYniA00wEUzfikgXC0yUjimnQ7BmX54n9eOydVo+udZpXMAEWdiFPSiBBWdQgSuoQg0IxHAPj/BkJMaD8Wy8TFozxnRmB/7AeP0BVPSVAw==</latexit> (d, qj)
<latexit sha1_base64="tB9SLyBQiR75W0VX0JoF9OB9yuI=">AAAB6nicbVC7SgNBFL3rM8ZXVLCxGQyCVdgVUcsQG8sEzQOSJczOziZDZmeWmVkhLPkEGwtFbG39C7/AzsZvcfIoNPHAhcM593LvPUHCmTau++UsLa+srq3nNvKbW9s7u4W9/YaWqSK0TiSXqhVgTTkTtG6Y4bSVKIrjgNNmMLge+817qjST4s4ME+rHuCdYxAg2VroNu163UHRL7gRokXgzUiwf1r7Ze+Wj2i18dkJJ0pgKQzjWuu25ifEzrAwjnI7ynVTTBJMB7tG2pQLHVPvZ5NQROrFKiCKpbAmDJurviQzHWg/jwHbG2PT1vDcW//PaqYmu/IyJJDVUkOmiKOXISDT+G4VMUWL40BJMFLO3ItLHChNj08nbELz5lxdJ46zkXZTOazaNCkyRgyM4hlPw4BLKcANVqAOBHjzAEzw73Hl0XpzXaeuSM5s5gD9w3n4Azr+RMg==</latexit>d1
<latexit sha1_base64="tB9SLyBQiR75W0VX0JoF9OB9yuI=">AAAB6nicbVC7SgNBFL3rM8ZXVLCxGQyCVdgVUcsQG8sEzQOSJczOziZDZmeWmVkhLPkEGwtFbG39C7/AzsZvcfIoNPHAhcM593LvPUHCmTau++UsLa+srq3nNvKbW9s7u4W9/YaWqSK0TiSXqhVgTTkTtG6Y4bSVKIrjgNNmMLge+817qjST4s4ME+rHuCdYxAg2VroNu163UHRL7gRokXgzUiwf1r7Ze+Wj2i18dkJJ0pgKQzjWuu25ifEzrAwjnI7ynVTTBJMB7tG2pQLHVPvZ5NQROrFKiCKpbAmDJurviQzHWg/jwHbG2PT1vDcW//PaqYmu/IyJJDVUkOmiKOXISDT+G4VMUWL40BJMFLO3ItLHChNj08nbELz5lxdJ46zkXZTOazaNCkyRgyM4hlPw4BLKcANVqAOBHjzAEzw73Hl0XpzXaeuSM5s5gD9w3n4Azr+RMg==</latexit>d1
<latexit sha1_base64="U6IYlsL4xKR1llAK+gC8mZQO7CE=">AAAB6nicbVBNS8NAEJ3Ur1q/qh69LBbBU0m0qMeiF48V7Qe0oWw2m3bpZhN2J0Ip/QlePCji1V/kzX/jts1Bqw8GHu/NMDMvSKUw6LpfTmFldW19o7hZ2tre2d0r7x+0TJJpxpsskYnuBNRwKRRvokDJO6nmNA4kbwejm5nffuTaiEQ94DjlfkwHSkSCUbTSfdg/75crbtWdg/wlXk4qkKPRL3/2woRlMVfIJDWm67kp+hOqUTDJp6VeZnhK2YgOeNdSRWNu/Mn81Ck5sUpIokTbUkjm6s+JCY2NGceB7YwpDs2yNxP/87oZRlf+RKg0Q67YYlGUSYIJmf1NQqE5Qzm2hDIt7K2EDammDG06JRuCt/zyX9I6q3oX1dpdrVK/zuMowhEcwyl4cAl1uIUGNIHBAJ7gBV4d6Tw7b877orXg5DOH8AvOxzfzH42X</latexit>d3
<latexit sha1_base64="U6IYlsL4xKR1llAK+gC8mZQO7CE=">AAAB6nicbVBNS8NAEJ3Ur1q/qh69LBbBU0m0qMeiF48V7Qe0oWw2m3bpZhN2J0Ip/QlePCji1V/kzX/jts1Bqw8GHu/NMDMvSKUw6LpfTmFldW19o7hZ2tre2d0r7x+0TJJpxpsskYnuBNRwKRRvokDJO6nmNA4kbwejm5nffuTaiEQ94DjlfkwHSkSCUbTSfdg/75crbtWdg/wlXk4qkKPRL3/2woRlMVfIJDWm67kp+hOqUTDJp6VeZnhK2YgOeNdSRWNu/Mn81Ck5sUpIokTbUkjm6s+JCY2NGceB7YwpDs2yNxP/87oZRlf+RKg0Q67YYlGUSYIJmf1NQqE5Qzm2hDIt7K2EDammDG06JRuCt/zyX9I6q3oX1dpdrVK/zuMowhEcwyl4cAl1uIUGNIHBAJ7gBV4d6Tw7b877orXg5DOH8AvOxzfzH42X</latexit>d3
<latexit sha1_base64="Aw+aOLLEDMRuac/Th/qumtgVlxY=">AAAB6nicbVBNS8NAEJ3Ur1q/qh69LBbBU0lErMeiF48V7Qe0oWw2k3bpZhN2N0Ip/QlePCji1V/kzX/jts1BWx8MPN6bYWZekAqujet+O4W19Y3NreJ2aWd3b/+gfHjU0kmmGDZZIhLVCahGwSU2DTcCO6lCGgcC28Hodua3n1BpnshHM07Rj+lA8ogzaqz0EPZr/XLFrbpzkFXi5aQCORr98lcvTFgWozRMUK27npsaf0KV4UzgtNTLNKaUjegAu5ZKGqP2J/NTp+TMKiGJEmVLGjJXf09MaKz1OA5sZ0zNUC97M/E/r5uZ6NqfcJlmBiVbLIoyQUxCZn+TkCtkRowtoUxxeythQ6ooMzadkg3BW355lbQuqt5V9fL+slK/yeMowgmcwjl4UIM63EEDmsBgAM/wCm+OcF6cd+dj0Vpw8plj+APn8wf5L42b</latexit>d7
<latexit sha1_base64="Aw+aOLLEDMRuac/Th/qumtgVlxY=">AAAB6nicbVBNS8NAEJ3Ur1q/qh69LBbBU0lErMeiF48V7Qe0oWw2k3bpZhN2N0Ip/QlePCji1V/kzX/jts1BWx8MPN6bYWZekAqujet+O4W19Y3NreJ2aWd3b/+gfHjU0kmmGDZZIhLVCahGwSU2DTcCO6lCGgcC28Hodua3n1BpnshHM07Rj+lA8ogzaqz0EPZr/XLFrbpzkFXi5aQCORr98lcvTFgWozRMUK27npsaf0KV4UzgtNTLNKaUjegAu5ZKGqP2J/NTp+TMKiGJEmVLGjJXf09MaKz1OA5sZ0zNUC97M/E/r5uZ6NqfcJlmBiVbLIoyQUxCZn+TkCtkRowtoUxxeythQ6ooMzadkg3BW355lbQuqt5V9fL+slK/yeMowgmcwjl4UIM63EEDmsBgAM/wCm+OcF6cd+dj0Vpw8plj+APn8wf5L42b</latexit>d7………Coverage-based EvalFinalRetrieved Context <latexit sha1_base64="Tjd7J19N44tZCgPNRXGIgLC99Iw=">AAAB7nicbZA9SwNBEIbn/IzxK2ppsxgEq3AnonYGbSwjmA9MjrC32UuW7O0du3NCOPIjbCwUsfX32PlT7NxcUmjiCwsP7zvDzkyQSGHQdb+cpeWV1bX1wkZxc2t7Z7e0t98wcaoZr7NYxroVUMOlULyOAiVvJZrTKJC8GQxvJnnzkWsjYnWPo4T7Ee0rEQpG0VrNHukIRR66pbJbcXORRfBmUL76DnPVuqXPTi9macQVMkmNaXtugn5GNQom+bjYSQ1PKBvSPm9bVDTixs/yccfk2Do9EsbaPoUkd393ZDQyZhQFtjKiODDz2cT8L2unGF76mVBJilyx6UdhKgnGZLI76QnNGcqRBcq0sLMSNqCaMrQXKtojePMrL0LjtOKdV87u3HL1GqYqwCEcwQl4cAFVuIUa1IHBEJ7gBV6dxHl23pz3aemSM+s5gD9yPn4ARTySkg==</latexit>d2Z
<latexit sha1_base64="Tjd7J19N44tZCgPNRXGIgLC99Iw=">AAAB7nicbZA9SwNBEIbn/IzxK2ppsxgEq3AnonYGbSwjmA9MjrC32UuW7O0du3NCOPIjbCwUsfX32PlT7NxcUmjiCwsP7zvDzkyQSGHQdb+cpeWV1bX1wkZxc2t7Z7e0t98wcaoZr7NYxroVUMOlULyOAiVvJZrTKJC8GQxvJnnzkWsjYnWPo4T7Ee0rEQpG0VrNHukIRR66pbJbcXORRfBmUL76DnPVuqXPTi9macQVMkmNaXtugn5GNQom+bjYSQ1PKBvSPm9bVDTixs/yccfk2Do9EsbaPoUkd393ZDQyZhQFtjKiODDz2cT8L2unGF76mVBJilyx6UdhKgnGZLI76QnNGcqRBcq0sLMSNqCaMrQXKtojePMrL0LjtOKdV87u3HL1GqYqwCEcwQl4cAFVuIUa1IHBEJ7gBV6dxHl23pz3aemSM+s5gD9yPn4ARTySkg==</latexit>d2Z
Fig.1: LANCER consists of three stages (blue boxes). The final retrieved context
Zis evaluated with nugget coverage metrics.
3 Preliminaries
In this work, we aim at improving the retrieval module of an automated report
generation system [27], which has two particular characteristics that differ from
typical long-form RAG problems: (i) the input is a nuanced report request with
multiple information needs, and (ii) the expected output consists of sentences
with citations that provide a comprehensive overview of relevant information
found in a document corpusC. Formally, given a report requestx, we define the
entire report generation process as:
y=G(x, Z),whereZ← R(x, ¯C).(1)
Gis a report generator that takes the retrieved contextZas an input for syn-
thesizing the final reporty. We defineZas the retrieved context, which is the
intermediate output from retrieval componentR. Notably, we adopt the two-
stage retrieval pipeline and focus on the second-stage reranking as mentioned
earlier. ¯Cdenotes the top-kdocument candidates (k≪ |C|) retrieved from a
given corpusC.
ToevaluatetheretrievalcomponentRinRAG,weassessboththeintermedi-
ate retrieved contextZand the final generated reporty, representing the direct
and the propagated impact of the retrieval pipeline [10, 18]. Detailed evaluation
setting is depicted in Figure 1 and Section 5.1.
4 Method: LANCER
Inspired by the CRUX framework for automatically judging the information cov-
erage of retrieved documents [18], we adapt CRUX’s steps to perform rerank-
ing by removing its usage of evaluation. Doing so yields LANCER: an LLM
reranking approach for nugget coverage optimization, which aims to increase
the number of nuggets of relevant information covered. As depicted in Figure 1,

LANCER: LLM Reranking for Nugget Coverage 5
Sub-question Generation
Instruction: Given the following report request, write {n} diverse and non-repeating sub-
questions that can help guide the creation of a focused and comprehensive report. The
sub-questions should help break down the topic into key areas that need to be investigated or
explained. Each sub-question should be short (ideally under 20 words) and should focus on a
single aspect or dimension of the report.
Report Request:
{x}
Output format:
- List each sub-question on a new line. Do not number the sub-questions.
- Do not add any comment or explanation.
- Output without adding additional questions after the specified {n}.
Begin with “<START OF LIST>” and, when you are finished, output “<END OF LIST>”.
Never ever add anything else after “<END OF LIST>”, my life depends on it!!!
Now, generate the {n} sub-questions:
Fig.2: Sub-question generation prompt to produce a list of sub-questions.
LANCER consists of three stages: 1)generating synthetic sub-questionsthat
should be answered, 2)generating sub-question answerability judgmentsto pre-
dict to what extent the sub-questions are answered by documents, and 3)per-
forming coverage-based aggregationto rerank documents for coverage.
Synthetic Sub-question Generation.Given a report requestx, we first de-
rivemultipledetailedinformationneedsbygeneratingdiversesub-questionsfrom
the request. We instruct an LLM to generate a set ofnquestions that are benefi-
cial for the downstream report generation task, denoted as{q j}n
j=1. The prompt
we used is shown in Figure 2, where the report request is the only input. Detailed
analysis is reported in Section 5.3.
Answerability Judgments Generation.Once thensub-questions are gen-
erated, we use the LLM to judge whether documents answer each sub-question.
Specifically, we instruct an LLM to judge the answerability of a documentd
given a report requestxconcatenated with the generated sub-questionq j:
rd,qj=Ψ(d, x⊕q j),wherer∈[0,5].(2)
The functionΨindicates the rubric-based LLM document judgment [8, 18],
which produces a rating between scale 0 and 5 using the prompt shown in Fig-
ure 3. Each output rating indicates the answerability of a synthetic sub-question,
which collectively indicate how much each document satisfies the multi-aspect
information needs of the report requestx. The multi-aspect ratings are then
used to rerank documents in the next stage.
Coverage-based Aggregation Strategies.In this step, we use the multi-
aspect ratings to produce a reranked list that optimizes for coverage. To do

6 Ju. et al.
Document Answerability Judgment [8, 18]
Instruction: Determine whether the question can be answered based on the provided context?
Rate the context with on a scale from 0 to 5 according to the guideline below. Do not write
anything except the rating.
Guideline:
5: The context is highly relevant, complete, and accurate.
4: The context is mostly relevant and complete but may have minor gaps or inaccuracies.
3: The context is partially relevant and complete, with noticeable gaps or inaccuracies.
2: The context has limited relevance and completeness, with significant gaps or inaccuracies.
1: The context is minimally relevant or complete, with substantial shortcomings.
0: The context is not relevant or complete at all.
Question: {q}
Context: {c}
Rating:
Fig.3: Rubric-based answerability judgment prompt. The output rating is con-
verted into 0 to 5, and the output with incorrect formats is assigned to 0.
so, we explore several coverage-based aggregation strategies, including simple
summation, rank fusion, and greedy selection.
Summation (sum & sum-τ).A straightforward strategy is to sumnratings to
produce a single score for each documentd:
ssum(d|x) =nX
j=1rd,qj.(3)
In addition, we experiment with hard thresholding: amongnratings, we incor-
porate only the ratings that are greater than or equal to a thresholdτ, denoted
assum-τ.
Reciprocal Rank Fusion (RRF).Each multi-aspect rating can also be viewed
as a separate score, resulting in multiple ranked lists (i.e., one list for each sub-
question).Underthisview,aclearapproachistousereciprocalrankedfusion[6].
The final score of the documentdis thereby obtained fromndistinct rankings
with reciprocal rank normalization:
sRRF(d|x) =nX
j=11
κ+ Rank j(d),(4)
whereRank j(d)indicates different ranks of documentd∈ ¯Csorted using the
answerability of different sub-questionq j. Following common practice [6], we set
κas 60.
Greedy Utility Selection (greedy-sum, greedy-α, & greedy-cov).Instead of naively
aggregating multi-aspect ratings for each document independently, we adopt a

LANCER: LLM Reranking for Nugget Coverage 7
greedy algorithm that iteratively selects the document maximizing some utility
function (e.g.,sum, coverage,α-nDCG). The algorithm begins with an empty
listZ(0). At each stept, we compute the utility of all the remaining document
candidatesd∈ ¯Cand select the one that yields the highest utility gain. The
selected one is then removed from the candidate set and appended to the list
asZ(t). Until the utility gains of every remaining document become zero, the
remaining are then concatenated to the list in descending order of utility.
The utility function takes the document listd∈Z(t)as input. In our exper-
iment, we implement several utility functions for a given document listZ. They
are detailed as follows:
–greedy-sumextends simplesumaggregation in Eq. (3) with first taking the
maximum rating of each sub-question over documents in the list:
Usum(Z) =nX
j=1max
d∈Zrd,qj.
–greedy-αis defined according to the denominator of evaluation metricα-
nDCG [5]. We first obtain the binary weight by applying a thresholdτon
multi-aspect ratings. Then, we compute the ideal discounted cumulative gain
with the penalty factorα, which decays the gain given the counts of docu-
ments that covered the sub-questionq j, denoted asc di′<i,j, formulated as:
Uα(Z) =|Z|X
i=1nX
j=11(rdi,qj≥τ)i−1Y
i′=1(1−α)cdi′,j
.
–greedy-covis derived from the coverage metric (Cov) [18, 48], where we
similarly take the maximum ratings of each sub-question over documents.
Afterwards, we sum the binary weights to get the final coverage:
Ucov(Z) =nX
j=11(max
d∈Zrd,qj≥τ).
These aggregation strategies enable LANCER to rank retrieved documents as a
whole and to provide the refined retrieved context for the generator. Detailed
parameter analysis is reported in Section 5.3.
5 Experiments and Results
5.1 Experimental Setup
Evaluation Datasets.Evaluating information coverage requires nugget-level
judgments,whicharerare.WeevaluateLANCERusingtwolong-formRAGeval-
uationdatasets:theTRECNeuCLIR’24ReportGeneration(NeuCLIR’24Re-
portGen)[7],andtheCRUXmulti-documentsummarywithDUC’04(CRUX-
MDS-DUC’04) [18]. Both evaluation datasets provide multi-faceted report

8 Ju. et al.
Table 1: Dataset statistics.
NeuCLIR’24 ReportGen CRUX-MDS DUC’04
# Requests 19 50
Avg. request length 55.95 48.46
Avg. # nuggets per request 21.84 15
Corpus size 10,038,768 565,015
requests along with corresponding nuggets, which indicate what information
should be provided in the final generated report. For NeuCLIR’24 ReportGen,
we combine the 19 topics that were judged across all three languages as our
test set and use the remaining 3 that are incomplete among the languages (top-
ics 324, 361, and 387) as development topics. In NeuCLIR’24 ReportGen, each
nugget is written in the form of a question with multiple acceptable answers. For
CRUX-MDS-DUC’04, the nuggets are at the question-level and derived from a
human-written summary. Dataset statistics are reported in Table 1.
Evaluation Methods.To evaluate retrieval for long-form RAG, we adopt the
CRUX framework [18] to assess the quality of the intermediate retrieved context
Z. We reportα-nDCG and Coverage (Cov) for information coverage as our
primary metrics. For reference, we also report nDCG and precision (Prec.) to
measure relevance. All metrics are calculated with a rank cutoff at 10, given that
a limited number of documents can typically fit in the input of the downstream
generation models.
First-stage Retrieval.In the following experiments, we adopt a standard
two-stage retrieval pipeline to augment the final retrieved context for the report
generation task, as formulated in Eq. (1). First, we retrieve documents using one
of three first-stage retrieval approaches: BM256, learned sparse retrieval (LSR)
using MILCO [29] or SPLADEv3 [21], and Qwen-3-Embed [49]. NeuCLIR is
a multilingual corpus; we use the official English translation of the corpus for
BM25 and use documents in their source languages for LSR and Qwen3-Embed
since they are natively multilingual models. On NeuCLIR we use the MILCO
multilingual LSR model [29], whereas on DUC we use the English SPLADEv3
LSR model [21]. For each first-stage retrieval setting, we retrieve the top-100
candidate documents using the report request. These candidates are then passed
to different second-stage reranking methods.
Second-stage Reranking.We implement the other LLM reranking meth-
ods as comparable baselines, includingPointwise[31, 52] with the document
relevance estimated via softmax-normalized over “Yes”/“No” token logits,List-
wise[25, 42] with a default window size of 20 and stride of 10, andSetwise
6The parametersk 1, bare set to(1.2,0.75)for the NeuCLIR corpus; and(0.9,0.7)for
the CRUX-MDS corpus.

LANCER: LLM Reranking for Nugget Coverage 9
Table 2: Evaluation results on two datasets. The first column group for each
dataset contains relevance-based metrics, whereas the shaded columns report
our primary coverage-based metrics. All metrics use a cut-off of 10. Bold and
underlined scores denote thebestand second-best results within the same first-
stage retrieval. Superscripts indicate when a metric shows a significant improve-
mentoveranotherapproachaccordingtoapairedt-testasfollows:first-stage(f),
Pointwise(p), Listwise(l), Setwise(s), and LANCER(†).
NeuCLIR’24 ReportGen CRUX-MDS-DUC’04
Relevance CoverageRelevance Coverage
nDCG Prec. α-nDCG CovnDCG Prec. α-nDCG Cov
BM25 67.7 65.3 53.0 64.1 53.0 51.4 44.5 54.1
+ Pointwise89.3 89.5 67.0 72.2 76.1 73.4 58.6 65.6
+ Listwise 86.0 84.7 61.3 69.577.1 74.6 60.3 64.9
+ Setwise 84.2 81.6 64.5 71.3 69.2 64.0 57.8 63.5
+ LANCER 86.2f85.8f65.5fl72.7fls73.8fs72.4fs60.5fs66.4fs
+ LANCER Q∗88.0f85.8f76.7fpls†79.1fpls80.3fpls†76.6fps†73.7fpls†74.6fpls†
LSR 83.1 81.6 62.9 73.7 70.4 68.0 55.8 64.0
+ Pointwise 90.7 90.5 66.4 72.983.1 82.4 63.2 70.6
+ Listwise 92.3 90.5 71.2 74.9 80.6 79.8 61.1 67.1
+ Setwise 91.0 89.5 68.6 72.1 71.3 68.4 58.4 64.8
+ LANCER92.9f91.6f72.4fp77.378.9fs78.0fs63.5fs68.8fls
+ LANCER Q∗90.9f90.5f78.9fpls†81.4fpls†86.3fpls†84.8fls†81.3fpls†79.3fpls†
Qwen3-Embed88.6 86.8 62.7 69.5 75.9 73.8 60.8 66.8
+ Pointwise 85.1 86.3 63.3 69.683.9 83.8 63.0 70.2
+ Listwise 88.486.8 65.1 68.4 81.7 81.4 63.2 67.5
+ Setwise 84.6 80.5 68.5 71.2 75.3 72.4 61.9 66.6
+ LANCER 88.0 85.3 70.7fpl75.3fp80.8fs80.0fs64.4fs68.5fs
+ LANCER Q∗88.6 88.4s78.8fpls†80.8fpls†88.2fpls†86.8fpls†82.9fpls†80.1fpls†
reranking [53] with 5 child nodes and the heap sort algorithm. All the reranking
methods usemeta-llama/Llama-3.3-70B-Instruct[28] with a suitable max-
imum context length.7running on top of vLLM inference infrastructure.8The
temperature is set to 0 for better reproducibility. As a default, we generate 2 sub-
questions9and aggregate answerability ratings withsumstrategy. We explore
the impact of these parameters in Section 5.3.
5.2 Main Results
Table 2 presents our empirical evaluation results on NeuCLIR’24 ReportGen and
CRUXMDS-DUC’04,whereeachblockofrowscorrespondstodistinctfirst-stage
710,240 for Setwise, 20,480 for Listwise, and 8196 for the others.
8https://github.com/vllm-project/vllm
9For CRUX-MDS-DUC’04, we useQwen/Qwen3-Next-80B-A3B-Instructfor generat-
ing sub-questions, to avoid biases due to the fact that this dataset contains data
synthesized by Llama 3.1 [18].

10 Ju. et al.
retrievers. In addition to the targeted coverage-based metrics (α-nDCG@10 and
Cov@10, shown in the last two shaded columns), we also report relevance-based
metrics for reference (the first two columns).
Zero-shot Reranking Comparisons.We compare our proposed LANCER
to three common LLM-based reranking methods:Pointwise[31, 52],List-
wise[25, 42], andSetwisereranking [53]. Improvements are observable across
three different first-stage retrieved candidate environments (different blocks) and
both evaluation datasets, showing the reranking robustness and generalizability.
However, we found that the improvements are relatively minor on CRUX-MDS-
DUC’04 in terms ofCov, where we attribute this to the smaller number of
ground-truth nuggets, limiting the possible number of documents that can be
credited inCov, and, thus, lower scores. These improvements are sometimes sig-
nificant, but it is difficult to reach statistical significance given the small sizes of
the available datasets with nugget-level judgments.
Trade-offBetweenRelevanceandCoverage.Inaddition,weobservetrade-
offsbetweenrerankingforrelevanceandcoverage.Relevance-basedrerankingim-
proves first-stage retrieval in terms of nDCG and precision; however, their gains
on the coverage-based metrics are limited and even slightly decreased when using
strongerfirst-stageretrievers.Forexample,LSRwithSetwisererankingincreases
nDCG and precision but reduces coverage (-2.8), suggesting that reranking for
relevance can filter out irrelevant documents but may fail to pull up documents
that cover different aspects. On the contrary, LANCER achieves better coverage
without trading off much relevance. Using Qwen3-Embed on NeuCLIR’24 Re-
portGen, LANCER outperforms Listwise reranking on Coverage (75.3 vs. 68.4)
but without substantially reducing precision (85.3 vs. 86.8). Listwise reranking
withQwen3-Embed,whileachievingbetterprecision(onlyhigherthanLANCER
by 1.5 points), is 5 points lower inα-nDCG and almost 7 points lower in Cov-
erage. On NeuCLIR ReportGen with LSR as the first stage, LANCER even
exhibits stronger performance on both relevance and coverage effectiveness than
the baselines, showing no trade-off between the two.
Oracle Setting with Ground-truth Sub-questions.To explore the opti-
mal effectiveness of LANCER, we replace thensynthetic sub-questions with
ground-truth nugget questions to remove the noise of sub-question generation
and control other inference settings, including the LLM generation and the rank-
ingoptimizationstrategyassum.ThisoracleconditionisdenotedasLANCER Q∗
and reported in the last row of each block in Table 2. With the ground-truth
nuggetquestions,LANCERachievessubstantiallyhigherα-nDCGandCovcom-
pared to the other methods. This illustrates that the sub-questions are crucial
for optimizing nugget-coverage, echoing previous work on the challenging re-
search directions of nugget generation for RAG [20, 35] and highlighting nugget
generation as a promising direction for improvement.

LANCER: LLM Reranking for Nugget Coverage 11
2 4 6 8 10304050607080Cov@k
BM25
2 4 6 8 10
LSR
2 4 6 8 10
Qwen3
LANCE-RQ*
LANCE-R
Pointwise
Listwise
Setwise
w/o rerank
Fig.4: Coverage (Cov) grows with respect to the top-kcutoff on NeuCLIR’24
ReportGen evaluation data. Each line indicates the retrieved contexts from dif-
ferent retrieval pipelines.
Top-ranking Retrieved Context.We further analyze coverage with different
cutoffs (top-k) from LANCER. Figure 4 shows Coverage@10 at different depths,
reporting the dynamics of how the multi-aspect information needs are satisfied.
Notably, with LSR and Qwen3-Embed as first-stage retrieval, we found that,
afterk= 4, LANCER starts to outperform the other reranking approaches and
consistently improves askincreases. The trend is less consistent with BM25,
though LANCER with oracle nugget questions still eventually achieves similar
Coverage with largerk. This difference may be due to the fact that the retrieved
candidates were harder for any reranking methods to distinguish due to lexi-
cal overlap with the original queries [1]. Nevertheless, LANCER performs well
overall and LANCER with oracle nuggets still consistently outperforms other
methods across different top-kand first-stage retrieval. We therefore set a fu-
ture goal of conducting an in-depth investigation into reducing the gap between
synthesized and ground-truth sub-questions.
Impact of Retrieved Context on Generation.To analyze the downstream
impactofLANCERonthegeneratedreport,weadditionallymeasurethenugget-
coverage of the final RAG resulty. We employGPTResearcher[9],10an open-
source report generation method, and input it with the different retrieved con-
texts to produce final reports. The report nugget-coverage scores are obtained
from Auto-ARGUE [46], an automatic evaluation framework implementing AR-
GUE [27] withLlama-3.3-70B-Instruct. For a fair comparison, we fix the
generation settings and use the same number of top-kretrieved documents11
for all 18 retrieval pipelines (rows in Table 2). We observe a good Spearman
correlation (0.78 and 0.7) between the report nugget-coverage (percentage of the
unique gold nuggets in the generated report) and the two coverage-based evalu-
ation metrics (α-nDCG andCov, respectively). Notably, LANCER with oracle
nugget questions achieves additional +3.5 and +4 nugget-coverage over other
10https://github.com/assafelovic/gpt-researcher
11kis set to 8 and the temperature is set to 0.35.

12 Ju. et al.
657075-nDCG@10
n=1n=2n=3n=4n=5n=7n=107580Cov@10
 First-stage
BM25
LSR
Qwen3
Fig.5: Evaluation results on the NeuCLIR’24 ReportGen with different numbers
of synthetic sub-questions (x-axis). We use thesumstrategy for all the settings.
The colors indicate the three first-stage retrieval.
reranking methods. However, it only changes +1.3 when paired with LSR first-
stage retrieval, which may indicate noise in the downstream generation steps in
incorporating information in retrieved documents due to various known issues
such as positions [24], content [2], and parametric memory [47].
5.3 Parameter Analysis
Number of Sub-questions.Figure 5 shows how varying the number of syn-
thetic sub-questions in the LANCER pipeline affectsα-nDCG andCovon Neu-
CLIR ReportGen. Surprisingly, the results suggest that a few sub-questions (2
or 3) are sufficient, while adding more does not substantially reduce performance
compared ton= 2but only offers a marginal benefit. When using BM25, in-
creasing the number of sub-questions yields a more substantial improvement
compared to the other first-stage retrievers, withα-nDCG@10 increasing asn
increases and the highestCovatn= 7. Diminishing returns and drops in per-
formance may be due to topic drift as the number of sub-questions increases.
However,thisisnotthecasefortheoraclenuggetquestions,whicharemorethan
10 but contribute significant benefits on coverage, indicating more sub-questions
can still be useful if they remain aligned with the original information need. We
leave such question generation for future work aiming to explore more useful
questions for LANCER.
Different Optimization Strategies.In addition, we investigate different
strategies of utilizing multi-aspect ratingsr d,qj. To control the impact of syn-
thetic questions, we adopt the oracle nugget questions to judge answerability
(i.e., LANCER Q∗). Figure 6 shows the results of applying 5 strategies (Sec-
tion 4) with or without thresholdingτ∈[2,5]. We found that thesumstrategy
generallyperformswellonα-nDCG.Incontrast,greedyselections(G.-*)achieve
betterCovat threshold 3 or 4 (τ 3, τ4), which is what it is optimizing for. How-
ever, interestingly, they drop substantially when applying thresholds at 2 or 5.
An exception isgreedy-sum, which combines ratings additively, and thus is less

LANCER: LLM Reranking for Nugget Coverage 13
707580-nDCG@10
sumsum-2
sum-3
sum-4
sum-5
RRF
G.-sumG.-sum-2
G.-sum-3
G.-sum-4
G.-sum-5
G.-Cov-2
G.-Cov-3
G.-Cov-4
G.-Cov-5
G.--2
G.--3
G.--4
G.--5
758085Cov@10
First-stage
BM25
LSR
Qwen3
Fig.6: Evaluation results on the NeuCLIR’24 ReportGen. Thex-axis shows dif-
ferent aggregation strategies. The colors indicate the three first-stage retrieval.
sensitive to thresholding. These empirical results imply that the human’s nugget
identification aligns closer to an LLM answerability judgment of 3 or 4, present-
ing the fact that there is an uncertainty of LLM-judgment especially when the
predicted rating is low. To effectively reduce noise and integrate lower ratings
better in LANCER, we hypothesize incorporating the logit-trick [12, 30] has
potential to address this issue, as evidenced in Zhuang et al. [51] and by the
observed performance of Pointwise reranking in Table 2.
6 Conclusion
In this paper, we propose LANCER, an LLM re-ranking method that targets
nugget-coverage for the retrieval of long-form RAG. As opposed to existing
relevance-basedretrievalapproaches,LANCERgeneratessub-questionsasproxy
nuggets and produces multi-aspect ratings with a coverage-based aggregation.
Empirical evaluation shows that LANCER is able to effectively rank documents
based on nugget coverage without losing the ability to perform relevance rank-
ing, highlighting its suitability as a retrieval method for long-form RAG tasks.
Our analyses further highlights promising directions for future nugget-coverage
optimization, as evidenced by the quality of proxy sub-question and unstable
LLM answerability judgment.
Acknowledgments
This research was supported by the Hybrid Intelligence Center, a 10-year pro-
gram funded by the Dutch Ministry of Education, Culture and Science through
the Netherlands Organisation for Scientific Research, project VI.Vidi.223.166
of the NWO Talent Programme which is (partly) financed by the Dutch Re-
search Council (NWO) and NWO project NWA.1389.20.183. We acknowledge
the Dutch Research Council for awarding this project access to the LUMI super-
computer, owned by the EuroHPC Joint Undertaking, hosted by CSC (Finland)
and the LUMI consortium through project number NWO-2024.050. Views and

14 Ju. et al.
opinions expressed are those of the author(s) only and do not necessarily reflect
those of their respective employers, funders and/or granting authorities.
Disclosure of Interests
The authors have no competing interests to declare that are relevant to the
content of this article.
References
1. Alaofi, M., Thomas, P., Scholer, F., Sanderson, M.: Llms can be fooled into
labelling a document as relevant: best café near me; this paper is perfectly
relevant. In: Proc. of SIGIR-AP, p. 32–41 (2024)
2. Belém, C.G., Pezeshkpour, P., Iso, H., Maekawa, S., Bhutani, N., Hruschka,
E.: From single to multi: How LLMs hallucinate in multi-document summa-
rization. In: Findings of NAACL, pp. 5276–5309 (2025)
3. Carbonell, J., Goldstein, J.: The use of MMR, diversity-based reranking for
reordering documents and producing summaries. In: Proc. of SIGIR, pp.
335–336 (1998)
4. Chen, H.T., Choi, E.: Open-world evaluation for retrieving diverse perspec-
tives. arXiv [cs.CL] (2024)
5. Clarke, C.L.A., Kolla, M., Cormack, G.V., Vechtomova, O., Ashkan, A.,
Büttcher, S., MacKinnon, I.: Novelty and diversity in information retrieval
evaluation. In: Proc. of SIGIR, p. 659–666 (2008)
6. Cormack, G.V., Clarke, C.L., Buettcher, S.: Reciprocal rank fusion outper-
forms condorcet and individual rank learning methods. In: Procs. of SIGIR,
pp. 758–759 (2009)
7. Dawn, L., Sean, M., James, M., Paul, M., Douglas, W.O., Luca, S., Eugene,
Y.: Overview of the TREC 2024 NeuCLIR track. arXiv [cs.IR] (2025)
8. Dietz, L.: A workbench for autograding retrieve/generate systems. In: Proc.
of SIGIR, pp. 1963–1972 (2024)
9. Duh, K., Yang, E., Weller, O., Yates, A., Lawrie, D.: Hltcoe at liverag: Gpt-
researcher using colbert retrieval (2025)
10. Es, S., James, J., Espinosa-Anke, L., Schockaert, S.: RAGAs: Automated
evaluation of Retrieval Augmented Generation. Proc. of EACL pp. 150–158
(2023)
11. Fabbri, A., Li, I., She, T., Li, S., Radev, D.: Multi-News: A Large-
Scale Multi-Document Summarization Dataset and Abstractive Hierarchical
Model. In: Proc. of ACL, pp. 1074–1084 (2019)
12. Gangi Reddy, R., Doo, J., Xu, Y., Sultan, M.A., Swain, D., Sil, A., Ji, H.:
FIRST: Faster improved listwise reranking with single token decoding. In:
Proc. of EMNLP, pp. 8642–8652 (2024)
13. Gao, H., Zhang, Y.: VRSD: Rethinking similarity and diversity for retrieval
in Large Language Models. arXiv [cs.IR] (2024)

LANCER: LLM Reranking for Nugget Coverage 15
14. Gao, T., Yen, H., Yu, J., Chen, D.: Enabling large language models to gen-
erate text with citations. In: Proc. of EMNLP, pp. 6465–6488 (2023)
15. Grusky, M., Naaman, M., Artzi, Y.: Newsroom: A dataset of 1.3 million
summaries with diverse extractive strategies. In: Proc. of NAACL-HLT, pp.
708–719 (2018)
16. Guo, F., Li, W., Zhuang, H., Luo, Y., Li, Y., Yan, L., Zhu, Q., Zhang, Y.:
Mcranker: Generating diverse criteria on-the-fly to improve pointwise llm
rankers. In: Proc. of WSDM, p. 944–953 (2025)
17. Guu, K., Lee, K., Tung, Z., Pasupat, P., Chang, M.W.: Realm: retrieval-
augmented language model pre-training. In: Proc. of ICML (2020)
18. Ju, J.H., Verberne, S., de Rijke, M., Yates, A.: Controlled retrieval-
augmented context evaluation for long-form RAG. In: Findings of EMNLP,
pp. 21102–21121 (2025)
19. Kwiatkowski, T., Palomaki, J., Redfield, O., Collins, M., Parikh, A., Alberti,
C., Epstein, D., Polosukhin, I., Devlin, J., Lee, K., Toutanova, K., Jones,
L., Kelcey, M., Chang, M.W., Dai, A.M., Uszkoreit, J., Le, Q., Petrov, S.:
Natural Questions: A benchmark for question answering research. TACL7,
453–466 (2019)
20. Łajewska, W., Balog, K.: Ginger: Grounded information nugget-based gen-
eration of responses. In: Proc. of SIGIR, p. 2723–2727 (2025)
21. Lassance, C., Déjean, H., Formal, T., Clinchant, S.: SPLADE-v3: New base-
lines for SPLADE. arXiv [cs.IR] (2024)
22. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N.,
Küttler, H., Lewis, M., Yih, W.t., Rocktäschel, T., Riedel, S., Kiela, D.:
Retrieval-augmented generation for knowledge-intensive nlp tasks. In: Proc.
of NIPS (2020)
23. Li, Z., Wang, J., Jiang, Z., Mao, H., Chen, Z., Du, J., Zhang, Y., Zhang, F.,
Zhang, D., Liu, Y.: DMQR-RAG: Diverse Multi-Query Rewriting for RAG.
arXiv [cs.IR] (2024)
24. Liu, N.F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F.,
Liang, P.: Lost in the middle: How language models use long contexts. TACL
12, 157–173 (2024)
25. Ma, X., Zhang, X., Pradeep, R., Lin, J.: Zero-shot listwise document rerank-
ing with a Large Language Model. arXiv [cs.IR] (2023)
26. MacAvaney, S., Macdonald, C., Murray-Smith, R., Ounis, I.: Intent5: Search
result diversification using causal language models (2021)
27. Mayfield, J., Yang, E., Lawrie, D., MacAvaney, S., McNamee, P., Oard,
D.W., Soldaini, L., Soboroff, I., Weller, O., Kayi, E., Sanders, K., Mason,
M., Hibbler, N.: On the evaluation of machine-generated reports. In: Proc.
of SIGIR, pp. 1904–1915 (2024)
28. MetaAI: The Llama 3 herd of models. arXiv [cs.AI] (2024)
29. Nguyen, T., Lei, Y., Ju, J.H., Yang, E., Yates, A.: Milco: Learned sparse
retrieval across languages via a multilingual connector. arXiv [cs.IR] (2025)
30. Nogueira, R., Jiang, Z., Pradeep, R., Lin, J.: Document ranking with a
pretrained sequence-to-sequence model. In: Findings of EMNLP (2020)

16 Ju. et al.
31. Nogueira, R., Yang, W., Cho, K., Lin, J.: Multi-stage document ranking
with BERT. arXiv [cs.IR] (2019)
32. Over, P., Yen, J.: An introduction to DUC-2004. National Institute of Stan-
dards and Technology (2004)
33. Pavlu, V., Rajput, S., Golbus, P.B., Aslam, J.A.: IR system evaluation using
nugget-based test collections. In: Proc. of WSDM, pp. 393–402 (2012)
34. Pradeep, R., Sharifymoghaddam, S., Lin, J.: RankZephyr: Effective and ro-
bust zero-shot listwise reranking is a breeze! arXiv [cs.IR] (2023)
35. Pradeep, R., Thakur, N., Upadhyay, S., Campos, D., Craswell, N., Lin, J.:
Initial nugget evaluation results for the TREC 2024 RAG Track with the
AutoNuggetizer framework. arXiv [cs.IR] (2024)
36. Qin, Z., Jagerman, R., Hui, K., Zhuang, H., Wu, J., Yan, L., Shen, J., Liu,
T., Liu, J., Metzler, D., Wang, X., Bendersky, M.: Large language models
are effective text rankers with pairwise ranking prompting. In: Findings of
NAACL, pp. 1504–1518 (2024)
37. Sachan, D., Lewis, M., Joshi, M., Aghajanyan, A., Yih, W.t., Pineau, J.,
Zettlemoyer, L.: Improving passage retrieval with zero-shot question gener-
ation. In: Proc. of EMNLP, pp. 3781–3797 (2022)
38. Santos, R.L.T., Macdonald, C., Ounis, I.: A survey of query auto completion
in information retrieval. Foundations and Trends in Information Retrieval
9(1), 1–90 (2015)
39. Shi, W., Min, S., Yasunaga, M., Seo, M., James, R., Lewis, M., Zettlemoyer,
L., Yih, W.t.: REPLUG: Retrieval-augmented black-box language models.
In: Proc. of NAACL-HLT, pp. 8371–8384 (2024)
40. Soboroff,I.,Harman,D.:Noveltydetection:TheTRECexperience.In:Proc.
of EMNLP-HLT, pp. 105–112 (2005)
41. Stelmakh,I.,Luan,Y.,Dhingra,B.,Chang,M.W.:ASQA:Factoidquestions
meet long-form answers. In: Proc. of EMNLP, pp. 8273–8288 (2022)
42. Sun, W., Yan, L., Ma, X., Wang, S., Ren, P., Chen, Z., Yin, D., Ren, Z.: Is
ChatGPT good at search? Investigating large language models as re-ranking
agents. In: Proc. of EMNLP, pp. 14918–14937 (2023)
43. Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., Gurevych, I.: BEIR: A
heterogeneous benchmark for zero-shot evaluation of information retrieval
models. In: Proc. of NeurIPS (2021)
44. Voorhees, E.: Overview of the TREC 2003 Question Answering Track (2004)
45. Voorhees, E.M.: Evaluating answers to definition questions. In: Proc. of
NAACL-HLT (2003)
46. William, W., Marc, M., Orion, W., Laura, D., Hannah, R., Bryan, L., Liu,
G.K.M., Yu, H., James, M., Eugene, Y.: Auto-ARGUE: LLM-Based Report
Generation Evaluation. arXiv [cs.IR] (2025)
47. Xie, J., Zhang, K., Chen, J., Lou, R., Su, Y.: Adaptive chameleon or stub-
born sloth: Revealing the behavior of large language models in knowledge
conflicts. In: Proc. of ICLR (2024)
48. Zhai, C.X., Cohen, W.W., Lafferty, J.: Beyond independent relevance: meth-
odsandevaluationmetricsforsubtopicretrieval.In:Proc.ofSIGIR,p.10–17
(2003)

LANCER: LLM Reranking for Nugget Coverage 17
49. Zhang, Y., Li, M., Long, D., Zhang, X., Lin, H., Yang, B., Xie, P., Yang,
A., Liu, D., Lin, J., Huang, F., Zhou, J.: Qwen3 embedding: Advancing
text embedding and reranking through foundation models. arXiv preprint
arXiv:2506.05176 (2025)
50. Zhong, Y., Yang, J., Fan, Y., Guo, J., Su, L., de Rijke, M., Zhang, R., Yin,
D., Cheng, X.: Reasoning-enhanced query understanding through Decom-
position and Interpretation. arXiv [cs.IR] (2025)
51. Zhuang, H., Qin, Z., Hui, K., Wu, J., Yan, L., Wang, X., Bendersky, M.: Be-
yond yes and no: Improving zero-shot LLM rankers via scoring fine-grained
relevance labels. In: Proc. of NAACL-HLT, pp. 358–370 (2024)
52. Zhuang, H., Qin, Z., Jagerman, R., Hui, K., Ma, J., Lu, J., Ni, J., Wang,
X., Bendersky, M.: RankT5: Fine-tuning T5 for text ranking with ranking
losses. In: Proc. of SIGIR, pp. 2308–2313 (2023)
53. Zhuang, S., Zhuang, H., Koopman, B., Zuccon, G.: A setwise approach for
effective and highly efficient zero-shot ranking with large language models.
In: Proc. of SIGIR, pp. 38–47 (2024)