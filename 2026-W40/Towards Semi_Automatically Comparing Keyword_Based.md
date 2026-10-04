# Towards Semi-Automatically Comparing Keyword-Based and Semantic Search Accuracy

**Authors**: Mohamed Ben Salha, Fiete Lüer, Maik Betka, Stefan Wagner

**Published**: 2026-09-29 15:00:22

**PDF URL**: [https://arxiv.org/pdf/2609.37749v1](https://arxiv.org/pdf/2609.37749v1)

## Abstract
The increasing importance of Information Retrieval (IR) in managing large datasets has highlighted significant limitations in traditional keyword-based search systems. Context-aware chat-based search methods, such as Retrieval Augmented Generation (RAG), have recently emerged, but their evaluation compared to keyword-based systems often relies on subjective user feedback. A rigorous, quantitative comparison between these paradigms remains lacking. This work introduces a novel, preliminary framework to quantitatively assess IR accuracy of search systems that produce different output formats, such as lists and messages. It focuses on two key aspects: the ranking accuracy for keyword-based systems and the completeness of retrieved information for semantic chat-based systems. Our approach enables semi-automatic comparisons of semantic and keyword-based methods using interchangeable equivalence classes tailored to domain-specific contexts (e.g., companies or problems). We validate the framework through an industrial case study, demonstrating statistically significant improvements in context-aware search over keyword-based methods, supported by analyses including the Mann-Whitney U-Test. With its adaptable design, the proposed framework provides a strong foundation for objectively assessing keyword-based and semantic chat-based search methods.

## Full Text


<!-- PDF content starts -->

TOWARDSSEMI-AUTOMATICALLYCOMPARING
KEYWORD-BASED ANDSEMANTICSEARCHACCURACY
Mohamed Ben Salha
Technical University of Munich
Heilbronn, Germany
mohamed.bensalha@tum.deFiete Lüer
Gofore GmbH
Munich, Germany
fiete.lueer@gofore.com
Maik Betka
University of Stuttgart
Stuttgart, Germany
maik.betka@iste.uni-stuttgart.deStefan Wagner
Technical University of Munich
Heilbronn, Germany
stefan.wagner@tum.de
ABSTRACT
The increasing importance of Information Retrieval (IR) in managing large datasets has highlighted
significant limitations in traditional keyword-based search systems. Context-aware chat-based search
methods, such as Retrieval Augmented Generation (RAG), have recently emerged, but their evaluation
compared to keyword-based systems often relies on subjective user feedback. A rigorous, quantitative
comparison between these paradigms remains lacking. This work introduces a novel, preliminary
framework to quantitatively assess IR accuracy of search systems that produce different output formats,
such as lists and messages. It focuses on two key aspects: the ranking accuracy for keyword-based
systems and the completeness of retrieved information for semantic chat-based systems. Our approach
enables semi-automatic comparisons of semantic and keyword-based methods using interchangeable
equivalence classes tailored to domain-specific contexts (e.g., companies or problems). We validate
the framework through an industrial case study, demonstrating statistically significant improvements
in context-aware search over keyword-based methods, supported by analyses including the Mann-
Whitney U-Test. With its adaptable design, the proposed framework provides a strong foundation for
objectively assessing keyword-based and semantic chat-based search methods.
KeywordsInformation Retrieval, Keyword-based Search, Conversational Search, Semantic Search, Context-Aware
Search, Search Accuracy, Retrieval-Augmented Generation (RAG)
1 Introduction
In today’s data-driven world, organizations and individuals rely heavily on vast repositories of documents and wikis
to store and access information. However, the growth in size and complexity of these collections makes efficient
Information Retrieval (IR) challenging. Traditional keyword-based search methods often fall short in understanding
context and handling typographical errors Rahman [2013]. In recent years, advancements in Natural Language
Processing (NLP) have paved the way for more sophisticated IR methods. One of the most important developments
is the emergence of RAG Lewis et al. [2020], which harnesses Large Language Models’ (LLMs) text understanding
and generation capabilities by incorporating an external retrieval mechanism. It provides the answer-generating LLM
with retrieved context, mostly not part of the training corpus of that LLM, enabling more context-aware and accurate
responses to user queries and reducing the likelihood of hallucinations.
While RAG is meant to offer better retrieval performance compared to traditional keyword-based search, a direct
quantitative comparison of both methods is still required and missing for a more complete evaluation. This work
addresses this gap by investigating how the correctness and completeness of information can be used to quantitatively
measure the accuracy of a context-aware search compared to a keyword-based one. We propose a preliminary framework
arXiv:2609.37749v1  [cs.IR]  29 Sep 2026

to semi-automatically evaluate different IR methods using interchangeable equivalence classes, allowing the fine-tuning
of the actual performance to problem and company-specific results. This is validated in an industrial case study, where
the IR accuracy of the search engine of Confluence1, a widely used keyword-based wiki, is compared with RAG. We
provide supplementary materials, including source code and resources, to facilitate the reproduction of our results2.
2 Goal
On the one hand, with an input of keywords, the output of keyword-based search engines has the format of a vertical
list of results, allowing for the display of several potentially relevant entries. The higher the position of an entry is, the
higher its relevance should be Rahman [2013]. Click-Through Rate (CTR) is a metric that expresses the percentage
of users who click on a particular result after viewing it in the search results list First Page Sage [2025]. This metric
demonstrates that users’ clicks on search results drop off exponentially the lower a result appears on the page.
On the other hand, the emergence of chat-based search systems and those capable of semantic search (e.g., RAG) has
marked a significant shift in the field of IR. Taking an input mostly in natural language, they generate text-based outputs
that are evaluated on completeness, hallucination, correctness, and faithfulness Yu et al. [2025], Zhu et al. [2025].
Figure 1 demonstrates the different output formats of both these search systems and the corresponding CTR for each
entry for list-based search results First Page Sage [2025].
     What is the value of PI?
The value of pi (π) is approximately
3.14159265. It is a mathematical constant
defined [...].
Source 1
Source 2
Source 3
Source 4
Source 5
    Ask anything..
Chat-based context-aware search system (e.g., RAG)     PI value
Search Result 1 39.8%
CTR
Search Result 2 18.7%
CTR
Search Result 3 10.2%
CTR
Search Result 10  1.6%
CTR
Keyword-based search system (e.g., Confluence).     .     .
Figure 1: Chat-based and keyword-based search approaches with their different output formats.
Given that both IR systems access the same data, the question of which one delivers more adequate results for users’
queries is both compelling and requires a thorough investigation. However, the different output formats make a direct
comparison of their quality without making further adjustments difficult. Our aim is to tackle this challenge by proposing
an evaluation framework.
3 Related Work
Khan et al. [2014] presented a comparative performance evaluation between three semantic web-based and two keyword-
based search engines. The authors use an ontology-based crawler to resolve the problem of synonyms and polysemy
for semantic search. The evaluation dataset contained 10 queries from various topics. The first twenty documents
on each retrieval output were classified using human judgment as being “relevant” or “non-relevant” to calculate the
1https://www.atlassian.com/software/confluence
2https://doi.org/10.5281/zenodo.23040515
2

precision in a further step. The authors find that semantic web-based search engines perform better than keyword-based
ones. To conclude their work, Khan et al. [2014] presented the advantages and limitations of both search approaches in
terms of input and output relevancy. This study compares semantic and keyword-based search engines that provide
search results as a list. However, it does not compare these systems to other systems that produce different results, such
as the conversational messages generated by chat-based systems (e.g., RAG) compared to traditional keyword lists.
Mala and Lobiyal [2016] conducted a similar experiment, but considered other search engines and queries from the
medical domain. Mesgar et al. [2019] developed a conversational agent to help NLP researchers efficiently explore
literature. For natural language understanding, they use both a rule-based and a machine learning-based approach. The
authors performed a user-based evaluation to assess usage aspects such as diversity and information coverage, achieving
satisfactory results. However, user-based evaluations have several challenges. They are time-consuming, costly, and the
subjective results may be biased due to an unrepresentative sample Almeida et al. [2017].
Despite extensive work in the domains of semantic and conversational search to overcome the challenges traditional
keyword-based systems have, an objective evaluation and comparison of these systems producing different output
formats remains lacking Zamani and Craswell [2020], Balog et al. [2020].
4 Evaluation Framework
The core of our evaluation framework is the quantitative comparison of different IR approaches despite their different
output formats (e.g., list, message). Both the correctness of the results (retrieved results should be relevant) and the
ranking (the most relevant result should be on top of the list) need to be accounted for. To achieve this, we perform the
following steps:
1.Creation of a ground truth to account for language peculiarities (e.g., semantic equivalence and contextual
reasoning). The ground truth is created by defining a checklist of what content should be covered in the
search result and creating queries where some parts are altered, leading to queries with: (1) exact keywords
(to evaluate direct matching of the terms used in the documents and the search), (2) synonyms (to evaluate
semantic equivalence), and (3) synonym-based queries on unrelated topics (to evaluate contextual reasoning).
Having an evaluation dataset spanning these categories better reflects the usage of IR systems because it covers
different aspects such as complexities, writing styles, and lengths.
2.Creation of task-specific equivalence classes containing outputs that are considered qualitatively similar. These
can be used to derive an ordinal scale whose levels can differ according to the problem, allowing a comparison
of approaches. As such, it is necessary that the scale allows for a clear ranking (e.g., "All relevant information
is retrieved", "Some relevant information is retrieved", and "No relevant information is retrieved"). Users are
free to define the equivalence classes and which ones to assign to which results.
3.Evaluation of the quality of the IR approaches via statistical tests (e.g., Mann-Whitney U-Test and Vargha-
Delaney A12), to assess statistically significant differences between the two approaches and measure their
effect size.
5 Experimental Setup and Results
To test the applicability of our preliminary framework in a real-world setting, our research adopts an industrial case study
approach where Confluence is used as a keyword-based wiki. We implemented a RAG application withChromaDB
as a vector database,Llama 3.2 8Bas a LLM for the generation part, andjinaai/jina-embeddings-v2-base-deas an
embedding model, as the Confluence data is in German. For chunking, we do not rely on the conventional method
that uses a predefined length. Instead, we chunk the document content based on its hierarchical structure to ensure
the retention of related information within each chunk. The retrieval component uses aK-based approach, whereK
specifies the number of elements to be retrieved based on their cosine similarity to the query Salemi and Zamani [2024].
In this work, we setK= 5.
While this paper focuses merely on the evaluation framework, we also conducted semi-structured interviews as part
of the broader research industrial case study to assess users’ perception of the Confluence search engine. Therefore,
the interviews are not described or discussed in detail here. However, we asked interviewees to create a diverse set of
example queries, in keywords and natural language, to reflect real-usage data. We then slightly adjusted these queries
and paraphrased each one three times to cover variations across three critical dimensions: complexity, language, and
length, resulting in a total of 63 queries for each IR approach. With that being done, we ensure the validation of our
evaluation framework with two different systems. We first compare Confluence with the entire RAG pipeline, as this
provides an explicitly summarized answer matching the query (so the user does not have to read the entire article).
3

However, this comes with a potential risk of missing or incorrect information (hallucination). Next, we compare
Confluence with only the retrieval component of RAG, as it represents the semantic search part, excluding hallucination
possibilities.
With the creation of the ground truth, the first step of the framework is completed. The second step encompasses the
creation of equivalence classes to group outputs with a similar retrieval quality.
5.1 Confluence vs. RAG
In this case, the industrial partner relies on three equivalence classes and uses an ordinal scale of three levels: {1,2,3}
(lower value is better IR accuracy), signaling if all, some, or no expected information is present. The Confluence search
returns multiple results in a list format, allowing for the display of several potentially relevant articles. However, the
most relevant entries (i.e., all information is present) are expected to be more salient. Referring to CTR allows us to
reward saliency (low value/equivalence class) and punish missing or non-salient information (high value/equivalence
class). Table 1 shows an overview of this scoring schema. This grouping reflects a discretionary decision, made in
consultation with the case company, based on practical considerations relevant to the case.
Value Confluence (CTR) RAG (via checklist)
1 1 - 2 (60%) Information is fully complete
2 3 - 7 (35%) Information is partially complete
3 8 - 10 (5%) Information is unavailable
Table 1: Evaluation framework for Confluence vs. RAG IR accuracy/information availability. Each row represents an
equivalence class assumed in this work. Lower values indicate better IR accuracy.
Having established the equivalence classes set, we proceed with the third step of our evaluation framework, namely the
evaluation of the IR quality by means of statistical tests. To quantify the difference in performance between groups,
we use the A12 statistic, a non-parametric effect size measure proposed by Vargha and Delaney Vargha and Delaney
[2000]. It quantifies in how many cases system 1 achieved a higher value than system 2. Since a higher equivalence
class value signifies a worse performance, a A12 statistic of 0.397 indicates that in approximately 40% of the evaluated
queries, RAG achieved higher values, denoting a worse performance. Thus, in 60% of evaluated cases, RAG had better
performance compared to the Confluence search engine regarding IR. Figure 2 depicts the aggregated performance
among all queries and underpins the advantages of RAG regarding IR.
5.2 Confluence vs. RAG Retrieval Component
We also compare only the retrieval component to Confluence, because it represents the semantic search part of RAG
without the generating component that possibly causes hallucination. Furthermore, it displays retrieved content in
a ranked list, allowing for a comparison with Confluence based on results positions, where each one represents an
equivalence class. For Confluence, search results within the top 10 positions are assigned an equivalence class equal to
their position number, while those beyond 10 receive a default value of 10. For retrieval, the top 5 results are assigned
an equivalence class equal to their position, while missing ones are assigned a default value of 10. This stricter scoring
reflects the higher expectation for retrieval accuracy when using a smaller subset. It emphasizes the importance of
ranking the most relevant documents at the top to ensure effective IR.
The data presented in Figure 3 substantiates the more accurate IR of the retrieval component of RAG compared
to Confluence. Considering that lower values indicate better IR accuracy, the data distribution and the box-plots
corroborate the Vargha-Delaney A12 statistic of 0.353, shown in Table 2 This comparison is another validation of the
preliminary framework and emphasizes its adaptiveness to different IR systems.
5.3 Statistical Tests
Table 2 presents the results of the Mann-Whitney U-Test and the Vargha-Delaney A12 statistic. The Mann-Whitney
U-Test, or Wilcoxon rank-sum test, is a non-parametric statistical test used to identify statistically significant differences
between the distributions of two independent samples. Unlike the T-Test, it does not assume normality and relies on the
ranking of values McKnight and Najab [2010]. We observe statistically significant differences between both approaches,
respectively (p<0.05 threshold).
4

0 10 20 30 40 50 60
Number of queriesConfluenceRAG
34 15 1446 11 61
2
3Figure 2: IR accuracy comparison between Confluence and RAG on a dataset of 63 queries. Lower values indicate
better IR accuracy.
Group 1 Group 2 U Statistic p-value Effect Size A12 Statistic
RAG Confluence 1574.5 0.0195 0.1782 0.397
Retrieval Confluence 1400.5 0.0015 0.2538 0.353
Table 2: Results of Mann-Whitney U-Test and Vargha-Delaney A12 statistic with a dataset of 63 entries for each group.
6 Discussion
We showcase the adaptiveness of the framework and its applicability to different IR systems. However, we note threats
to validity, which open up avenues for future work.
6.1 Threats to Validity
For our definition of IR accuracy, the focus is only on finding the desired information with a disregard for other aspects
(response time, interactions, etc). Second, we assumed the transferability of CTR from Google First Page Sage [2025]
to Confluence. Although variations are expected, we assume these are negligible since, similarly, no more than ten
results are displayed on the results page, and higher-ranked search results are expected to be more clicked Rahman
[2013]. Adaptions across domains are limited by the usage of different tools (with different CTRs) as well as the desired
goodness and resulting ranking. Ranking the queries can be difficult in systems where, e.g., many results fulfill all
aspects of a correct result, and setting good cut-off values for equivalence classes can be difficult in dynamic systems
with growing documentation. On the other hand, the CTR allows an up-front estimation of the goodness of different IR
algorithms even before their actual implementation in production environments.
The manual creation of the evaluation dataset may inadvertently leave out important entries or edge cases that are crucial
for the IR systems’ performance in production. However, we ensured to cover a broad spectrum of the underlying data
by involving employees from the case company.
The validation of our framework was performed through an industrial case study at only one company. Hence, the
results are not necessarily generalizable to other specific industrial environments and settings. However, this work
demonstrated a proof of concept and can be extended to further companies.
5

1 2 3 4 5 6 7 8 9 10
Information retrieval accuracy (equivalence classes)Confluence
Retrieval
Figure 3: IR accuracy comparison between Confluence and the Retrieval part of RAG on a dataset of 63 queries. Lower
values indicate better IR accuracy.
6.2 Future Work
Future work can tackle these challenges. By creating a freely available ground truth with well-defined contextual areas,
a stricter control over the assumptions is given. This would also open up reproducible and comparable evaluations
(without the need for repeated user studies). Setting up the ground truth can be automated more extensively using
current means (e.g., generating semantically similar queries or automatically creating preliminary checklists for
completeness/correctness or prelabeling of information based on the checklist). Since means to automate this process
are biased, a manual control of the generated ground truth is necessary.
Other aspects beyond the accuracy, such as response time and the number of interactions, may affect the overall
search experience and should be integrated into the framework as well. Additionally, a comparison between other
conversational-based search systems and keyword-based systems with a wider array of metrics could explore the
applicability of the preliminary evaluation framework. A broader validation by means of a large-scale experiment that
represents more demographic groups and industries would foster the generalizability of our framework.
7 Conclusion
We propose a semi-automatic approach for a quantitative comparison between traditional keyword-based and semantic
search systems, regardless of the respective output formats. By creating an evaluation dataset of queries (in both natural
language and via keywords) with a checklist for content to be found spanning different language styles, lengths, and
complexity levels, real-world usage can be simulated. We then create ordinal-scaled equivalence classes exhibiting
the same retrieval quality based on the checklists. Investigated (but expandable) criteria are correctness, completeness,
6

and ranking. To obtain comparison results, we run statistical tests (e.g., Mann-Whitney U-Test and Vargha-Delaney
A12). We validate our preliminary evaluation framework with an industrial case study comparing the Confluence search
engine with RAG and its retrieval component separately. This demonstrates statistically significant improvements in
context-aware search over traditional keyword-based methods, as well as small and moderate advantages of RAG and
only the retrieval component of RAG over keyword-based methods, respectively. With its adaptable design, the proposed
preliminary framework shows promising potential towards a more objective evaluation between keyword-based and
context-aware search methods. It alleviates the need for costly and time-consuming subjective user studies. At the same
time, it ensures a customized evaluation tailored to domain-specific contexts.
8 GenAI Usage Disclosure
We used ChatGPT to generate code snippets for the statistical tests and the plots. The code of the chunking method in
RAG, based on the hierarchical structure, is generated by ChatGPT. We utilized DeepL to translate the resources from
German to English. The authors carefully reviewed and adjusted all AI-generated content for accuracy and clarity.
References
Fernando Almeida, Daniel Faria, and André Queirós. Strengths and limitations of qualitative and quantitative research
methods.European Journal of Education Studies, 3:369–387, 09 2017. doi: 10.5281/zenodo.887089.
Krisztian Balog, Lucie Flekova, Matthias Hagen, Rosie Jones, Martin Potthast, Filip Radlinski, Mark Sanderson,
Svitlana Vakulenko, and Hamed Zamani. Common conversational community prototype: Scholarly conversational
assistant.arXiv preprint arXiv:2001.06910, 2020.
First Page Sage. Google click-through rates (ctrs) by ranking position in 2025. https://firstpagesage.com/
reports/google-click-through-rates-ctrs-by-ranking-position/, 2025. Last updated April 4, 2025.
Javed Ahmad Khan, Deepak Sangroha, Masroor Ahmad, and Md. Tanzillur Rahman. A performance evaluation of
semantic based search engines and keyword based search engines. In2014 International Conference on Medical
Imaging, m-Health and Emerging Communication Systems (MedCom), pages 168–173, 2014. doi: 10.1109/MedCom.
2014.7005997.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler,
Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks.
Advances in neural information processing systems, 33:9459–9474, 2020.
Vajenti Mala and D. K. Lobiyal. Semantic and keyword based web techniques in information retrieval. In2016
International Conference on Computing, Communication and Automation (ICCCA), pages 23–26, 2016. doi:
10.1109/CCAA.2016.7813724.
Patrick E McKnight and Julius Najab. Mann-whitney u test.The Corsini encyclopedia of psychology, pages 1–1, 2010.
doi: 10.1002/9780470479216.corpsy0524.
Mohsen Mesgar, Paul Youssef, Lin Li, Dominik Bierwirth, Yihao Li, Christian M. Meyer, and Iryna Gurevych. When
is acl’s deadline? a scientific conversational agent, 2019. URLhttps://arxiv.org/abs/1911.10392.
Mahmudur Rahman. Search engines going beyond keyword search: A survey.International Journal of Computer
Applications, 75:1–8, 08 2013. doi: 10.5120/13200-0357.
Alireza Salemi and Hamed Zamani. Evaluating retrieval quality in retrieval-augmented generation. InProceedings of
the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’24,
page 2395–2400, New York, NY , USA, 2024. Association for Computing Machinery. ISBN 9798400704314. doi:
10.1145/3626772.3657957. URLhttps://doi.org/10.1145/3626772.3657957.
András Vargha and Harold D Delaney. A critique and improvement of the cl common language effect size statistics of
mcgraw and wong.Journal of Educational and Behavioral Statistics, 25(2):101–132, 2000.
Hao Yu, Aoran Gan, Kai Zhang, Shiwei Tong, Qi Liu, and Zhaofeng Liu. Evaluation of retrieval-augmented generation:
A survey. In Wenwu Zhu, Hui Xiong, Xiuzhen Cheng, Lizhen Cui, Zhicheng Dou, Junyu Dong, Shanchen Pang,
Li Wang, Lanju Kong, and Zhenxiang Chen, editors,Big Data, pages 102–120, Singapore, 2025. Springer Nature
Singapore. ISBN 978-981-96-1024-2. doi: 10.1007/978-981-96-1024-2_8.
Hamed Zamani and Nick Craswell. Macaw: An extensible conversational information seeking platform. InProceedings
of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’20,
page 2193–2196, New York, NY , USA, 2020. Association for Computing Machinery. ISBN 9781450380164. doi:
10.1145/3397271.3401415. URLhttps://doi.org/10.1145/3397271.3401415.
7

Kunlun Zhu, Yifan Luo, Dingling Xu, Yukun Yan, Zhenghao Liu, Shi Yu, Ruobing Wang, Shuo Wang, Yishan Li,
Nan Zhang, Xu Han, Zhiyuan Liu, and Maosong Sun. Rageval: Scenario specific rag evaluation dataset generation
framework, 2025. URLhttps://arxiv.org/abs/2408.01262.
8