# Beating Harmful Stereotypes Through Facts: RAG-based Counter-speech Generation

**Authors**: Greta Damo, Elena Cabrio, Serena Villata

**Published**: 2025-10-14 09:20:01

**PDF URL**: [http://arxiv.org/pdf/2510.12316v1](http://arxiv.org/pdf/2510.12316v1)

## Abstract
Counter-speech generation is at the core of many expert activities, such as
fact-checking and hate speech, to counter harmful content. Yet, existing work
treats counter-speech generation as pure text generation task, mainly based on
Large Language Models or NGO experts. These approaches show severe drawbacks
due to the limited reliability and coherence in the generated countering text,
and in scalability, respectively. To close this gap, we introduce a novel
framework to model counter-speech generation as knowledge-wise text generation
process. Our framework integrates advanced Retrieval-Augmented Generation (RAG)
pipelines to ensure the generation of trustworthy counter-speech for 8 main
target groups identified in the hate speech literature, including women, people
of colour, persons with disabilities, migrants, Muslims, Jews, LGBT persons,
and other. We built a knowledge base over the United Nations Digital Library,
EUR-Lex and the EU Agency for Fundamental Rights, comprising a total of 32,792
texts. We use the MultiTarget-CONAN dataset to empirically assess the quality
of the generated counter-speech, both through standard metrics (i.e., JudgeLM)
and a human evaluation. Results show that our framework outperforms standard
LLM baselines and competitive approach, on both assessments. The resulting
framework and the knowledge base pave the way for studying trustworthy and
sound counter-speech generation, in hate speech and beyond.

## Full Text


<!-- PDF content starts -->

Beating Harmful Stereotypes Through Facts:
RAG-based Counter-speech Generation
Greta Damo1, Elena Cabrio1Serena Villata1
1Université Côte d’Azur, CNRS, Inria, I3S, France
Correspondence:greta.damo@univ-cotedazur.fr
Abstract
Counter-speech generation is at the core of
many expert activities, such as fact-checking
and hate speech, to counter harmful content.
Yet, existing work treats counter-speech gen-
eration as pure text generation task, mainly
based on Large Language Models or NGO ex-
perts. These approaches show severe draw-
backs due to the limited reliability and coher-
ence in the generated countering text, and in
scalability, respectively. To close this gap,
we introduce a novel framework to model
counter-speech generation as knowledge-wise
text generation process. Our framework inte-
grates advanced Retrieval-Augmented Genera-
tion (RAG) pipelines to ensure the generation
of trustworthy counter-speech for 8 main tar-
get groups identified in the hate speech litera-
ture, including women, people of colour, per-
sons with disabilities, migrants, Muslims, Jews,
LGBT persons, and other. We built a knowl-
edge base over the United Nations Digital Li-
brary, EUR-Lex and the EU Agency for Fun-
damental Rights, comprising a total of 32,792
texts. We use the MultiTarget-CONAN dataset
to empirically assess the quality of the gen-
erated counter-speech, both through standard
metrics (i.e., JudgeLM) and a human evalua-
tion. Results show that our framework outper-
forms standard LLM baselines and competi-
tive approach, on both assessments. The result-
ing framework and the knowledge base pave
the way for studying trustworthy and sound
counter-speech generation, in hate speech and
beyond.Warning: this paper contains explicit ex-
amples some readers may find offensive.
1 Introduction
The rise of social media has transformed global
communication, enabling the rapid exchange of
ideas and empowering marginalized communi-
ties (Lenhart et al., 2010; Ortiz-Ospina, 2019;
Siricharoen, 2023). Yet, these platforms have also
become fertile ground forhate speech (HS). Theanonymity and virality of online spaces allow abu-
sive and discriminatory messages to spread widely,
normalizing toxic discourse with little accountabil-
ity (Zimbardo, 1969; Mondal et al., 2017; Mathew
et al., 2019a).Counter-speech (CS)offers a con-
structive alternative to censorship-based measures.
Defined as a non-hostile response employing facts,
logic, or alternative perspectives, CS challenges
stereotypes and misinformation while fostering di-
alogue (Benesch, 2014; Schieb and Preuss, 2016).
Studies show that CS can reduce the persuasive
power of HS and promote more inclusive discourse
(Kiritchenko et al., 2021).
Figure 1: Examples of two CS generated with GPT-4o-
mini: the first without RAG, the second one with RAG.
NGOs and experts successfully deployed CS
campaigns, but the scale of HS makes manual CS
unsustainable (Chung et al., 2021b). This challenge
has motivated research on automatic CS generation,
an emerging NLP task. Early approaches relied on
curated datasets (Chung et al., 2019; Fanton et al.,
2021) and fine-tuning of pre-trained language mod-
els (PLMs) (Tekiro ˘glu et al., 2020; Zhu and Bhat,
2021; Tekiro ˘glu et al., 2022). While effective in-
1arXiv:2510.12316v1  [cs.CL]  14 Oct 2025

domain, such models often produce generic or off-
topic responses on unseen HS. To achieve factuality
and informativeness in the generated CS, it is there-
fore essential to embed knowledge on the target HS
groups in the generation process.
We tackle this challenging task by proposing
a novel knowledge-grounded framework for au-
tomatic counter-speech generation that integrates
advanced retrieval-augmented generation (RAG)
pipelines. Figure 1 shows two examples of CS,
with and without RAG augmentation. Our contri-
butions are fourfold:(1)we provide a systematic
comparison of multiple retrievers and LLMs for
CS generation, combining three retrieval methods
with four language models;(2)we enforce con-
cise, two-sentence outputs tailored for social media
deployment, ensuring responses remain natural, re-
latable, and effective in real-world contexts;(3)
we conduct a pairwise evaluation against existing
state-of-the-art systems (Russo, 2025; Wilk et al.,
2025) using JudgeLM, adapted to balance factual
grounding with conciseness and pragmatic suitabil-
ity, together with an extensive human evaluation,
showing that our framework steadily outperforms
standard baselines and state-of-the-art competitors;
and(4)we will release all generated CS along with
their top-3 retrieved evidence sentences, offering a
reusable resource for the research community1.
2 Related Work
Counter Speech Datasets.Early CS resources
relied on manual annotation. Mathew et al. (2019b)
introduced a dataset from YouTube comments, fol-
lowed by Qian et al. (2019) with large-scale Reddit
and Gab interventions. Later work emphasized
contextual coverage: Yu et al. (2022) added con-
versational context; Chung et al. (2019) released
the CONAN corpus, later expanded to multiple
HS targets in MT-CONAN (Fanton et al., 2021)
and to dialogue in DIALOCONAN (Bonaldi et al.,
2022). Knowledge-grounded variants (Chung et al.,
2021a) and semi-automatic annotation pipelines
(Tekiro ˘glu et al., 2022) further reduced costs.
Counter Speech Generation.Initial sequence-
to-sequence models produced generic outputs
(Qian et al., 2019). Subsequent pipelines improved
quality via retrieval and selection (Zhu and Bhat,
2021; Chung et al., 2021a; Jiang et al., 2023). Other
studies focused on style and control (Bonaldi et al.,
2023; Saha et al., 2022; Gupta et al., 2023), leverag-
1Data and code will be made available upon acceptance.ing argumentative and emotional cues. With LLMs,
few-shot and zero-shot prompting became popular
(Ashida and Komachi, 2022; Tekiro ˘glu et al., 2022;
Zhao et al., 2023), reducing annotation needs but
often lacking factual grounding.
Evaluation of Counter Speech.Automatic met-
rics (BLEU, ROUGE, BERTScore) correlate
poorly with human judgments, prompting ex-
ploration of novelty- and repetition-based scores
(Wang and Wan, 2018; Bertoldi et al., 2013), and
LLM-based frameworks (e.g., GPT-4, PandaLM,
JudgeLM, UniEval) (Zhu et al., 2023; Zhong et al.,
2022), which show improved reliability for multi-
aspect CS assessment (Jones et al., 2024; Damo
et al., 2025).
RAG for CS.RAG addresses a key limitation
of prior approaches by grounding outputs in verifi-
able evidence, thereby enhancing factuality and
persuasiveness in countering HS. Chung et al.
(2021a) proposed a retrieval-augmented pipeline
that first generates queries from HS using key-
word extraction, then employs BM25 (Robert-
son and Zaragoza, 2009) to retrieve relevant ar-
ticles from Newsroom (Grusky et al., 2018) and
WikiText-103 (Merity et al., 2016). From these,
the most relevant sentences are selected using the
ROUGE-L metric (Lin, 2004), before being passed
to GPT-2 (Radford et al., 2019) and XNLG (Chi
et al., 2020). Similarly, Jiang et al. (2023) intro-
duced RAUCG, which retrieves counter-arguments
from the ChangeMyView subreddit, selecting them
based on stance consistency, semantic overlap, and
a custom perplexity-based fitness function. The
final generation step employs energy-based decod-
ing to preserve factual knowledge while countering
HS fluently. Jiang et al. (2025) proposed ReZG, a
retrieval-augmented zero-shot approach that inte-
grates multi-dimensional hierarchical retrieval with
constrained decoding, enabling the generation of
more specific CS for unseen HS targets. Wilk et al.
(2025) leverage both curated background knowl-
edge and web search to improve factuality. Russo
(2025) evaluated reranker-based pipelines, showing
that fine-grained retrieval significantly improves
factuality and relevance.
Differently from previous approaches, in this
work we design a RAG pipeline built on a novel
large, authoritative knowledge base that minimizes
the risk of misinformation. Our approach consis-
tently produces CS that is factually rich, but also
concise and well-suited for social media contexts.
2

3 Knowledge Base Construction
As a first step, we built a comprehensive Knowl-
edge Base (KB) designed to ensure maximal cov-
erage of documents addressing social groups com-
monly targeted by hate speech. The goal of this
KB is to gather all relevant materials — such as
reports, resolutions, and legal texts — that pro-
vide evidence or context regarding these topics.
Specifically, we focused on the following 8 target
groups: women, people of colour, persons with
disabilities, migrants, Muslims, Jews, LGBT per-
sons, and other. These categories align with the
classic targets in the literature, including the ones
of our baseline MultiTarget-CONAN (Chung et al.,
2019). We used GPT-based prompting to generate
synonyms and semantically related keywords, en-
suring that queries captured diverse terminology
across cultural and policy contexts (see Appendix
B for the prompt). To guarantee relevance, we per-
formed a keyword-based search so that retrieved
knowledge was directly aligned with the hateful
messages targeting these groups. To ensure reli-
ability in the generated CS, we relied exclusively
on institutional publicly available sources, i.e., the
United Nations Digital Library, EUR-Lex, and the
European Union Agency for Fundamental Rights
(FRA). To construct our knowledge base, we follow
three main steps: document retrieval, PDF-to-text
conversion, and knowledge base integration.
Document Retrieval.TheUnited Nations Digital
Libraryserves as the central repository for official
UN documents on human rights, equality, and anti-
discrimination. We developed a custom crawler
using requests andBeautifulSoup to systemati-
cally combine three query dimensions—target key-
words, document types (e.g., resolutions, treaties,
NGO statements), and years (2000–2025). The
crawler paginated through results, extracted and
normalized documents in English, and organized
downloads by target group. A metadata file
recorded id, fname, target, type, year,
url. Error handling covered duplicate checks, re-
tries, and skipped completed downloads. At the
European level,EUR-Lexprovides access to EU
law, treaties, and legal acts, while theEU Agency
for Fundamental Rights (FRA)publishes reports
on human rights within the EU. Using the same
target keywords, we retrieved documents in En-
glish (2000–2025) from both sources, following
the same metadata structure as the UN corpus.
These sources complement the UN materials, form-ing a multi-level knowledge base—global and Eu-
ropean—that ensures comprehensive and reliable
grounding for CS generation.
PDF Text Extraction.UN and EU documents
were converted into plain text using PyMuPDF
to extract machine-readable text, while Tesseract
OCR handled scanned PDFs. The process was
parallelized for efficiency and designed to be fault-
tolerant, with outputs stored incrementally in JSON
batches linked to document IDs.
Knowledge Base Integration.All materials from
UN, EU, and FRA were standardized into JSON
with metadata (target group, document type, year,
URL). The KB spans the years 2000–2025 and
combines factual and policy-oriented resources, or-
ganized by target group and document type. To
the best of our knowledge, this is the first large-
scale, authoritative knowledge base specifically
constructed for counter-speech generation.
Keyword EU # EU wds UN # UN wds
Disabled 20 17,980 1,718 9,705
Human rights 72 35,070 13,396 10,048
Jews 29 25,111 178 5,707
LGBT 22 26,953 164 5,998
Migrants 61 25,948 3,784 8,748
Muslim 11 21,340 406 3,272
POC 38 27,897 4,840 8,365
Women 13 23,122 8,040 8,006
Total / Avg. 266 25,928 32,526 7,669
Table 1: Comparison of EU (FRA and Eur-Lex) and UN
documents by keyword.
Table 1 reports descriptive statistics of the UN
and EU corpora by target group keyword. While
the UN collection is considerably larger in terms of
the number of documents (over 32k), EU reports
are more detailed, with substantially higher aver-
age word counts per document ( ≈26k vs.≈7.7k ).
Coverage also varies by group: for example, the
UN corpus contains a large volume of material on
human rights and women, while EU provides more
in-depth analyses on migrants, LGBT+ individu-
als, and people with disabilities. Together, these
complementary sources balance breadth (UN) with
depth (EU), creating a diverse and representative
foundation for counter-speech generation. This
ensures that our knowledge base is both compre-
hensive and adaptable to different CS scenarios.
3

4 Pipeline
Our RAG-based framework integrates three key
components: (i) paragraph retrieval, (ii) paragraph
summarization, and (iii) counter-speech generation.
Figure 2 provides an overview of the pipeline.
Figure 2: Overview of our RAG-based CS generation
framework. Step 1: paragraph retrieval from a domain-
specific knowledge base; Step 2: LLM paragraph sum-
marization; Step 3: CS generation conditioned on the
summarized knowledge.
4.1 Paragraph Retrieval
Our domain-specific knowledge base (KB), as de-
scribed in Section 3 is segmented into paragraphs
and tokenized for fine-grained access. Each para-
graph is embedded and stored using FAISS (Douze
et al., 2024) to allow efficient similarity search. We
obtain a total of≈3 billions paragraphs.
Given a HS message h, we aim to retrieve
a small set of relevant paragraphs from the KB
(i.e., the most similar to the target HS). For
this, we employ three complementary retrieval
models: BM25, Sentence-BERT (SBERT), and
BGE-M3. Let the KB consist of paragraphs
P={p 1, p2, . . . , p N}. For each retriever r∈
{BM25,SBERT,BGE-M3} , we compute a simi-
larity score: sr(h, p i) =sim r(ϕ(h), ϕ(p i))where
ϕ(·) is the embedding (or term-weight) function
induced by the retriever. For BM25, the similarity
score is computed based on TF-IDF, and document
length normalization. For SBERT and BGE-M3,
similarity is measured as the cosine similarity be-
tween dense vector embeddings of the HS and can-
didate paragraph, capturing semantic relatedness
even in the absence of exact lexical overlap. For
each retriever, we obtain the top- kranked para-
graphs:R r(h) ={p r,1, pr,2, . . . , p r,k}, k= 3.
The choice of k= 3 ensures high precision while
providing enough external knowledge to support
short and specific CS responses suited for social
media. Each retrieved paragraph is stored together
with its source document identifier for traceability.4.2 Paragraph Summarization
Although retrieval provides relevant context, LLMs
have limited context windows, making direct use of
full paragraphs infeasible. To address this, we sum-
marize retrieved paragraphs before passing them to
the generation stage.
For each retriever rand each para-
graph pr,j∈R r(h), we obtain a sum-
mary using one out of four LLMs: m∈
{GPT-4o-mini,Llama,Command-R,Mistral}.
Formally, the summarization function is:
σm(pr,j),
which produces a condensed version of paragraph
pr,jwith respect to the generation task (with
max_new_tokens=150 ). This yields a set of sum-
marized paragraphs:
Sr,m(h) ={σ m(pr,1), σm(pr,2), σm(pr,3)}.
Summaries are stored in CSV format along with
their source IDs, ensuring alignment between re-
trieval and generation stages. The summarization
prompts are provided in Appendix B.
4.3 Counter-Speech Generation
In the final stage, the summaries are used as exter-
nal knowledge to generate CS. For each HS mes-
sageh, retriever r, and model m, the generation
function is defined as:
cr,m(h) =LLM m(h, S r,m(h)),
where cr,m(h)denotes the counter-speech gener-
ated by model mconditioned on the hate speech in-
stancehand the summarized knowledgeS r,m(h).
Since we use three retrievers and four LLMs, the
system produces: |C(h)|=|{c r,m(h)}|= 3×4 =
12CS outputs for each HS instance. This enables
systematic comparison across retrieval methods
and summarization strategies. To ensure that gen-
erated CS is deployable in (online) real-world con-
texts, we restrict outputs to a maximum of two
sentences. This reflects the communicative norms
of social media platforms, where posts are typically
1–2 sentences long (¸ Sahinuç and Toraman, 2021).
Prior work shows that overly verbose responses
are less engaging and less effective in countering
harmful narratives (Russo et al., 2023). Concise
and relatable CS has been repeatedly identified as
key to user engagement and effectiveness (Bonaldi
et al., 2024; Benesch et al., 2016).
4

4.4 Models and Retrievers
We evaluate our pipeline with four LLMs of compa-
rable parameter scale (7–8B), and three retrievers
representing both sparse and dense approaches.
LLMs.We employ Meta-Llama-3.1-8B-
Instruct2, Cohere’s Command-R-7B3(Cohere
et al., 2025), Mistral-7B-Instruct-v0.34, and
gpt-4o-mini-2024-07-18. These models were
selected as LLMs of similar size, balancing
efficiency and accuracy, allowing comparison
across open-weight and proprietary settings.
Retrievers. BM25(Robertson and Zaragoza,
2009) is a sparse lexical retriever based on TF-IDF
with document length normalization. It remains an
effective baseline for keyword-sensitive domains
where exact term overlap is important.Sentence-
BERT (SBERT)(Reimers and Gurevych, 2019)
encodes queries and passages into dense vector em-
beddings optimized for semantic similarity, with
ranking performed via cosine similarity.BGE-M3
(BAAI General Embeddings).BGE-M3 (Chen
et al., 2024) is a recent dense embedding model
trained for multilingual and multi-task semantic
retrieval. These retrievers capture complementary
dimensions of relevance, from exact term matching
(BM25) to semantic similarity (SBERT, BGE-M3),
ensuring robust retrieval across different types of
hateful content and knowledge sources.
5 Experimental Setup
We experimented on the Multi-Target CONAN
(MTCo) dataset (Bonaldi et al., 2022), which con-
tains 5,003 HS/CS pairs in English covering multi-
ple target groups, including people with disabilities,
Jews, LGBT+ individuals, Muslims, migrants, peo-
ple of color (POC), and women. Collected through
a human-in-the-loop process, MTCo provides high-
quality, contextually relevant CS, and serves as a
first baseline for our experiments. Additionally,
we compare our pipeline with the four instruction-
tuned LLMs of comparable size without RAG. Fi-
nally, we compare our results with competitive ap-
proaches (Russo, 2025; Wilk et al., 2025)5.
2https://huggingface.co/meta-llama/Llama-3.
1-8B-Instruct
3We included CommandR, optimized for grounding out-
puts in retrieved context.
4https://huggingface.co/mistralai/
Mistral-7B-Instruct-v0.3
5We cannot compare to (Jiang et al., 2023, 2025) due to
lack of publicly available code and data, even upon request.6 Metrics
Automatic metrics.We evaluate the quality of
generated CS using a combination of reference-
based, reference-less, and LLM-based methods.
Reference-based Metrics.To evaluate alignment
with human-written CS in MT-Co, we report:
BLEU-4(Papineni et al., 2002),ROUGE-L(Lin,
2004),METEOR(Banerjee and Lavie, 2005), and
BERTScore(Zhang et al., 2019).
Reference-less Metrics.To complement reference-
based evaluation, we assess intrinsic qualities of
generated CS.Distinct-1/2(Li et al., 2015) mea-
sures the proportion of unique unigrams and bi-
grams, reflecting lexical diversity.Repetition Rate
(RR)(Cettolo et al., 2014) represents the fraction
of repeated n-grams within a generation.Safety:
the OpenAI’s content moderation API scores each
output across categories of potential harm (e.g.,
hate, sexual, violence), with higher values indicate
safer counter-speech (Bonaldi et al., 2024).
LLM-as-a-Judge Evaluation.We use an adaptation
of JUDGELM (Zhu et al., 2023) tailored for CS
evaluation (Bonaldi et al., 2025; Zubiaga et al.,
2024). While traditional metrics capture surface
and semantic alignment, JudgeLM provides a more
holistic evaluation of CS quality along different
dimensions. We use the version of JUDGELM with
fine-tuned Llama-instruct-7B.
Human evaluation metrics.Automatic metrics
cannot fully account for pragmatic qualities of CS.
To address this, we conduct a human evaluation
focusing on dimensions directly relevant to the ef-
fectiveness and quality of CS (Stapleton and Wu,
2015; Bengoetxea et al., 2024; Bonaldi et al., 2024).
Participants assess each CS along the following cri-
teria, using a Likert scale from 1 to 3 (3 being the
highest score)(see Appendix F for full guidelines):
Relevance: the CS specifically addresses both the
topic and the intended target of the HS.
Correctness: CS stylistic quality, including fluency
and absence of offensive language.
Factuality: the CS introduces additional factual
information and whether these facts are accurate.
Cogency: the CS presents logically sound and rel-
evant arguments that effectively counter the HS.
In addition, two binary judgments are collected:
Effectiveness: the CS is considered persuasive
and likely to influence the perspective of the hate
speech author or audience.
Best Response: for each HS message, annotators
select the most effective CS among multiple CS.
5

Model BLEU METEOR ROUGE-L BERTScore F1 Distinct-1 Distinct-2 Repetition Rate Safety
No Retrieval (No-RAG)
LLaMA 0.0115 0.16380.11640.8575 0.0196 0.1238 0.0034 0.987
CommandR 0.0118 0.1432 0.11630.85900.0233 0.1462 0.0002 0.988
Mistral 0.0116 0.1435 0.1121 0.8575 0.0210 0.1059 0.2668 0.992
GPT 0.0105 0.1494 0.1124 0.8580 0.0156 0.1037 0.00180.993
BM25 Retrieval
LLaMA 0.0079 0.1639 0.1059 0.8491 0.0291 0.1976 0.0088 0.974
CommandR 0.0080 0.1527 0.1059 0.8484 0.0304 0.1963 0.0010 0.978
Mistral 0.0079 0.1442 0.0986 0.84900.0371 0.21730.0004 0.981
GPT 0.0070 0.1580 0.1015 0.8507 0.0231 0.16480.00000.983
Sentence-BERT Retrieval
LLaMA 0.0087 0.1683 0.1090 0.8508 0.0272 0.1835 0.0106 0.971
CommandR 0.0090 0.1614 0.1115 0.8504 0.0281 0.1780 0.0054 0.969
Mistral 0.0085 0.1469 0.1011 0.8500 0.0347 0.2009 0.0010 0.976
GPT 0.0075 0.1616 0.1036 0.8519 0.0213 0.15310.00000.979
BGE-M3 Retrieval
LLaMA 0.00890.17440.1119 0.8511 0.0237 0.1667 0.0042 0.972
CommandR 0.0091 0.1597 0.1143 0.8523 0.0247 0.1713 0.0002 0.974
Mistral0.00940.1505 0.1063 0.8537 0.0315 0.1939 0.0008 0.979
GPT 0.0079 0.1658 0.1086 0.8538 0.0203 0.15330.00000.982
Table 2: Automatic evaluation results for CS generation across different LLMs and retrieval strategies (averages).
7 Results
Analysis of the automatic metrics.Table 2 re-
ports average results, that can be interpreted along
three main dimensions: content quality, diversity,
and safety.Content Quality:scores on BLEU,
METEOR, and ROUGE-L remain relatively low,
as expected for open-ended generation, while
BERTScore values are consistently high ( ≈0.85),
indicating strong semantic similarity to MT-Co
references. Non-RAG outputs achieve slightly
higher lexical and semantic overlap, suggesting
that retrieval introduces content that, while factu-
ally richer, diverges from exact reference phrasing.
RAG outputs show less words overlapping, likely
reflecting the inclusion of factual information from
the knowledge base.Diversity:incorporating re-
trieval substantially improves lexical diversity and
reduces repetition. Mistral with BGE-M3 shows
the largest gains, while No-RAG models, partic-
ularly Mistral, exhibit high repetition and lower
diversity.Safety:No-RAG configurations achieve
the highest safety, with GPT leading (0.993). RAG
outputs show slightly lower safety, likely due to oc-
casional noise introduced from the retrieval process.
Nevertheless, safety remains high overall, indicat-
ing that fact-grounded augmentation does not sub-
stantially compromise non-harmfulness.Model-
level Trends:GPT consistently delivers the safest
outputs and competitive BERTScore, though with
lower lexical diversity than Mistral. Mistral ex-
cels in diversity with RAG but performs poorly
in No-RAG due to high repetition. CommandRand LLaMA provide stable but moderate perfor-
mance across metrics. Overall, RAG with BGE-M3
achieves the best balance between quality, diversity,
and informativeness. All results differ significantly
(see Appendix D for more details).
LLM-as-a-Judge Evaluation.We evaluate CS
quality using JUDGELM, focusing on four dimen-
sions: factuality, number of relevant facts, rele-
vance to the HS, and specificity. This prompt (Ap-
pendix B) guides pairwise comparisons between
model outputs, emphasizing informative and tar-
geted CS over surface-level similarity. Table 3
reports JUDGELM results for RAG vs. No-RAG
models across retrieval strategies. BGE-M3 yields
the strongest results, followed by SentBERT and
BM25, confirming the advantage of semantically
rich retrieval. Among generators, GPT wins most
comparisons, showing effective integration of re-
trieved content. Mistral also benefits notably from
RAG, while CommandR performs less consistently.
Table 4 compares the best RAG setups (LLM +
BGE-M3) against human-written MT-Co CS. All
RAG systems perform strongly, with GPT + BGE-
M3 approaching human-level quality and Mistral
+ BGE-M3 remaining competitive, whereas Com-
mandR trails slightly but still surpasses baselines.
Interpretation.Together, automatic metrics and
JUDGELM evaluations reveal clear trends:(1)
RAG outputs are less repetitive, and more diverse
than No-RAG, reflecting the addition of factual con-
tent. Lexical and semantic similarity remains high,
6

Model BM25 SentBERT BGE-M3
Mistral 3011 3103 3557
LLaMA 2576 2805 3532
GPT3524 3941 4223
CommandR 2490 (lost) 2620 3125
Table 3:JudgeLM wins (out of 5003 pairwise comparisons)
comparing corresponding LLMs w/ and w/out RAG with dif-
ferent retrieval strategies. “Lost” indicates that the model was
outperformed.
Model BM25 SentBERT BGE-M3
Mistral 4721 4831 4944
LLaMA 4701 4757 4893
GPT4866 4948 4976
CommandR 4161 4351 4653
Table 4: JudgeLM scores comparing the best RAG meth-
ods against CS from MT-Co.
ensuring CS remains aligned with the original in-
tent.(2)Safety is only slightly reduced with RAG,
likely due to occasional noisy content.(3)RAG
consistently outperforms No-RAG and MT-Co CS
in both automatic and JUDGELM evaluations.
These findings demonstrate that RAG —par-
ticularly with semantically rich BGE-M3 embed-
dings— enhances CS quality, diversity, and factual
grounding, while maintaining strong safety and
overall effectiveness.
8 Human Evaluation
We conducted a human evaluation with 26 partici-
pants to assess the quality and effectiveness of the
generated CS6. Our evaluation setup consisted of
10 HS examples, extracted randomly from MT-Co
keeping the target distribution, with 9 CS candi-
dates for each HS. We selected the CS from MT-
Co and the CS generated by our four LLMs with-
out the RAG pipeline, and the corresponding CS
with RAG using BMG-M3 as retriever, since it
has shown the strongest performance in the auto-
matic and LLM-based metrics. Participants rated
all 90 CS candidates along four dimensions, i.e.,
Relevance, Factuality, Cogency, and Correctness
(1–3 scale), judged whether each CS was effec-
tive (Yes/No) and selected the best CS per HS. We
collected a total of 2340 evaluations.
Metric Scores.Table 5 reports average scores for
Relevance, Factuality, Cogency, and Correctness
across CS methods. GPT RAG achieved the highest
6Participants’ age is between 18 and 50, there is a bal-
ance between genders, and their level of education spans high
school diploma to PhD. Detailed statistics are in Appendix E.scores overall, particularly in Factuality (2.75) and
Cogency (2.41). RAG-based methods generally
outperformed their non-RAG counterparts, while
MT-Co scored lowest across all metrics, indicat-
ing lower relevance, factual accuracy, and overall
effectiveness.
Method Rel. Fact. Cog. Corr.
MT-Co 1.67 2.28 1.38 2.07
Llama No RAG 2.49 2.77 2.14 2.72
Llama RAG 2.53 2.75 2.50 2.66
CommandR No RAG 2.25 2.75 1.85 2.63
CommandR RAG 2.48 2.74 2.25 2.75
Mistral No RAG 2.11 2.69 1.77 2.58
Mistral RAG 2.30 2.69 2.16 2.60
GPT No RAG 2.34 2.73 1.91 2.62
GPT RAG 2.55 2.75 2.41 2.68
Table 5: Average scores (1–3) for Relevance (Rel.), Fac-
tuality (Fact.), Cogency (Cog.), and Correctness (Corr.),
per CS method.
Best CS.RAG-based methods are chosen more
frequently as best CS, implying that they are clearly
preferred compared to their No RAG counter-parts,
with GPT RAG receiving the highest number of
votes (71). Complete statistics for each method are
available in Appendix C.
Effectiveness.Out of the 2340 evaluations col-
lected, 1312 were marked as effective, correspond-
ing to 56% of the cases. More specifically, Llama
RAG was considered effective most frequently
(with 197 votes), followed closely by GPT RAG
(191) and CommandR RAG (172). The baseline
method MTCo received the fewest effectiveness
votes (48) (see Appendix C for full results). This
indicates that the RAG-based methods generally
produced more effective CS than their non-RAG
counterparts and the baseline. The non-RAG mod-
els and MTCo were less effective overall. This
indicates that the RAG-based methods generally
produced more effective CS than their non-RAG
counterparts and the baseline, suggesting that both
model choice and augmentation strategy strongly
influence performance.
In conclusion, RAG-based methods outperform
non-RAG counterparts across all the four metrics,
especially in Factuality, Cogency and Effective-
ness, and are preferred by annotators in best-choice
votes. The baseline method MT-Co consistently
scores lowest across all metrics and effectiveness.
Overall,GPT RAGis the best CS method across
all HS and annotators. Results from the human
evaluation are aligned with both automatic metrics
and JUDGELM evaluations.
7

Model vs Baseline Target Total Wins (ours) % Wins
vs Russo (2025)JEWS 23 22 95.7
LGBT+ 19 17 89.5
MIGRANTS 30 30 100.0
POC 15 11 73.3
WOMEN 37 34 91.9
Total124 11493.0
vs Wilk et al. (2025)DISABLED 36 25 69.4
JEWS 204 102 50.0
LGBT+ 164 105 64.0
MIGRANTS 310 176 56.8
MUSLIMS 518 286 55.2
POC 109 58 53.2
WOMEN 232 142 61.2
Other 124 68 54.8
Total1697 96257.0
Table 6: Per-target JudgeLM comparison of our systems
against Russo (2025) and Wilk et al. (2025). Results
show the number and percentage of pairwise wins across
HS target groups, with totals included for each baseline.
9 Comparison with competitors
To contextualize our results, we compared our
RAG-based CS with publicly available samples
from two recent studies. Wilk et al. (2025) (Base-
line 1) released 1,697 GPT-4o-generated CS re-
sponses from their RAG pipeline addressing MT-
Co hate speech, while Russo (2025) (Baseline 2)
submitted 400 CS samples to the COLING 2025
shared task using LLaMA-EUS-8B with retrieved
knowledge. Since the samples were multilingual,
we retained only English CS aligned with MT-Co,
totaling 124 samples.7For fair comparison, we
reproduced settings closest to the original models:
GPT-4o-mini + BGE-M3 for Wilk et al. (2025),
and LLaMA-8B + BGE-M3 for Russo (2025). This
alignment minimizes differences due to model size,
isolating the effects of retrieval and pipeline de-
sign. Pairwise performance was then assessed with
JUDGELM (prompt available in Appendix B). It is
fine-tuned to rate helpfulness, relevance, accuracy,
and level of detail of CS responses, which already
accounts for evidence and factuality in the original
prompt. To additionally emphasize conciseness,
and suitability for real-world deployment on so-
cial media, we limit the CS to a maximum of two
sentences.
Table 6 shows that our RAG-augmented CS con-
sistently outperforms the baseline samples from
both studies. GPT-4o-mini + BGE-M3 outperforms
Baseline 1 in 962 out of 1,697 pairwise battles,
while LLaMA-8B + BGE-M3 outperforms Base-
7Russo (2025) produced 100 English CS, 24 HS instances
overlapped with MT-CONAN and were retained with multiple
CS variants.line 2 in 114 out of 124 battles. These results
demonstrate that our RAG pipeline produces CS
that is not only factually richer but also concise and
suitable for social media, aligning with the intended
communicative goals of real-world deployment.
Per-Target results.Table 6 also reports the per-
target breakdown of our pairwise comparison.
Against Baseline 1, our system achieves over-
all consistent improvements. Gains are observed
across all HS targets, with particularly strong
performance forDISABLED(69.4%),LGBT+
(64.0%), andWOMEN(61.2%). For larger target
categories such asMUSLIMSandMIGRANTS, our
model still maintains a clear advantage with 55.2%
and 56.8% wins respectively, despite the higher
difficulty and variability in these groups. The
most balanced outcome is seen forJEWS, where
results are split evenly (50%). Against Baseline
2, our system achieves consistently strong results
across all targets, outperforming it in 93% of cases.
The largest margins are observed forMIGRANTS
(100% wins) andJEWS(95.7%).
These results indicate that while our models con-
sistently outperform prior baselines, the improve-
ment varies by target group. In particular, our
LLaMA-based system demonstrates decisive supe-
riority over Baseline 2 comparable setup, whereas
the GPT-based comparison with Baseline 1 high-
lights more incremental but robust gains across di-
verse HS categories, highlighting the effectiveness
of our RAG approach in improving both informa-
tiveness and practical usability of the CS.
10 Conclusion
In this paper, we propose a RAG-based frame-
work for automatic counter-speech generation. We
systematically compared three retrieval methods
and four LLMs for CS generation targeting height
groups (women, people of colour, persons with dis-
abilities, migrants, Muslims, Jews, LGBT persons,
other), relying on a novel and unique knowledge
base, built over three institutional sources. We
conducted an extensive experimental evaluation
against existing state-of-the-art systems (Russo,
2025; Wilk et al., 2025) using JudgeLM, and a
human evaluation. Our results show that our ap-
proach outperforms competitive approaches and
standard baselines on both of them. Our experi-
ments demonstrated the versatility and soundness
of our framework for counter-speech generation to
fight online abusive content.
8

Limitations
First, our retrieval process operates at the paragraph
level, meaning documents in the knowledge base
are split into shorter segments rather than used in
full. This improves efficiency but may potentially
fragment contextual information, omitting relevant
background or nuance. Moreover, we restrict the
retrieved context to the top- k= 3 most similar
paragraphs to control input length and maintain
conciseness in generation. Although this design
balances informativeness and computational effi-
ciency, varying kcould influence the factual rich-
ness and diversity of the generated counter-speech.
Second, despite employing strong retrievers
(BM25, SBERT, and BGE-M3), retrieval qual-
ity depends on the coverage and relevance of the
knowledge base. Gaps or biases in external sources
can propagate into the generated responses, particu-
larly for emerging or culturally specific hate topics,
even if we mitigated this by choosing recognized
authoritative sources.
Third, LLMs may still produce partially hallu-
cinated or stylistically inconsistent outputs, espe-
cially when retrieved evidence is noisy or ambigu-
ous. Limiting counter-speech length to two sen-
tences enhances realism and readability but can
also reduce nuance and emotional depth in the gen-
erated CS.
11 Ethical Statement
This study involves the use of hate speech exam-
ples for the development and evaluation of counter-
speech generation systems. We acknowledge that
the inclusion of HS content poses potential risks of
exposure to harmful language and emotional dis-
tress for researchers and annotators. All individuals
involved in data handling were informed of these
risks and participated voluntarily, following insti-
tutional ethical guidelines. Although our goal is to
promote positive and factual discourse, automatic
CS generation can inadvertently reinforce biases,
produce factually incorrect content, or convey un-
intended tones. To mitigate these risks, we rely on
institutional sources (UN, EU, FRA) for retrieval,
explicitly evaluate factuality and correctness, and
use human oversight in all analyses. The system
is presented for research purposes only and is not
intended for unsupervised deployment. We further
recognize that the perceived effectiveness and ap-
propriateness of CS depend on social and cultural
context. Our methods and findings should thereforenot be generalized without careful adaptation and
ethical review. No personal or private data were
used; all retrieved materials come from publicly
available institutional sources.
References
Mana Ashida and Mamoru Komachi. 2022. Towards
automatic generation of messages countering online
hate speech and microaggressions. InProceedings
of the Sixth Workshop on Online Abuse and Harms
(WOAH), pages 11–23.
Satanjeev Banerjee and Alon Lavie. 2005. Meteor: An
automatic metric for mt evaluation with improved cor-
relation with human judgments. InProceedings of
the acl workshop on intrinsic and extrinsic evaluation
measures for machine translation and/or summariza-
tion, pages 65–72.
Susan Benesch. 2014. Countering dangerous speech:
New ideas for genocide prevention.SSRN Electronic
Journal. Available at SSRN 3686876.
Susan Benesch, Derek Ruths, Kelly P. Dillon, Haji Mo-
hammad Saleem, and Lucas Wright. 2016. Counter-
speech on twitter: A field study. Technical report,
Dangerous Speech Project.
Jaione Bengoetxea, Yi-Ling Chung, Marco Guerini,
and Rodrigo Agerri. 2024. Basque and Spanish
counter narrative generation: Data creation and evalu-
ation. InProceedings of the 2024 Joint International
Conference on Computational Linguistics, Language
Resources and Evaluation (LREC-COLING 2024),
pages 2132–2141, Torino, Italia. ELRA and ICCL.
Nicola Bertoldi, Mauro Cettolo, and Marcello Federico.
2013. Cache-based online adaptation for machine
translation enhanced computer assisted translation.
InProceedings of Machine Translation Summit XIV:
Papers, Nice, France.
Helena Bonaldi, Giuseppe Attanasio, Debora Nozza,
and Marco Guerini. 2023. Weigh your own words:
Improving hate speech counter narrative generation
via attention regularization. InProceedings of the
1st Workshop on CounterSpeech for Online Abuse
(CS4OA), pages 13–28, Prague, Czechia. Association
for Computational Linguistics.
Helena Bonaldi, Greta Damo, Nicolás Benjamín
Ocampo, Elena Cabrio, Serena Villata, and Marco
Guerini. 2024. Is safer better? the impact of
guardrails on the argumentative strength of LLMs in
hate speech countering. InProceedings of the 2024
Conference on Empirical Methods in Natural Lan-
guage Processing, pages 3446–3463, Miami, Florida,
USA. Association for Computational Linguistics.
Helena Bonaldi, Sara Dellantonio, Serra Sinem
Tekiro ˘glu, and Marco Guerini. 2022. Human-
machine collaboration approaches to build a dialogue
dataset for hate speech countering. InProceedings
9

of the 2022 Conference on Empirical Methods in
Natural Language Processing, pages 8031–8049. As-
sociation for Computational Linguistics.
Helena Bonaldi, María Estrella Vallecillo-Rodríguez,
Irune Zubiaga, Arturo Montejo-Ráez, Aitor Soroa,
María Teresa Martín-Valdivia, Marco Guerini, Ro-
drigo Agerri, and 1 others. 2025. The first workshop
on multilingual counterspeech generation at coling
2025: Overview of the shared task. InProceedings
of the First Workshop on Multilingual Counterspeech
Generation, pages 92–107.
Mauro Cettolo, Nicola Bertoldi, and Marcello Federico.
2014. The repetition rate of text as a predictor of the
effectiveness of machine translation adaptation. In
Proceedings of the 11th Conference of the Associa-
tion for Machine Translation in the Americas: MT
Researchers Track, pages 166–179.
Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu
Lian, and Zheng Liu. 2024. Bge m3-embedding:
Multi-lingual, multi-functionality, multi-granularity
text embeddings through self-knowledge distillation.
arXiv preprint arXiv:2402.03216.
Zewen Chi, Li Dong, Furu Wei, Wenhui Wang, Xian-
Ling Mao, and Heyan Huang. 2020. Cross-lingual
natural language generation via pre-training. InPro-
ceedings of the AAAI Conference on Artificial Intelli-
gence, volume 34, pages 7570–7577.
Yi-Ling Chung, Elizaveta Kuzmenko, Serra Sinem
Tekiro ˘glu, and Marco Guerini. 2019. Conan –
counter narratives through nichesourcing: a multilin-
gual dataset of responses to fight online hate speech.
InProceedings of the 57th Annual Meeting of the As-
sociation for Computational Linguistics, pages 2819–
2829. Association for Computational Linguistics.
Yi-Ling Chung, Serra Sinem Tekiro ˘glu, and Marco
Guerini. 2021a. Towards knowledge-grounded
counter narrative generation for hate speech. InFind-
ings of the Association for Computational Linguistics:
ACL-IJCNLP 2021, pages 899–914, Online. Associa-
tion for Computational Linguistics.
Yi-Ling Chung, Serra Sinem Tekiro ˘glu, Sara Tonelli,
and Marco Guerini. 2021b. Empowering ngos in
countering online hate messages.Online Social Net-
works and Media, 24:100150.
Team Cohere, Aakanksha, Arash Ahmadian, Marwan
Ahmed, Jay Alammar, Yazeed Alnumay, Sophia Al-
thammer, Arkady Arkhangorodsky, Viraat Aryabumi,
Dennis Aumiller, Raphaël Avalos, Zahara Aviv, Sam-
mie Bae, Saurabh Baji, Alexandre Barbet, Max Bar-
tolo, Björn Bebensee, Neeral Beladia, Walter Beller-
Morales, and 207 others. 2025. Command a: An
enterprise-ready large language model.Preprint,
arXiv:2504.00698.
Greta Damo, Elena Cabrio, and Serena Villata. 2025.
Effectiveness of counter-speech against abusive con-
tent: A multidimensional annotation and classifica-
tion study.arXiv preprint arXiv:2506.11919.Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff
Johnson, Gergely Szilvasy, Pierre-Emmanuel Mazaré,
Maria Lomeli, Lucas Hosseini, and Hervé Jégou.
2024. The faiss library.
Margherita Fanton, Helena Bonaldi, Serra Sinem
Tekiro ˘glu, and Marco Guerini. 2021. Human-in-
the-loop for data collection: a multi-target counter
narrative dataset to fight online hate speech.arXiv
preprint arXiv:2107.08720.
Max Grusky, Mor Naaman, and Yoav Artzi. 2018.
Newsroom: A dataset of 1.3 million summaries
with diverse extractive strategies.arXiv preprint
arXiv:1804.11283.
Rishabh Gupta, Shaily Desai, Manvi Goel, Anil Band-
hakavi, Tanmoy Chakraborty, and Md. Shad Akhtar.
2023. Counterspeeches up my sleeve! intent dis-
tribution learning and persistent fusion for intent-
conditioned counterspeech generation. InProceed-
ings of the 61st Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers),
pages 5792–5809, Toronto, Canada. Association for
Computational Linguistics.
Shuyu Jiang, Wenyi Tang, Xingshu Chen, Rui Tang,
Haizhou Wang, and Wenxian Wang. 2025. Rezg:
Retrieval-augmented zero-shot counter narrative
generation for hate speech.Neurocomputing,
620:129140.
Shuyu Jiang, Wenyi Tang, Xingshu Chen, Rui Tanga,
Haizhou Wang, and Wenxian Wang. 2023. Raucg:
Retrieval-augmented unsupervised counter narra-
tive generation for hate speech.arXiv preprint
arXiv:2310.05650.
Jaylen Jones, Lingbo Mo, Eric Fosler-Lussier, and Huan
Sun. 2024. A multi-aspect framework for counter
narrative evaluation using large language models. In
Proceedings of the 2024 Conference of the North
American Chapter of the Association for Computa-
tional Linguistics: Human Language Technologies
(Volume 2: Short Papers), pages 147–168, Mexico
City, Mexico. Association for Computational Lin-
guistics.
Svetlana Kiritchenko, Isar Nejadgholi, and Kathleen C.
Fraser. 2021. Confronting abusive language online:
A survey from the ethical and human rights per-
spective.Journal of Artificial Intelligence Research,
71:431–478.
Amanda Lenhart, Kristen Purcell, Aaron Smith, and
Kathryn Zickuhr. 2010. Social media & mobile inter-
net use among teens and young adults. millennials.
Jiwei Li, Michel Galley, Chris Brockett, Jianfeng Gao,
and Bill Dolan. 2015. A diversity-promoting objec-
tive function for neural conversation models.arXiv
preprint arXiv:1510.03055.
Chin-Yew Lin. 2004. ROUGE: A package for auto-
matic evaluation of summaries. InText Summariza-
tion Branches Out, pages 74–81, Barcelona, Spain.
Association for Computational Linguistics.
10

Binny Mathew, Ritam Dutt, Pawan Goyal, and Animesh
Mukherjee. 2019a. Spread of hate speech in online
social media. InProceedings of the 10th ACM Con-
ference on Web Science, pages 173–182.
Binny Mathew, Punyajoy Saha, Hardik Tharad, Subham
Rajgaria, Prajwal Singhania, Suman Kalyan Maity,
Pawan Goyal, and Animesh Mukherjee. 2019b. Thou
shalt not hate: Countering online hate speech. In
Proceedings of the International AAAI Conference
on Web and Social Media, volume 13, pages 369–
380.
Stephen Merity, Caiming Xiong, James Bradbury, and
Richard Socher. 2016. Pointer sentinel mixture mod-
els.arXiv preprint arXiv:1609.07843.
Mainack Mondal, Leandro Araújo Silva, and Fabrício
Benevenuto. 2017. A measurement study of hate
speech in social media. InProceedings of the 28th
ACM Conference on Hypertext and Social Media,
pages 85–94.
Esteban Ortiz-Ospina. 2019. Over 2.5 billion people
use social media. this is how it has changed the world.
World Economic Forum Agenda.
Kishore Papineni, Salim Roukos, Todd Ward, and Wei-
Jing Zhu. 2002. Bleu: a method for automatic evalu-
ation of machine translation. InProceedings of the
40th Annual Meeting of the Association for Compu-
tational Linguistics, pages 311–318.
Jing Qian, Anna Bethke, Yinyin Liu, Elizabeth Belding,
and William Yang Wang. 2019. A benchmark dataset
for learning to intervene in online hate speech.arXiv
preprint arXiv:1909.04251.
Alec Radford, Jeff Wu, Rewon Child, David Luan,
Dario Amodei, and Ilya Sutskever. 2019. Language
models are unsupervised multitask learners. Techni-
cal report, OpenAI.
Nils Reimers and Iryna Gurevych. 2019. Sentence-bert:
Sentence embeddings using siamese bert-networks.
InProceedings of the 2019 Conference on Empirical
Methods in Natural Language Processing and the
9th International Joint Conference on Natural Lan-
guage Processing (EMNLP-IJCNLP), pages 3982–
3992, Hong Kong, China. Association for Computa-
tional Linguistics.
Stephen Robertson and Hugo Zaragoza. 2009. The prob-
abilistic relevance framework: Bm25 and beyond.
Foundations and Trends® in Information Retrieval,
3:333–389.
Daniel Russo. 2025. Trenteam at multilingual counter-
speech generation: Multilingual passage re-ranking
approaches for knowledge-driven counterspeech gen-
eration against hate. InProceedings of the First
Workshop on Multilingual Counterspeech Genera-
tion, pages 77–91.Daniel Russo, Shane Kaszefski-Yaschuk, Jacopo Sta-
iano, and Marco Guerini. 2023. Countering misin-
formation via emotional response generation. InPro-
ceedings of the 2023 Conference on Empirical Meth-
ods in Natural Language Processing, pages 11476–
11492, Singapore. Association for Computational
Linguistics.
Punyajoy Saha, Kanishk Singh, Adarsh Kumar, Binny
Mathew, and Animesh Mukherjee. 2022. Coun-
tergedi: A controllable approach to generate po-
lite, detoxified and emotional counterspeech.arXiv
preprint arXiv:2205.04304.
Furkan ¸ Sahinuç and Cagri Toraman. 2021. Tweet length
matters: A comparative analysis on topic detection in
microblogs. InEuropean Conference on Information
Retrieval, pages 471–478. Springer.
Carla Schieb and Mike Preuss. 2016. Governing hate
speech by means of counterspeech on facebook. In
66th ICA Annual Conference, pages 1–23, Fukuoka,
Japan.
Waralak V . Siricharoen. 2023. Social media as
communication–transformation tools. InInformation
Systems for Intelligent Systems, pages 1–11. Springer
Nature Singapore.
Paul Stapleton and Yanming Amy Wu. 2015. Assessing
the quality of arguments in students’ persuasive writ-
ing: A case study analyzing the relationship between
surface structure and substance.Journal of English
for Academic Purposes, 17:12–23.
Serra Sinem Tekiro ˘glu, Helena Bonaldi, Margherita
Fanton, and Marco Guerini. 2022. Using pre-trained
language models for producing counter narratives
against hate speech: a comparative study. InFind-
ings of the Association for Computational Linguis-
tics: ACL 2022, pages 3099–3114, Dublin, Ireland.
Association for Computational Linguistics.
Serra Sinem Tekiro ˘glu, Yi-Ling Chung, and Marco
Guerini. 2020. Generating counter narratives against
online hate speech: Data and strategies. InProceed-
ings of the 58th Annual Meeting of the Association
for Computational Linguistics, pages 1177–1190. As-
sociation for Computational Linguistics.
Ke Wang and Xiaojun Wan. 2018. Sentigan: Generating
sentimental texts via mixture adversarial networks.
InIJCAI, pages 4446–4452.
Brian Wilk, Homaira Huda Shomee, Suman Kalyan
Maity, and Sourav Medya. 2025. Fact-based counter
narrative generation to combat hate speech. InPro-
ceedings of the ACM on Web Conference 2025, pages
3354–3365.
Xinchen Yu, Eduardo Blanco, and Lingzi Hong. 2022.
Hate speech and counter speech detection: Conver-
sational context does matter. InProceedings of the
2022 Conference of the North American Chapter of
the Association for Computational Linguistics: Hu-
man Language Technologies, pages 5918–5931.
11

Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q
Weinberger, and Yoav Artzi. 2019. Bertscore: Eval-
uating text generation with bert.arXiv preprint
arXiv:1904.09675.
Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang,
Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen
Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen
Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang,
Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, and
3 others. 2023. A survey of large language models.
arXiv preprint arXiv:2303.18223.
Ming Zhong, Yang Liu, Da Yin, Yuning Mao, Yizhu
Jiao, Pengfei Liu, Chenguang Zhu, Heng Ji, and
Jiawei Han. 2022. Towards a unified multi-
dimensional evaluator for text generation. InPro-
ceedings of the 2022 Conference on Empirical Meth-
ods in Natural Language Processing, pages 2023–
2038, Abu Dhabi, United Arab Emirates. Association
for Computational Linguistics.
Lianghui Zhu, Xinggang Wang, and Xinlong Wang.
2023. Judgelm: Fine-tuned large language
models are scalable judges.arXiv preprint
arXiv:2310.17631.
Wanzheng Zhu and Suma Bhat. 2021. Generate, prune,
select: A pipeline for counterspeech generation
against online hate speech. InFindings of the Associ-
ation for Computational Linguistics: ACL-IJCNLP
2021, pages 134–149, Online. Association for Com-
putational Linguistics.
Philip G. Zimbardo. 1969. The human choice: Indi-
viduation, reason, and order versus deindividuation,
impulse, and chaos. InNebraska Symposium on Mo-
tivation. University of Nebraska Press.
Irune Zubiaga, Aitor Soroa, and Rodrigo Agerri. 2024.
A llm-based ranking method for the evaluation of au-
tomatic counter-narrative generation.arXiv preprint
arXiv:2406.15227.
A Keywords Used for the Knowledge
Base Queries
The following keywords were used to retrieve doc-
uments from the United Nations Digital Library,
the FRA and EUR-Lex portals, covering the pe-
riod 2000–2025 and targeting multiple groups and
thematic areas relevant to our study:
Target Groups:
•People of Color: People of color, Racism,
Anti-Black racism, Systemic racism, Racial
inequality, Racial equality, Racial profiling,
White privilege, Black Lives Matter, Colo-
nialism, Racial discrimination, discrimina-
tion against black people, blacks, race, black
women, black people, blacks hate speech,african descent, ethnic minorities, ethnic in-
equalities, minority
•LGBT: LGBT rights, Homophobia, Trans-
phobia, Biphobia, Gender identity, Conver-
sion therapy, Same-sex marriage, Stonewall ri-
ots, LGBT, LGBTQIA+, Gay, Lesbian, Trans-
gender, Non-binary, LGBT hate speech, dis-
crimination against LGBT people, gay rights
movement, LGBT discrimination, LGBT hate
crimes, sexual orientation, HIV/AIDS &
gay/lesbian
•Disabled: Disability rights, Accessibility, So-
cial model of disability, disability, disabled,
down syndrome, autism, mental disability,
physical disability, neurodiversity, ableism, in-
clusive design, Discrimination against people
with disabilities
•Muslims: Islamophobia, Discrimination
against Muslims, Anti-Muslim hate crimes,
Muslim communities, Religious discrimina-
tion, islam, muslim, muslim hate speech, reli-
gion, discrimination against muslims, muslim
communities
•Jews: Antisemitism, Jewish identity, Anti-
Jewish violence, Nazi propaganda, jews, jews
hate speech, antisemitism hate speech, ju-
daism, hebrews, jews hate crimes, jewish his-
tory, jewish diaspora, zionism, zionist move-
ment, holocaust, israel, holocaust denial
•Women: Sexism, Misogyny, Feminism, Gen-
der inequality, Women’s rights, Me Too move-
ment, Gender-based violence, women, women
hate speech, feminism, violence against
women, gender inequality, glass ceiling, dis-
crimination against women
•Migrants: Xenophobia, Anti-immigration,
Refugee crisis, Asylum seekers, Undocu-
mented immigrants, Immigration law, refugee,
migrants, immigrants, immigration, migra-
tion, immigration hate speech, migrants rights,
illegal aliens, immigration and crime, immi-
gration and unemployment, discrimination
against migrants, aliens
Thematic Categories:
•Hate Speech: hate speech, hate speech laws,
hate crime, hate crime legislation, hate speech
regulation, hate speech prevention, online hate
12

speech, online harassment, cyberbullying, cen-
sorship, freedom of expression, speech ethics,
disinformation, radicalization, extremism, on-
line moderation
•Human Rights and Law: human rights, hu-
man rights treaties, universal declaration of
human rights, international human rights law,
civil rights, social justice, equality before the
law, international court of justice, refugee
rights, minority rights, gender equality law,
european charter of human rigths
B Prompts used
In this section, we list all the prompts used in our
experiments. All summarization and generation ex-
periments were conducted on an A100 GPU, with
the following parameters: max_new_tokens =150,
andtemperature =0.5. For GPT-based models the
cost was≈10$.
To expand the set of keywords for the knowledge
base creation, we prompt GPT-5 with the following:
Prompt for keyword retrieval
We are conducting a thematic search across
major institutional databases — the UN Dig-
ital Library, EUR-Lex (EU law), and the EU
Agency for Fundamental Rights (FRA). Our
goal is to identify documents, resolutions,
laws, and reports related to discrimination,
rights, and inclusion of specific social groups
that are the targets of MultiTarget CONAN
dataset:women, people of color, persons
with disabilities, migrants, Muslims, Jews,
LGBT persons, other marginalized or minor-
ity groups.
Please suggest related and institutionally ac-
curate keywords or controlled vocabulary
terms used in UN, EU, and FRA contexts
for each target group. For each group, in-
clude: Formal institutional terminology (used
in UN or EU documents); Common synonyms
and variants (used in academic or policy
discourse); Relevant policy or legal frame-
works (e.g., conventions, charters, declara-
tions); Suggested Boolean search strings for
use in UN Digital Library, EUR-Lex, and
FRA databases. Prioritize terminology that
aligns with international human rights lan-
guage and European policy frameworks (e.g.,
CERD, CEDAW, CRPD, FRA reports).For Step 2 of our pipeline (document summa-
rization), for each of the 3 paragraphs, we prompt
the LLMs with the following:
Summarization prompt
Summarize this paragraph paragraph_text
into exactly 2 sentences, without truncating
the last sentence:
Summary:
For Step 3 of our pipeline (CS generation), we
prompt all LLMs with the following:
CS generation W/OUT RAG
Hateful message: hateful_message Gener-
ate a counter-speech that is concise, respect-
ful, persuasive, self-contained, and suitable
for social media. Keep it short (maximum 2
sentences). Do NOT truncate the last sentence.
Counter-speech:
CS generation WITH RAG
Here are three evidence summaries you MUST
use to inform your response:context.
Task: Respond to the hateful message below
by writing a short counter-speech that is re-
spectful, persuasive, self-contained, and suit-
able for social media (maximum 2 sentences).
Ground your response in the evidence above
and do NOT truncate the last sentence.
Hateful message:hateful_message
Counter-speech:
In order to use JUDGELM as a metric, we used the
following prompts. The first one is used to com-
pare RAG methods against No RAG methods, and
against the MT-Co baseline. The second prompt is
used to compare our RAG-based CS against those
generated with competitive approaches (Wilk et al.,
2025; Russo, 2025).
1. JudgeLM — RAG vs No RAG
Provide a brief counter-narrative in response
to the following hate speech: {hs} . You are
an impartial evaluator. You will be given a
piece of hate speech and two counter-speech
responses. Your task is to judge which counter-
speech is better according to the following
criteria, in order of importance:
1)Factuality– Does the response contain ac-
curate and verifiable information? A factu-
13

ally correct response is always preferred. 2)
Number of facts– Does the response provide
multiple relevant facts, not just one? More
well-chosen facts are better. 3)Relevance–
Does the response directly address the hate
speech? 4)Specificity– Is the response spe-
cific and non-generic?
2. JudgeLM — Method Comparison
You are an impartial evaluator. You will be
given a piece of hate speech {hs} and two
counter-speech responses. Your task is to
judge which response is better according to
the following criteria:
- The response must be easy to read, empa-
thetic, and suitable as a short social media
comment. - Friendly and colloquial language
is preferred. Responses that sound like essays,
lectures, or academic articles should be pe-
nalized. - Conciseness: The response should
be maximum 2 sentences long. Very long re-
sponses should be penalized, even if factually
rich.
CS Method Effectiveness “Yes” V otes Best CS V otes
Llama RAG19767
GPT RAG 19171
CommandR RAG 172 26
Llama No RAG 171 33
Mistral RAG 156 24
GPT No RAG 148 14
CommandR No RAG 120 7
Mistral No RAG 109 11
MTCo 48 7
Table 7: Comparison of counter-speech methods show-
ing both the number of times a CS was voted “Yes” for
Effectiveness and the number of Best Choice votes re-
ceived.
C Additional Results from the Human
Evaluation
Table 7 shows the total number of times each CS
method was rated effective, and how many times
it has been selected as the best one across all HS
and annotators. RAG-based methods are clearly
preferred compared to their No RAG counter-parts.
Concerning effectiveness, Llama RAG was consid-
ered effective most frequently (197 “Yes” votes),
followed closely by GPT RAG (191) and Com-
mandR RAG (172). The baseline method MTCo
received the fewest effective votes (48). This indi-
cates that RAG-based methods generally producedmore effective CS than their non-RAG counterparts
and the baseline. Furthermore, RAG methods are
selected more often as best ones compared to their
No RAG counterparts, with GPT RAG and Llama
RAG achieving the best results.
D Statistical Significance of Retrieval
Effects
We conducted non-parametric Friedman tests fol-
lowed by Bonferroni-corrected Wilcoxon signed-
rank tests to evaluate whether RAG retrieval
strategies (BM25, Sentence-BERT, BGE) signif-
icantly affected model outputs compared to the No-
RAG baseline for the per-sample automatic met-
rics (BLEU, METEOR, ROUGE-L, BERTScore),
which allow direct pairwise comparison across sys-
tems. In contrast, diversity metrics such as Distinct-
1, Distinct-2, and Repetition Rate are computed
at the corpus level, producing a single value per
model. Because these measures do not yield per-
sample scores and thus lack within-system vari-
ance, statistical testing is not applicable. For all
models, Friedman tests revealed significant overall
effects ( p <0.001 ), and pairwise comparisons con-
firmed that retrieval methods consistently induced
statistically significant differences ( p <0.001 )
across quality and safety metrics.
E Participants’ Demographic
Characteristics
In this section, we report the demographic charac-
teristics of the 26 participants of the human evalua-
tion. Figures 3, 4, 5, 6 show the distribution of age
groups, gender, geographical area of origin, and
the highest obtained education level. Age varies
between 18 and 50 years, with the majority group
being between 18 and 35. Gender is evenly dis-
tributed among females and males with one person
identifying as non-binary. The geographical area
of origin covers all major areas, with a majority
of European people. Education level spans from
high school diploma to PhD, with the majority of re-
spondents having pursued a PhD. We also asked the
respondents their area of expertise, which covers
the following fields: Computer Science, Computer
Engineering, Data Science, Psychology, Natural
Language Processing, Linguistics, and Manage-
ment.
Figure 7 shows the target groups in which the
respondents identify. They belong to almost all the
HS targets we considered in our analysis, with the
14

Model Metric FriedmanpNo-RAG vs BM25pNo-RAG vs SentBERTpNo-RAG vs BGEp
LLaMA BLEU2.0×10−3071.1×10−1822.3×10−1441.4×10−132
METEOR4.4×10−397.2×10−35.5×10−119.0×10−34
ROUGE-L1.5×10−339.3×10−377.3×10−188.4×10−9
BERTScore F1 <10−300<10−3006.8×10−2761.8×10−257
Safety3.6×10−175<10−79<10−134<10−173
CommandR BLEU<10−3002.5×10−3079.4×10−2202.5×10−177
METEOR2.2×10−1024.5×10−371.2×10−908.8×10−79
ROUGE-L3.5×10−314.9×10−303.2×10−71.5×10−1
BERTScore F1 <10−300<10−300<10−300<10−300
Safety<10−300<10−185<10−300<10−300
Mistral BLEU<10−3004.9×10−2432.1×10−1946.6×10−124
METEOR7.1×10−181.8×10−45.4×10−102.3×10−21
ROUGE-L2.9×10−586.5×10−582.9×10−375.4×10−11
BERTScore F1 <10−300<10−300<10−3004.4×10−127
Safety<10−300<10−217<10−251<10−281
GPT-4 BLEU<10−300<10−3005.5×10−2993.6×10−226
METEOR1.9×10−971.6×10−391.5×10−633.4×10−92
ROUGE-L1.1×10−531.5×10−514.0×10−341.6×10−6
BERTScore F1 <10−300<10−300<10−3001.8×10−205
Safety<10−300<10−207<10−300<10−300
Table 8: Statistical significance (Friedman and Bonferroni-corrected Wilcoxon tests) comparing No-RAG against
retrieval-based setups for all metrics. Extremely small p-values ( <10−300) indicate strong evidence that retrieval
methods significantly affect generation outcomes.
Figure 3: Distribution of age of the 26 participants.
g
Figure 4: Distribution of gender of the 26 participants.
majority of them identifying as women.
F Guidelines for Human Evaluation
As described in Section 6, we carried out a human
evaluation. Participants were voluntarily recruited
and could opt out from the study at any time. Pre-
cise instructions were given, highlighting the po-
tential risks and distress of the study. We define the
Figure 5: Distribution of the geographical area of origin
of the 26 participants.
Figure 6: Distribution of level of education of the 26
participants.
following metrics used for the human evaluation:
Relevance, Factuality, Cogency, and Correctness
use a Likert scale from 1 to 3 (with 3 being the
best possible score), while Effectiveness, and Is the
Best are binary dimensions to which the partici-
pants could reply "Yes" or "No". They are defined
as follows:
15

Figure 7: Distribution of the target group identification
of the 26 participants.
Relevance.How relevant is the counter-speech
to the hate speech, in terms of topic and target of
hate (i.e. the offended minority)?
3.The CS addresses both the correct HS target
and the topic.
2.The CS addresses correctly only the topic or
the target of hate.
1.The CS is very general: the same message
could reply to whatever HS.
Factuality.How informative is the counter-
speech as a response to its hate speech, in terms of
quantity and factual correctness of included facts?
3.The CS provides multiple logically correct
arguments and they are all sound and relevant.
2.The CS provides only one logically correct
argument which is sound and relevant.
1.No reasons are provided for the CS claim, or
none of the reasons are relevant to or support
the CS claim.
Cogency.This dimension measures the quantity
of the supporting logically correct arguments pro-
vided by the counter-speech.
3.The CS provides multiple information not
present in the HS, and they are all factually
correct.
2.The CS provides only one information not
present in the HS, and it is correct.
1.The CS provides no additional information
with respect to the HS.Correctness.How much is the style of the CS
correct? I.e., the CS is free of grammatical and
syntactical errors and it is not hateful (does not
contain toxic language).
3.The CS has not grammatical or syntactical
errors AND does not have toxic language.
2.The CS has grammatical and syntactical errors
OR it has a toxic language.
1.The CS has grammatical or syntactical errors
AND has a toxic language.
Effectiveness.Is the counter-speech likely to be
persuasive and change someone’s perspective on
the issue? Is the CS able to change the opinion of
the author of the HS?
Is the Best?Refers to whether the counter-
speech is the best one among the ones generated for
the same HS. Select the CS that you prefer based
on the scores you provided.
We also provided the following examples to better
understand the metrics’ scores.
HS example:“Immigrants are lazy and just
come here to steal jobs.”
1. Relevance
Score 3 (Topic + Target correct):
“That’s not true—immigrants contribute
greatly to the economy by working hard in
sectors like healthcare and construction.”
Score 2 (Only topic or target correct):
“Jobs are important for everyone, and we
should all value hard work.”(Topic correct,
but not addressing immigrants directly.)
“Immigrants deserve respect.”(Target
correct, but no engagement with the specific
topic of jobs/laziness.)
Score 1 (Very general):
“We should all just be kind to each other.”
2. Correctness
Score 3 (No errors, no toxicity):
“Immigrants are not lazy—many work
multiple jobs to support their families.”
Score 2 (Some errors OR mild toxicity,
but not both):
16

“Immigrants isn’t lazy, they works a lot.”
(Grammatical errors but not offensive.)
“That’s a dumb thing to say, immigrants
work hard.”(Toxic tone but grammatically
correct.)
Score 1 (Errors AND toxic):
“U r idiot, immigrants aren’t ur slaves.”
3. Factuality
Score 3 (Multiple correct facts):
“Immigrants make up 17% of the healthcare
workforce in the US and many work in
agriculture, which keeps food affordable.”
Score 2 (One correct fact):
“Studies show that immigrants often take
jobs that citizens don’t want, like farm
labor.”
Score 1 (No new info):
“Immigrants work hard.”(Assertion without
factual support.)
4. Cogency
Score 3 (Multiple sound reasons):
“It’s unfair to call immigrants lazy—many
work multiple jobs, they often contribute
more in taxes than they receive in benefits,
and they fill critical roles in the economy.”
Score 2 (One sound reason):
“It’s unfair to call immigrants lazy because
many of them work long hours.”
Score 1 (No or irrelevant reasons):
“Immigrants are great, stop hating.”(Claim,
but no logical support.)
G Examples of Generated
Counter-speech
Here, we provide different examples of counter-
speech addressing the same hateful message from
MT-CONAN. We show CS from GPT No RAG
and GPT RAG with BGE-M3.
HS: Their power to have movies banned and
directors censured is proof that the Jews control
Hollywood.MT-Co CS: The ultra rich people in this indus-
try - Jewish or not - have the power and influ-
ence. Not sure where you got your facts from
regarding Jewish people "controlling" Holly-
wood.
GPT No RAG CS: Attributing the actions of a
few individuals to an entire group perpetuates
harmful stereotypes and ignores the diverse
voices that contribute to the film industry. Let’s
focus on promoting understanding and cele-
brating the rich tapestry of talent in Hollywood,
rather than spreading division.
GPT RAG (BGE-M3) CS: It’s important to
recognize that attributing the actions of indi-
viduals or organizations in Hollywood to an
entire group, such as Jews, perpetuates harm-
ful stereotypes that echo historical anti-Semitic
propaganda like "The Protocols of the Elders of
Zion." Instead of fostering division, we should
advocate for a more nuanced understanding of
the complex dynamics in the film industry and
promote equal rights and representation for all
communities.
17