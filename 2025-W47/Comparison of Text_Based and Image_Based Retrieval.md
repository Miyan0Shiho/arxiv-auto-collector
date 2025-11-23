# Comparison of Text-Based and Image-Based Retrieval in Multimodal Retrieval Augmented Generation Large Language Model Systems

**Authors**: Elias Lumer, Alex Cardenas, Matt Melich, Myles Mason, Sara Dieter, Vamse Kumar Subbiah, Pradeep Honaganahalli Basavaraju, Roberto Hernandez

**Published**: 2025-11-20 18:56:49

**PDF URL**: [https://arxiv.org/pdf/2511.16654v1](https://arxiv.org/pdf/2511.16654v1)

## Abstract
Recent advancements in Retrieval-Augmented Generation (RAG) have enabled Large Language Models (LLMs) to access multimodal knowledge bases containing both text and visual information such as charts, diagrams, and tables in financial documents. However, existing multimodal RAG systems rely on LLM-based summarization to convert images into text during preprocessing, storing only text representations in vector databases, which causes loss of contextual information and visual details critical for downstream retrieval and question answering. To address this limitation, we present a comprehensive comparative analysis of two retrieval approaches for multimodal RAG systems, including text-based chunk retrieval (where images are summarized into text before embedding) and direct multimodal embedding retrieval (where images are stored natively in the vector space). We evaluate all three approaches across 6 LLM models and a two multi-modal embedding models on a newly created financial earnings call benchmark comprising 40 question-answer pairs, each paired with 2 documents (1 image and 1 text chunk). Experimental results demonstrate that direct multimodal embedding retrieval significantly outperforms LLM-summary-based approaches, achieving absolute improvements of 13% in mean average precision (mAP@5) and 11% in normalized discounted cumulative gain. These gains correspond to relative improvements of 32% in mAP@5 and 20% in nDCG@5, providing stronger evidence of their practical impact. We additionally find that direct multimodal retrieval produces more accurate and factually consistent answers as measured by LLM-as-a-judge pairwise comparisons. We demonstrate that LLM summarization introduces information loss during preprocessing, whereas direct multimodal embeddings preserve visual context for retrieval and inference.

## Full Text


<!-- PDF content starts -->

Comparison of T ext-Based and Image-Based Retrieval in Multimodal
Retrieval Augmented Generation Large Language Model Systems
Elias Lumer, Alex Cardenas, Matt Melich, Myles Mason, Sara Dieter,
V amse Kumar Subbiah, Pradeep Honaganahalli Basavaraju, and Roberto Hernandez
PricewaterhouseCoopers U.S.
Keywords:
Multimodal RAG, Jina v4, Image Embeddings, V ector Search, LLM-as-Judge
Abstract:
Recent advancements in Retrieval-Augmented Generation (RAG) have enabled Large Language
Models (LLMs) to access multimodal knowledge bases containing both text and visual informa-
tion such as charts, diagrams, and tables in financial documents. However, existing multimodal
RAG systems rely on LLM-based summarization to convert images into text during preprocessing,
storing only text representations in vector databases, which causes loss of contextual informa-
tion and visual details critical for downstream retrieval and question answering. T o address this
limitation, we present a comprehensive comparative analysis of two retrieval approaches for mul-
timodal RAG systems, including text-based chunk retrieval (where images are summarized into
text before embedding) and direct multimodal embedding retrieval (where images are stored na-
tively in the vector space). W e evaluate all three approaches across 6 LLM models and a two
multi-modal embedding models on a newly created financial earnings call benchmark comprising
40 question-answer pairs, each paired with 2 documents (1 image and 1 text chunk). Experimental
results demonstrate that direct multimodal embedding retrieval significantly outperforms LLM-
summary-based approaches, achieving absolute improvements of 13% in mean average precision
(mAP@5) and 11% in normalized discounted cumulative gain. These gains correspond to relative
improvements of 32% in mAP@5 and 20% in nDCG@5, providing stronger evidence of their prac-
tical impact. W e additionally find that direct multimodal retrieval produces more accurate and
factually consistent answers as measured by LLM-as-a-judge pairwise comparisons. W e demon-
strate that LLM summarization introduces information loss during preprocessing, whereas direct
multimodal embeddings preserve visual context for retrieval and inference.
1 INTRODUCTION
Recent advancements in Large Language Mod-
els (LLMs) have enabled powerful question an-
swering systems that leverage external knowl-
edge bases through Retrieval-Augmented Gen-
eration (RAG) (Lewis et al., 2020; Gao et al.,
2024; Huang et al., 2024a). With RAG, these
systems can retrieve relevant information from
vector databases and inject that context into
the model’s prompt at inference time, improv-
ing factual accuracy and reducing hallucina-
tions. Current RAG systems handle text docu-
ments effectively through dense retrieval methods
(Karpukhin et al., 2020), though they face sig-
nificant challenges when applied to multimodal
documents containing both text and visual infor-mation such as charts, diagrams, and tables in
financial reports or presentations.
Despite advancements in RAG pipelines for
text-based retrieval, a significant gap remains
in effectively handling multimodal content (Mei
et al., 2025; Abootorabi et al., 2025). Current
multimodal RAG systems rely on LLM-based
summarization to convert images into text dur-
ing preprocessing, where a vision-language model
generates textual descriptions of each image, and
only these text summaries are stored in the vector
database. This approach introduces information
loss, as visual context, spatial relationships, and
numerical precision are degraded or omitted dur-
ing text conversion. Additionally , inference-time
solutions such as direct multimodal embedding
retrieval, where images are stored natively in thearXiv:2511.16654v1  [cs.CL]  20 Nov 2025

same vector space as text, remain underexplored
in production RAG workflows.
Recent breakthroughs in multimodal embed-
ding models such as CLIP (Radford et al., 2021)
and Jina v4 (Günther et al., 2025; Jina AI,
2025) offer a promising alternative. These vision-
language models can embed text and images into
a unified semantic vector space, enabling text
queries to retrieve both textual passages and vi-
sual content based on shared meaning. How-
ever, existing multimodal RAG research has not
systematically compared direct image embedding
retrieval with conventional LLM-summary-based
approaches across full end-to-end workflows en-
compassing retrieval accuracy , answer quality ,
and model robustness, particularly for financial
documents (Gong et al., 2025; Gondhalekar et al.,
2025; Setty et al., 2024).
In this paper, we present a comprehensive em-
pirical comparison of multimodal retrieval strate-
gies for RAG systems, evaluating text-based
chunk retrieval (LLM-summary-based) and di-
rect multimodal embedding retrieval. W e eval-
uate both approaches across 6 LLM models and
2 embedding models on a newly created financial
earnings call benchmark comprising 40 question-
answer pairs, each paired with targeted docu-
ments (answer-relevant images and text chunks).
Our evaluation spans both retrieval performance
(Precision@5, Recall@5, mean A verage Precision
(mAP@5), normalized Discounted Cumulative
Gain (nDCG@5)) and end-to-end answer qual-
ity through LLM-as-a-judge pairwise comparisons
across six criteria: correctness, numerical fidelity ,
missing information, unsupported additions, con-
ciseness, and clarity .
Experimental results demonstrate that direct
multimodal embedding retrieval significantly out-
performs text-based approaches, achieving 13%
absolute improvement in mean average precision
and 11% improvement in normalized discounted
cumulative gain. These correspond to relative im-
provements of 32% in mAP@5 and approximately
20% in nDCG@5, indicating substantial ranking
and relevance gains from preserving visual in-
formation in native form. When retrieved con-
text is provided to downstream models, image-
based retrieval leads to more accurate and fac-
tually consistent answers, particularly for larger
models with stronger multimodal reasoning ca-
pabilities. These findings suggest that preserv-
ing visual information in its native form substan-
tially improves both retrieval precision and gen-
erated response quality within multimodal RAGsystems.
2 RELA TED WORKS
2.1 Retrieval-Augmented Generation
Retrieval-Augmented Generation (RAG) has
emerged as an effective approach to improve fac-
tual accuracy and reduce hallucinations in large
language models by grounding generation in re-
trieved external knowledge (Lewis et al., 2020).
Lewis et al. introduced the foundational RAG ar-
chitecture combining a dense retriever with a gen-
erative model for knowledge-intensive question
answering, demonstrating that retrieval strength-
ens output grounding (Lewis et al., 2020). Build-
ing on this foundation, dense passage retrieval
methods such as DPR leverage bi-encoders to
map queries and documents into shared embed-
ding spaces, enabling semantic matching beyond
lexical overlap (Karpukhin et al., 2020). Recent
surveys comprehensively review RAG techniques
and confirm that while text-based RAG improves
factual recall, most systems remain limited to tex-
tual content, leaving significant gaps in handling
structured, spatial, or visual information com-
mon in complex documents (Gao et al., 2024;
Huang et al., 2024a). Advanced RAG methods
further enhance retrieval through query rewriting
(Ma et al., 2023), hypothetical document embed-
dings (Gao et al., 2022), corrective retrieval (Y an
et al., 2024), self-reflective generation (Asai et al.,
2023), and hybrid retrieval strategies combining
semantic and lexical approaches (Sawarkar et al.,
2024). Reranking techniques using LLMs as
re-ranking agents improve retrieval precision by
reordering candidates based on query-document
relevance (Sun et al., 2023). However, these ad-
vances predominantly focus on text-based docu-
ments, motivating extensions to multimodal con-
tent where visual and textual information must
be jointly indexed and retrieved.
2.2 Multimodal Embedding Models
Multimodal embedding models enable unified
representations of text and images in shared se-
mantic spaces, facilitating cross-modal retrieval.
CLIP pioneered vision-language pretraining by
learning transferable visual representations from
natural language supervision, aligning image and
text embeddings through contrastive learning on
large-scale web data (Radford et al., 2021). Re-

cent advances extend multimodal embeddings
to support multilingual and task-specific re-
trieval. Jina Embeddings v3 introduced task-
specific Low-Rank Adaptation (LoRA) for text
embeddings, enabling flexible adaptation across
diverse retrieval scenarios (Sturua et al., 2024).
Jina Embeddings v4 further advances this capa-
bility with universal embeddings supporting mul-
timodal and multilingual retrieval, enabling di-
rect comparison of text queries against image con-
tent in a unified vector space (Günther et al.,
2025). These models eliminate the need for sepa-
rate text and image encoders, providing a princi-
pled approach to multimodal retrieval without in-
termediate text conversion. Despite these archi-
tectural advances, systematic empirical compar-
isons evaluating how direct multimodal embed-
dings perform against conventional text-based
image summarization in end-to-end RAG work-
flows remain limited.
2.3 Document Understanding and
Visual Preprocessing
Several approaches address multimodal document
understanding by converting visual content into
text representations. Donut introduced an OCR-
free document understanding transformer that
learns from visual tokens rather than recognized
text, reducing transcription errors but ultimately
producing text sequences that discard visual lay-
out and spatial relationships (Kim et al., 2021).
Pix2Struct further advanced chart and interface
understanding through screenshot parsing as pre-
training, demonstrating improved visual reason-
ing with language supervision (Lee et al., 2022).
While these methods make visual content search-
able through text, they inherently remove lay-
out, scale, and numeric precision critical for fi-
nancial documents and data-rich visualizations
(Setty et al., 2024). Industry frameworks from
Microsoft Azure emphasize multimodal RAG for
complex document structures, advocating for
Document Intelligence to extract and structure
visual content before indexing (Microsoft Azure
AI, 2024b; Microsoft Azure Architecture Center,
2024). Practitioner guides similarly recommend
multimodal pipelines for production systems (An-
alytics Vidhya Editorial T eam, 2024a; Analytics
Vidhya Editorial T eam, 2024b; SimplAI, 2024).
However, these approaches predominantly rely on
preprocessing images into text summaries, intro-
ducing potential information loss that our work
systematically evaluates.2.4 Multimodal RAG Systems
Recent research has begun integrating text and
visual modalities within RAG frameworks for
document question answering. Comprehensive
surveys on multimodal RAG highlight the grow-
ing interest in systems that handle diverse modal-
ities including text, images, tables, and charts
(Abootorabi et al., 2025; Mei et al., 2025; Mul-
timodal RAG Survey Authors, 2025). MHier-
RAG proposed hierarchical and multi-granularity
reasoning for visual-rich document question an-
swering, indexing both text and image data to
improve document-level retrieval (Gong et al.,
2025). However, its image features derive from
text captions and descriptions rather than na-
tive image embeddings, potentially losing vi-
sual fidelity . MultiFinRAG introduced an op-
timized multimodal RAG framework for finan-
cial question answering where figures and tables
are summarized into structured text before in-
dexing, demonstrating improvements on finan-
cial datasets (Gondhalekar et al., 2025). Sim-
ilarly , work on improving retrieval for financial
documents emphasizes preprocessing strategies
to extract textual representations from complex
layouts (Setty et al., 2024). While these ap-
proaches demonstrate the value of incorporating
visual context, they uniformly depend on LLM-
generated summaries or OCR-extracted text as
intermediaries, raising questions about informa-
tion preservation and retrieval fidelity . Our work
directly addresses this gap by empirically compar-
ing text-based retrieval against direct multimodal
embedding retrieval across retrieval metrics and
downstream answer quality , isolating the impact
of embedding modality choice within a controlled
experimental framework.
2.5 Evaluation Methodologies for
Retrieval Systems
Evaluation of retrieval systems relies on estab-
lished information retrieval metrics to measure
ranking quality and relevance. Normalized Dis-
counted Cumulative Gain (nDCG) has become a
standard metric for evaluating ranked retrieval re-
sults, accounting for both relevance and position
in the ranking (Järvelin and Kekäläinen, 2002).
T raditional lexical retrieval baselines including
TF-IDF (Papineni, 2001) and BM25 (Robertson
and Zaragoza, 2009) provide reference points for
dense retrieval evaluation. F or end-to-end sys-
tem evaluation, recent work introduces LLM-as-

a-judge methodologies where language models as-
sess response quality through pairwise compar-
isons, enabling scalable evaluation beyond exact
string matching (Zheng et al., 2023). Our evalua-
tion framework combines standard retrieval met-
rics (Precision@5, Recall@5, mAP@5, nDCG@5)
with LLM-as-a-judge assessment across correct-
ness, numerical fidelity , completeness, concise-
ness, and clarity to provide comprehensive eval-
uation spanning retrieval accuracy and down-
stream generation quality .
3 METHODS
In this section, we present our comparative eval-
uation framework for multimodal retrieval strate-
gies in RAG systems. Our methodology consists
of three main components: (1) a manually cu-
rated financial earnings benchmark with multi-
modal ground truth annotations (3.1), (2) two
retrieval approaches spanning LLM-summary-
based and direct multimodal embedding strate-
gies (3.2), and (3) a comprehensive evaluation
framework combining retrieval metrics and LLM-
as-a-judge answer quality assessment (3.3).
3.1 Dataset Construction
W e construct a financial earnings benchmark con-
sisting of 40 multimodal question-answer pairs
targeting information from a F ortune 500 com-
pany’s publicly available earnings calls. Unlike
existing financial QA datasets that focus solely
on text, our benchmark explicitly requires inte-
gration of both textual and visual (graph) infor-
mation to answer questions correctly .
3.1.1 Data Collection
Financial documents were sourced from quarterly
earnings calls of a F ortune 500 company , includ-
ing both earnings call transcripts (text docu-
ments) and corresponding investor presentation
slide decks (visual documents). These materials
contain critical financial information presented
across modalities: narrative explanations in tran-
scripts and data visualizations such as revenue
charts, margin breakdowns, and growth metrics
in slide decks. Each document collection repre-
sents a complete earnings event, ensuring that
paired text and visual content share temporal and
topical coherence.3.1.2 Question-Answer Pair Generation
F orty multimodal question-answer pairs were
manually created to reflect realistic financial ana-
lyst queries requiring multi-hop reasoning across
both text and images. Questions were designed
to necessitate retrieval and synthesis of informa-
tion from both earnings transcripts and presen-
tation slides. F or each question, ground truth
answers were manually annotated along with rel-
evant page numbers from both document types,
establishing explicit retrieval targets for evalua-
tion. Each question is paired with their relevant
text chunks and relevant images, corresponding
to specific pages in the transcript and slide deck
respectively . This paired structure enables direct
comparison of text-only and multimodal retrieval
performance on identical information needs.
3.1.3 Document Preprocessing
Earnings call transcripts were segmented into
passages corresponding to logical sections or
speaker turns. Presentation slide decks were con-
verted to individual slide images, with each slide
treated as a distinct retrievable unit. This pre-
processing ensures that both modalities are in-
dexed at comparable granularity , where each re-
trievable unit represents a coherent information
block. Ground truth relevance labels for retrieval
evaluation were assigned based on page numbers,
enabling automated computation of precision, re-
call, and ranking metrics.
3.2 T wo Retrieval Approaches
W e compare two retrieval strategies representing
different approaches to handling multimodal con-
tent in RAG systems. Both approaches utilize
Azure AI Search as the vector database backend
and retrieve the top-5 most relevant documents
for each query .
3.2.1 Approach 1: T ext-only Retrieval
In this approach, visual content from the earn-
ings presentation is converted into text before re-
trieval. Each slide image is passed to a OpenAI
GPT-5 model to produce a textual description
intended to capture the key information present
in the visual, including chart labels, numerical
values, and high-level context. These text de-
scriptions serve as surrogates for the original im-
ages, a common strategy in production RAG sys-

tems (Microsoft Azure AI, 2024b; Analytics Vid-
hya Editorial T eam, 2024a)
Both the earnings call transcript chunks and
the LLM slide descriptions are embedded us-
ing OpenAI text-embedding-ada-002, a widely
adopted dense embedding model for semantic re-
trieval (Lumer et al., 2025c; Lumer et al., 2025a;
Lumer et al., 2025b; Chen et al., 2024; Huang
et al., 2024b; Lumer et al., 2024; Lumer et al.,
2025d). At query time, user queries are also
embedded using the same embedding model and
matched against this text-only representation of
the full document set. Because visual informa-
tion is represented indirectly through generated
text rather than native image embeddings, this
approach may omit spatial structure, layout, or
numeric precision present in the original image.
This configuration reflects the current typical
multimodal RAG practice of converting images
into text before retrieval, and therefore serves as
the comparison point for evaluating the benefits
of direct image embedding in our experiment.
3.2.2 Approach 2: Direct Multimodal
Embedding Retrieval
The direct multimodal embedding approach
leverages Jina Embeddings v4 (Günther et al.,
2025), a unified multimodal embedding model
that maps both text and images into a shared
semantic vector space. Unlike the text-based ap-
proach, images are stored natively in their vi-
sual form without intermediate text conversion.
Both earnings call transcript chunks and slide
deck images are embedded directly using Jina v4
and indexed in the same Azure AI Search vec-
tor database. At query time, text queries are
embedded into the same multimodal space, en-
abling semantic retrieval of both textual passages
and visual content based on shared meaning.
This approach preserves visual information in its
native representation, avoiding information loss
from text conversion while enabling cross-modal
retrieval. Images retrieved by this method are
provided directly to downstream vision-language
models during answer generation, allowing mod-
els to interpret visual content with full fidelity .
3.3 Evaluation F ramework
Our evaluation spans both retrieval performance
and end-to-end answer quality , providing com-
prehensive assessment of how embedding modal-
ity choice impacts multimodal RAG systems.3.3.1 Models Evaluated
W e evaluate six OpenAI language models span-
ning multiple capability tiers: OpenAI GPT-4o,
GPT-4o-mini, GPT-4.1, GPT-4.1-mini, GPT-5,
and GPT-5-mini. These models represent varying
levels of reasoning capability and multimodal un-
derstanding, enabling analysis of how model scale
interacts with retrieval strategy effectiveness. F or
both approaches, the embedding models remains
constant while the downstream LLM varies. This
design isolates the impact of retrieval modality
while controlling for embedding model choice, re-
flecting prior findings that embedding model vari-
ation has negligible impact on retrieval perfor-
mance in text-only settings (Lumer et al., 2025c;
Chen et al., 2024; Huang et al., 2024b; Lumer
et al., 2025a; Lumer et al., 2025b).
3.3.2 Retrieval Metrics
Retrieval performance is measured using four
standard information retrieval metrics computed
over the top-5 retrieved documents. Precision@5
measures the fraction of retrieved documents that
are relevant, while Recall@5 measures the frac-
tion of all relevant documents successfully re-
trieved. Mean A verage Precision (mAP@5) com-
putes the average precision across all queries,
accounting for ranking order. Normalized Dis-
counted Cumulative Gain (nDCG@5) evaluates
ranking quality by assigning higher weight to rel-
evant documents appearing earlier in the ranking
(Järvelin and Kekäläinen, 2002). Ground truth
relevance is determined by page numbers anno-
tated during dataset construction: a retrieved
document is considered relevant if its page num-
ber matches the ground truth page number for
the corresponding question. These metrics pro-
vide a comprehensive view of retrieval accuracy ,
coverage, and ranking quality across the two ap-
proaches.
3.3.3 Answer Quality Assessment
End-to-end answer quality is evaluated us-
ing LLM-as-a-judge methodology (Zheng et al.,
2023), where OpenAI GPT-5 performs pairwise
comparisons between answers generated by the
text-based approach and the direct multimodal
embedding approach. F or each of the 40 ques-
tions, both approaches retrieve relevant context
and generate answers using the same downstream
LLM. OpenAI GPT-5 then evaluates answer pairs
across six binary criteria: Correctness (factual

alignment with ground truth), Numerical Fidelity
(accuracy of numeric values), Missing Informa-
tion (content completeness), No Unsupported
Additions (absence of hallucinations), Concise-
ness (eﬀicient wording), and Clarity (readability).
F or each criterion, the judge assigns a score of 1 to
the preferred answer and 0 to the other, enabling
aggregation across questions and models. This
evaluation design isolates the impact of retrieval
modality on downstream generation quality when
both approaches retrieve multimodal content but
differ in how images are represented during re-
trieval.
3.3.4 Answer Generation Pipeline
Retrieved context from each approach is provided
to downstream language models through a stan-
dard RAG prompt template. The prompt in-
cludes the user question, retrieved text chunks
and image summaries (for text-only approach),
or retrieved text chunks and native images (for
multimodal approach), and instructs the model to
generate an answer grounded in the provided con-
text. F or the multimodal approach, image qual-
ity is handled by the OpenAI API based on high-
est allowed resolution settings, following standard
vision model preprocessing (OpenAI, 2024). This
pipeline ensures that differences in answer quality
arise from retrieval strategy rather than prompt
engineering or generation parameters, providing a
controlled comparison of how embedding modal-
ity impacts end-to-end RAG performance.
4 EXPERIMENTS
4.1 Experimental Settings
W e evaluate the two retrieval approaches on our
financial earnings benchmark consisting of 40
multimodal question-answer pairs, each associ-
ated with relevant text chunks and relevant im-
ages. Retrieval performance is measured using
four standard information retrieval metrics com-
puted over the top-5 retrieved documents: Pre-
cision@5, Recall@5, mAP@5, and nDCG). F or
end-to-end answer quality evaluation, we gener-
ate answers using six OpenAI language models:
GPT-4o, GPT-4o-mini, GPT-4.1, GPT-4.1-mini,
GPT-5, and GPT-5-mini. OpenAI GPT-5 serves
as the judge model for pairwise comparisons, eval-
uating answers across six binary criteria: Cor-
rectness (factual alignment), Numerical Fidelity(number accuracy), Missing Information (content
completeness), No Unsupported Additions (hallu-
cination control), Conciseness (eﬀicient wording),
and Clarity (readability). Each criterion receives
a score of 1 for the preferred answer and 0 for the
other. All experiments use Azure AI Search as
the vector database backend with OpenAI text-
embedding-ada-002 for text-only approaches and
Jina Embeddings v4 for multimodal retrieval.
4.2 Retrieval Performance Results
T able 1 presents the macro-averaged retrieval per-
formance comparing direct multimodal embed-
ding retrieval (IMG) against text-based image re-
trieval (LLM_IMG). Direct multimodal retrieval
significantly outperforms the text-LLM-summary
approach across all metrics. The multimodal ap-
proach achieves mAP@5 of 0.5234 compared to
0.3963 for the text-based approach, represent-
ing an improvement of 0.1271 or 32% relative
gain. Similarly , the multimodal approach obtains
nDCG@5 of 0.6543 compared to 0.5448 for the
text-based approach, showing an improvement of
0.1095 or 20% relative gain. Precision@5 im-
proves from 0.480 to 0.540, an increase of 0.060
or 12.5%, while Recall@5 increases from 0.5362 to
0.5529, an improvement of 0.0167 or 3%. These
results demonstrate that direct image embed-
dings capture relevant documents more effectively
than purely text-based embeddings. The high-
est gains appear in mAP@5 and nDCG@5, indi-
cating that multimodal embeddings not only re-
trieve more relevant documents but also produce
superior ranking quality , placing the most rele-
vant documents higher in the result list. This im-
proved ranking directly benefits downstream an-
swer generation by providing models with better-
ordered context.
Method Precision@5 Recall@5 mAP@5 nDCG@5
IMG 0.540 0.5529 0.5234 0.6543
LLM_IMG 0.480 0.5362 0.3963 0.5448
T able 1: Comparison of macro-averaged retrieval
results for direct multimodal embedding retrieval
(IMG) and text-based image retrieval (LLM_IMG).
Bold indicates best performance. Direct multi-
modal embeddings achieve substantial improvements
in mAP@5 and nDCG@5, demonstrating superior
ranking quality .
4.3 Answer Quality Results
End-to-end answer quality was evaluated through
pairwise comparisons between the multimodal

Figure 1: A veraged pairwise comparison scores across all six LLM models. Y ellow bars represent IMG (direct
multimodal embedding retrieval) and red bars represent LLM_IMG (LLM-summary-based retrieval). Scores
indicate win rate proportions, with IMG consistently outperforming LLM_IMG across all models, particularly
for larger non-mini variants.
and text-based approaches using OpenAI GPT-
5 as the judge model. T able 2 presents the av-
eraged judge scores across all six LLM models,
where scores represent the proportion of times
each approach was preferred. Figure 1 visualizes
these averaged results, showing consistent prefer-
ence for the multimodal approach across all mod-
els.
The multimodal approach achieves an over-
all average win rate of 0.612 compared to 0.388
for the LLM-summary approach when averaged
across all criteria and models. The gap is par-
ticularly pronounced for larger models: GPT-5
shows the strongest preference for the multimodal
approach with an average score of 0.82 compared
to 0.18 for the LLM-summary approach, while
OpenAI GPT-4o achieves 0.60 compared to 0.40
and o1 obtains 0.52 compared to 0.48. In con-
trast, smaller mini models exhibit more balanced
scores, with OpenAI GPT-4o-mini at 0.57 com-
pared to 0.43, o1-mini at 0.70 compared to 0.30,
and o5-mini at 0.50 compared to 0.50, indicat-
ing that these models derive limited benefit from
enhanced multimodal context during answer gen-
eration.
Figure 2 breaks down the pairwise compari-
son results for GPT-5 across the six evaluation
criteria. The multimodal approach demonstrates
substantial advantages in Correctness (0.70 com-
pared to 0.30), Numerical Fidelity (0.80 com-
pared to 0.20), and No Unsupported Additions
(0.90 compared to 0.10), indicating that direct
multimodal retrieval produces more factually ac-
curate answers with fewer hallucinations.
Missing Information scores favor the multi-
modal approach (0.60 compared to 0.40), sug-T able 2: A veraged pairwise comparison scores across
all six LLM models. Scores represent the propor-
tion of times each approach was preferred by OpenAI
GPT-5 across six evaluation criteria. Higher scores
indicate stronger preference.
Model IMG LLM_IMG
OpenAI GPT-4o 0.60 0.40
OpenAI GPT-4o-mini 0.57 0.43
OpenAI GPT-4.1 0.52 0.48
OpenAI GPT-4.1-mini 0.70 0.30
OpenAI GPT-5 0.82 0.18
OpenAI GPT-5-mini 0.50 0.50
A verage 0.612 0.388
gesting more complete answers when visual con-
text is preserved. Conciseness shows a near tie
(0.90 compared to 0.10), while Clarity achieves
perfect preference for the multimodal approach
(1.00 compared to 0.00). The most striking result
appears in hallucination control, where the multi-
modal approach prevents unsupported additions
90% of the time, substantially reducing the in-
formation loss and fabrication that occurs when
images are converted to text summaries during
preprocessing.
4.4 Discussion
The experimental results demonstrate that direct
multimodal embedding retrieval substantially
outperforms text-based approaches across both
retrieval metrics and downstream answer quality .
The 32% relative improvement in mAP@5 and
20% improvement in nDCG@5 indicate that pre-
serving visual information in its native form en-
ables more accurate semantic matching between
queries and multimodal documents. This im-

Figure 2: Breakdown of pairwise comparison scores for GPT-5 across six evaluation criteria. Y ellow bars
represent IMG and red bars represent LLM_IMG. IMG shows substantial advantages in Correctness, Numerical
Fidelity , and No Unsupported Additions (hallucination control), demonstrating that native image embeddings
preserve critical information lost during text conversion.
proved retrieval accuracy translates directly to
higher-quality generated answers, particularly for
larger models with stronger multimodal reasoning
capabilities. The pronounced gap in hallucina-
tion control, where the multimodal approach re-
duces unsupported additions by 80 absolute per-
centage points for GPT-5, suggests that LLM
summarization introduces not only information
loss but also fabricated details that propagate to
downstream generation. The diminishing returns
observed for mini models indicate that effective
utilization of multimodal context requires suﬀi-
cient model capacity for cross-modal reasoning.
These findings have practical implications for pro-
duction RAG systems handling multimodal docu-
ments: while LLM summarization offers a conve-
nient preprocessing strategy compatible with ex-
isting text-only infrastructure, it fundamentally
limits retrieval and generation quality compared
to native multimodal embeddings. Organizations
should prioritize multimodal embedding models
when deploying RAG systems for document types
where visual information carries critical seman-
tics, particularly in domains such as financial re-
porting where charts, tables, and numerical visu-
alizations convey information diﬀicult to capture
through text alone.
5 LIMIT A TIONS
While direct multimodal embedding retrieval ad-
vances multimodal RAG systems, a key limita-
tion concerns preprocessing complexity for mul-
timodal embeddings. Unlike text-based ap-
proaches that convert images to text through asingle LLM call, multimodal embedding pipelines
require explicit image detection, extraction, and
format conversion steps. Documents must be
parsed to identify charts, tables, and images, with
each visual element saved as a separate file be-
fore embedding. This preprocessing burden in-
creases for diverse document types, as Power-
Point presentations where entire slides serve as re-
trievable units differ fundamentally from PDF re-
ports where individual figures must be extracted.
Automated preprocessing tools such as Docling
(Docling Project, 2024), Azure Document Intel-
ligence (Microsoft Azure AI, 2024a), and Un-
structured.io (Unstructured IO T eam, 2024) pro-
vide partial solutions, but distinguishing between
tables and images remains challenging. F uture
work should develop robust document parsing
pipelines that automatically segment and classify
visual elements across document formats, reduc-
ing the operational overhead of deploying multi-
modal RAG systems in production environments.
6 CONCLUSION
Multimodal RAG systems must effectively re-
trieve and reason over both textual and visual
content from documents containing charts, ta-
bles, and images. W e present a comparative
evaluation of two retrieval approaches for multi-
modal RAG systems, including text-based chunk
retrieval where images are converted to text dur-
ing preprocessing, and direct multimodal embed-
ding retrieval where images are stored natively
in vector space. W e evaluated both approaches
across six OpenAI language models on a newly

created financial earnings benchmark comprising
40 question-answer pairs requiring integration of
textual and visual information. Experimental
results demonstrate that direct multimodal em-
bedding retrieval substantially outperforms text-
based approaches, achieving a 32% relative im-
provement in mean average precision and an over-
all win rate of 0.612 compared to 0.388 in LLM-
as-a-judge pairwise comparisons. These findings
provide empirical evidence that preserving visual
information in native form rather than convert-
ing to text summaries enables more accurate re-
trieval and downstream generation in multimodal
RAG systems. F uture work should extend eval-
uation to diverse domains including medical, le-
gal, and scientific documents where visual content
serves different purposes, while developing auto-
mated pipelines to reduce operational overhead.
As multimodal embedding models mature and
vision-language models strengthen their cross-
modal reasoning capabilities, the performance ad-
vantages of direct multimodal retrieval are likely
to widen further.
REFERENCES
Abootorabi, M. M., Zobeiri, A., Dehghani, M., et al.
(2025). Ask in Any Modality: A Comprehen-
sive Survey on Multimodal Retrieval-Augmented
Generation. In Findings of the Association for
Computational Linguistics: ACL 2025.
Analytics Vidhya Editorial T eam (2024a). A Com-
prehensive Guide to Building Multimodal RAG
Systems.
Analytics Vidhya Editorial T eam (2024b). RAG
with Multimodality and Azure Document Intel-
ligence.
Asai, A., W u, Z., W ang, Y., Sil, A., and Hajishirzi,
H. (2023). Self-RAG: Learning to Retrieve,
Generate, and Critique through Self-Reflection.
Preprint, arXiv:2310.11511.
Chen, Y., Y oon, J., Sachan, D. S., W ang, Q., Cohen-
Addad, V., Bateni, M., Lee, C.-Y., and Pfister,
T. (2024). Re-invoke: T ool invocation rewriting
for zero-shot tool retrieval.
Docling Project (2024). Docling – open source docu-
ment processing for gen ai.
Gao, L., Ma, X., Lin, J., and Callan, J. (2022).
Precise Zero-Shot Dense Retrieval without Rel-
evance Labels. Preprint, arXiv:2212.10496.
Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi,
Y., Dai, Y., Sun, J., W ang, M., and W ang,
H. (2024). Retrieval-Augmented Generation for
Large Language Models: A Survey . Preprint,
arXiv:2312.10997.Gondhalekar, C., Patel, U., and Y eh, F.-C.
(2025). MultiFinRAG: An Optimized Multi-
modal Retrieval-Augmented Generation (RAG)
F ramework for Financial Question Answering.
Preprint, arXiv:2506.20821.
Gong, Z., Mai, C., and Huang, Y. (2025).
MHier-RAG: Multi-Modal RAG for Visual-Rich
Document Question-Answering via Hierarchi-
cal and Multi-Granularity Reasoning. Preprint,
arXiv:2508.00579.
Günther, M., Sturua, S., Akram, M. K., et al. (2025).
jina-embeddings-v4: Universal Embeddings for
Multimodal Multilingual Retrieval. Preprint,
arXiv:2506.18902.
Huang, D., Zhu, P ., Liu, W., et al. (2024a). A
Survey on Retrieval-Augmented T ext Genera-
tion for Large Language Models. Preprint,
arXiv:2404.10981.
Huang, T., Jung, D., and Chen, M. (2024b). Planning
and editing what you retrieve for enhanced tool
learning.
Jina AI (2025). Jina Embeddings v4: Universal Em-
beddings for Multimodal Multilingual Retrieval.
Järvelin, K. and Kekäläinen, J. (2002). Cumu-
lated Gain-Based Evaluation of IR T echniques.
ACM T ransactions on Information Systems,
20(4):422–446.
Karpukhin, V., Oguz, B., Min, S., Lewis, P ., W u, L.,
Edunov, S., Chen, D., and Yih, W.-t. (2020).
Dense Passage Retrieval for Open-Domain Ques-
tion Answering. In Proceedings of the 2020 Con-
ference on Empirical Methods in Natural Lan-
guage Processing. Preprint, arXiv:2004.04906.
Kim, G., Hong, T., Yim, M., et al. (2021). OCR-
free Document Understanding T ransformer.
Preprint, arXiv:2111.15664.
Lee, K., Joshi, M., T urc, I., et al. (2022).
Pix2Struct: Screenshot Parsing as Pretraining
for Visual Language Understanding. Preprint,
arXiv:2210.03347.
Lewis, P ., Perez, E., Piktus, A., et al. (2020).
Retrieval-augmented generation for knowledge-
intensive NLP tasks. In Advances in Neu-
ral Information Processing Systems. Preprint,
arXiv:2005.11401.
Lumer, E., Basavaraju, P . H., Mason, M., Burke,
J. A., and Subbiah, V. K. (2025a). Graph rag-
tool fusion.
Lumer, E., Gulati, A., Subbiah, V. K., Basavaraju,
P . H., and Burke, J. A. (2025b). Memtool: Opti-
mizing short-term memory management for dy-
namic tool calling in llm agent multi-turn con-
versations.
Lumer, E., Gulati, A., Subbiah, V. K., Basavaraju,
P . H., and Burke, J. A. (2025c). Scalemcp:
Dynamic and auto-synchronizing model context
protocol tools for llm agents.
Lumer, E., Nizar, F., Gulati, A., Basavaraju, P . H.,
and Subbiah, V. K. (2025d). T ool-to-agent

retrieval: Bridging tools and agents for scal-
able llm multi-agent systems. arXiv preprint
arXiv:2511.01854.
Lumer, E., Subbiah, V. K., Burke, J. A., Basavaraju,
P . H., and Huber, A. (2024). T oolshed: Scale
tool-equipped agents with advanced rag-tool fu-
sion and tool knowledge bases.
Ma, X., Gong, Y., He, P ., Zhao, H., and Duan,
N. (2023). Query Rewriting for Retrieval-
Augmented Large Language Models. Preprint,
arXiv:2305.14283.
Mei, L., Mo, S., Y ang, Z., and Chen, C. (2025). A Sur-
vey of Multimodal Retrieval-Augmented Gener-
ation. Preprint, arXiv:2504.08748.
Microsoft Azure AI (2024a). Azure ai document in-
telligence.
Microsoft Azure AI (2024b). Build Intelligent
RAG for Multimodality and Complex Document
Structure.
Microsoft Azure Architecture Center (2024). Com-
plex Data Extraction using Document Intelli-
gence and RAG.
Multimodal RAG Survey Authors (2025). Multi-
modal RAG Survey Project Page.
OpenAI (2024). Images and vision: Specify image
input detail level.
Papineni, K. (2001). Why Inverse Document F re-
quency? In Second Meeting of the North Amer-
ican Chapter of the Association for Computa-
tional Linguistics.
Radford, A., Kim, J. W., Hallacy , C., et al. (2021).
Learning T ransferable Visual Models F rom Nat-
ural Language Supervision. In Proceedings of
the 38th International Conference on Machine
Learning. Preprint, arXiv:2103.00020.
Robertson, S. and Zaragoza, H. (2009). The Prob-
abilistic Relevance F ramework: BM25 and Be-
yond. F oundations and T rends in Information
Retrieval, 3(4):333–389.
Sawarkar, K., Mangal, A., and Solanki, S. R. (2024).
Blended RAG: Improving RAG (Retriever-
Augmented Generation) Accuracy with Seman-
tic Search and Hybrid Query-Based Retrievers.
Preprint, arXiv:2404.07220.
Setty , S., Thakkar, H., Lee, A., Chung, E., and Vidra,
N. (2024). Improving Retrieval for RAG based
Question Answering Models on Financial Docu-
ments. Preprint, arXiv:2404.07221.
SimplAI (2024). Building a Multi-modal Production
RAG.
Sturua, S., Mohr, I., Akram, M. K., et al. (2024). jina-
embeddings-v3: Multilingual Embeddings With
T ask LoRA. Preprint, arXiv:2409.10173.
Sun, W., Y an, L., Ma, X., et al. (2023). Is Chat-
GPT Good at Search? Investigating Large Lan-
guage Models as Re-Ranking Agents. Preprint,
arXiv:2304.09542.
Unstructured IO T eam (2024). Unstructured – trans-
forming and ingesting diverse documents for
llms.Y an, S.-Q., Gu, J.-C., Zhu, Y., and Ling, Z.-H.
(2024). Corrective Retrieval Augmented Gen-
eration. Preprint, arXiv:2401.15884.
Zheng, L., Chiang, W.-L., Sheng, Y., et al. (2023).
Judging LLM-as-a-Judge with MT-Bench and
Chatbot Arena. Preprint, arXiv:2306.05685.