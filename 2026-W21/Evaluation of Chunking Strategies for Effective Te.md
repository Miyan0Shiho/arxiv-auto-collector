# Evaluation of Chunking Strategies for Effective Text Embedding in Low-Resource Language on Agricultural Documents

**Authors**: Sovandara Chhoun, Pichdara Po, Sereiwathna Ros, Wan-Sup Cho, Saksonita Khoeurn

**Published**: 2026-05-21 09:06:13

**PDF URL**: [https://arxiv.org/pdf/2605.22203v1](https://arxiv.org/pdf/2605.22203v1)

## Abstract
In this study, we compare the performance of four text chunking approaches: Recursive, Khmer-Aware, Sentence-Based, and LLM-Based within a Retrieval-Augmented Generation (RAG) framework applied to Khmer agricultural documents. The document chunks are encoded using the BGE-M3 multilingual embedding model and retrieved using the FAISS library. Performance is evaluated using four metrics: Average Retrieval Score (L2 distance), Answer Relevance, Khmer Coverage, and Khmer Intersection over Union, all measured against ground-truth question-answer pairs. For evaluation, we perform 5-fold cross-validation over 18 question-answer pairs. We observe the best performance for the character-based Recursive chunking method with a chunk size of 300 characters, achieving the lowest L2 distance (0.4295 +- 0.0461), highest Answer Relevance (0.8663 +- 0.0199), and highest Khmer IoU (0.6441 +- 0.0347). A paired t-test shows a statistically significant improvement over the Sentence-Based chunking method in L2 distance (p = 0.0121). These results highlight the importance of segmentation granularity and structural preservation for optimizing dense retrieval in morphologically complex, low-resource languages such as Khmer.

## Full Text


<!-- PDF content starts -->

Evaluation of Chunking Strategies for Effective T ext Embedding in
Low-Resource Language on Agricultural Documents
Sovandara Chhoun1, Pichdara Po2, Sereiwathna Ros2, W an-Sup Cho3, and Saksonita Khoeurn3
1Department of Big Data, Chungbuk National University, Cheongju-si, South Korea,
2Department of Computer Science, Chungbuk National University, Cheongju-si, South Korea,
3BigDataLabs Co., Ltd. Department of Management Information Systems, Chungbuk National University,
South Korea,
Abstract
In this study , we compare the performance of four text chunking approaches — Recursive,
Khmer-A ware, Sentence-Based, and LLM-Based — within a Retrieval-Augmented Generation
(RAG) framework applied to Khmer agricultural documents. The document chunks are en-
coded using the BGE-M3 multilingual embedding model and retrieved using the F AISS library .
Performance is evaluated using four metrics: A verage Retrieval Score (L2 distance), Answer Rele-
vance, Khmer Coverage, and Khmer Intersection over Union, all measured against ground-truth
question–answer pairs. F or evaluation, we perform 5-fold cross-validation over 18 question–
answer pairs. W e observe the best performance for the character-based Recursive chunking
method with a chunk size of 300 characters, achieving the lowest L2 distance (0.4295 ± 0.0461),
highest Answer Relevance (0.8663 ± 0.0199), and highest Khmer IoU (0.6441 ± 0.0347). A
paired t-test shows a statistically significant improvement over the Sentence-Based chunking
method in L2 distance (p = 0.0121). These results highlight the importance of segmentation
granularity and structural preservation for optimizing dense retrieval in morphologically com-
plex, low-resource languages such as Khmer.
1 Introduction
Recent breakthroughs in embedding-based natural language processing have revolutionized the ap-
proach to handling textual data, including indexing, retrieval, and semantic analysis within modern
intelligent systems [ 1]. The dense vector representations learned via deep neural embedding archi-
tectures facilitate semantic similarity search, document retrieval, and question-answering beyond
simple word-level matching [ 2]. The standard pipeline involves document preprocessing, chunk-
ing, embedding generation, and vector-based indexing using approximate nearest neighbor search
libraries such as F AISS [ 3]. Although significant research has improved embedding architectures
and retrieval paradigms, there has been relatively little work on improving the document chunk-
ing step, which is essential for dividing long documents into semantically coherent chunks before
embedding [ 4].
Chunking is arguably the most significant step in document preparation since it directly in-
fluences the granularity and semantic content of embedded document representations [ 5,6]. Poor
document segmentation can result in embeddings being too generic, combining unrelated concepts,
or too fine-grained, disrupting semantic relationships between concepts and procedures. Hence, doc-
ument chunking is an understudied yet vital component in embedding-based document retrieval
and semantic search systems [ 1,7]. Despite its importance, there has been little work on evaluating
document chunking strategies, especially in languages with limited resources [ 8].
1arXiv:2605.22203v1  [cs.CL]  21 May 2026

These issues are particularly exacerbated when dealing with low-resource languages, where lin-
guistic properties make preprocessing and modeling more diﬀicult [ 8,9]. Khmer is an interesting
case study as it is an oﬀicial language of Cambodia with no word boundaries, complex orthography ,
and limited downstream natural language processing (NLP) resources [ 10]. T raditional chunking
methods, such as fixed window-based or sentence-based chunking, have been developed with refer-
ence to high-resource languages with well-established punctuation and grammatical rules [ 11,12].
Applying such methods to Khmer may lead to a loss of semantic continuity or even damage pro-
cedural content. This is particularly critical when dealing with agricultural documents that often
require multi-step procedural content, specialized vocabulary , and hierarchical explanatory con-
tent [ 8,9]. Maintaining semantic continuity is critical to ensure accurate retrieval and question
answering [ 1,7].
Inspired by this problem, this study aims to investigate how chunking strategy selection affects
embedding-based retrieval performance on Khmer agricultural documents. More precisely , this
study aims to answer the following research questions:
• 1) How do different chunking strategies affect retrieval accuracy in Khmer agricultural docu-
ments?
• 2) What chunk size optimizes semantic coherence and retrieval precision within a recursive
segmentation framework?
• 3) What are the computational trade-offs between rule-based and LLM-based chunking ap-
proaches?
T o conduct this investigation, we carry out an extensive evaluation of four different chunking
methods: recursive chunking, Khmer-aware chunking, sentence-based chunking, and large language
model chunking. The methods we employ are based on various segmentation philosophies, which
can be broadly categorized as rule-based segmentation and semantic segmentation [ 12].
F or embedding and retrieval, we utilize the multilingual model BGE-M3 [ 2], whose vector
representations are indexed using F AISS [ 3]. The evaluation metrics are based on multiple criteria,
including A verage Retrieval Score (L2 Distance), Answer Relevance (Cosine Similarity), Khmer
Coverage, and Khmer Intersection-over-Union (Khmer IoU) [ 10]. W e adopt a 5-fold cross-validation
approach to ensure statistical robustness in our evaluation.
2 Related W orks
2.1 T ext Chunking in Embedding-Based Retrieval
T ext chunking is a key component in embedding-based information retrieval systems, as it deter-
mines the semantic granularity of the content being retrieved. In text-based information retrieval
systems that incorporate a retrieval-augmented generation (RAG) mechanism, text segmentation
is a key component that influences the accuracy of document embeddings and, in turn, information
retrieval accuracy [ 1,7]. T ext chunking is a technique that splits long documents into smaller se-
mantic units that can easily be embedded and processed in large language models [ 13]. As a result,
the text segmentation technique significantly determines the accuracy of document embeddings in
representing contextual information in the source text.
Previous techniques in text-based information retrieval systems relied on fixed-size text chunk-
ing, in which a text document is segmented into smaller chunks of equal character or token size,
usually with overlapping segments to preserve contextual information. Although this technique is
2

eﬀicient in text processing and information retrieval, it is prone to information fragmentation, in
which semantically significant information is spread across multiple chunks or irrelevant informa-
tion is embedded in a chunk [ 5]. Large chunk sizes may result in a lack of semantic specificity ,
while small chunk sizes may fragment contextual information in a text document. Recent studies
show that recursive text segmentation techniques can significantly improve information retrieval
accuracy compared to fixed-size text segmentation techniques [ 4,6]. However, there is a lack of
systematic evaluations of various text segmentation techniques [ 11,12].
2.2 Sentence-Based and Language-A ware Chunking
Sentence-based chunking methods are designed to maintain semantic integrity by dividing the text
according to sentence boundaries. More advanced semantic chunking methods involve computing
embeddings for adjacent groups of sentences and identifying topic shifts based on semantic distance
thresholds [ 4]. This method is likely to yield optimal results for resource-rich languages for which
sentence tokenizers are robust and punctuation is standardized.
Sentence-based chunking methods, however, are likely to face challenges in low-resource lan-
guages, where punctuation may be irregular and robust NLP tools may not be available. Language-
aware preprocessing methods have been developed to overcome these challenges. F or example, in
the case of the Vietnamese language, NLP methods such as V nCoreNLP use conditional random
fields (CRF) and neural sequence models to improve phrase segmentation [ 14]. Similar methods
have been developed for the Khmer language, including CRF-based word segmentation [ 8], seg-
mentation methods based on BiLSTM networks [ 9], and surveys of word segmentation methods
for the Khmer language [ 10]. Despite these studies highlighting the importance of language-aware
segmentation, their effect on the performance of embedding-based retrieval systems has not been
extensively explored [ 15,16].
2.3 LLM-Based Document Segmentation
Recently , large language models (LLMs) have been utilized for document segmentation using
prompt-based methods. Contextual retrieval-based methods use the entire document to prompt
an LLM to generate enriched descriptions for document segments before embedding, as shown in
previous works [ 17]. On the other hand, LLM-based semantic chunking methods prompt an LLM
to split documents based on semantic boundaries inferred by the model.
While these methods provide flexibility for different languages and domains, they come at the
cost of increased computational latency . Contextual retrieval-based methods, such as the combi-
nation of dense embeddings and sparse retrieval, have been shown to yield significant reductions
in retrieval errors [ 17]. However, the assessment of LLM-based document chunking strategies has
been inadequate for low-resource languages such as Khmer [ 15,16].
2.4 Embedding Models for Multilingual Retrieval
Dense models of embeddings allow text segments to be mapped to a continuous vector space,
facilitating semantic similarity measurement using cosine similarity or L2 distance. Multilingual
embeddings are an extension of this concept that aims to perform this operation even in low-
resource languages. Embeddings such as mBER T, LaBSE, and E5 have been shown to perform
robust cross-lingual retrieval in a multilingual environment [ 2].
BGE-M3 is a recent advancement in the family of multilingual embedding models that supports
over 100 languages and incorporates dense, sparse, and multi-vector-based retrieval models [ 2]. Its
ability to process long text and simultaneously perform robust cross-lingual retrieval makes it a
3

1055 Chunks (Balanced)Khmer Agricultural
Documents
Hierachical Structure,
Low_Resource LanguageRecursive (300-Char)
 Structural
      PreservationKhmer_A ware
  Linguistic 
 Heuristics     LLM-based
          Content 
          Consolidation
Sentence-based
Sentence
Boundary
       Segmentation318 Chunks (Fewer , Larger)
455 Chunks (Moderate)
1206 Chunks (Fragmented)
 Avg Retr (L2) ↓
Khmer Cov .↑
Ans. Rel. ↑
Khmer IoU ↑ Output Chunks
Processed
ByEvaluate Using
Key Metrics   
     
Unified Evaluation Framework
                      BGE-M3
           Embedding Model 
              Retrieval-Augmented
  Generation & 
              Embedding Similarity
                5-Fold Cross-V alidation
    
                   
                    
                      
             TextFigure 1: End-to-end pipeline chunking evaluation.
suitable candidate for Khmer text retrieval tasks. BGE-M3 demonstrated robust performance in a
multilingual environment in the MIRACL benchmark, where it handled various diverse languages,
including low-resource ones [ 18]. Nevertheless, it is important to note that even with a robust
embedding model, document segmentation strategies are of equal importance in determining the
effectiveness of a retrieval model [ 4,6].
2.5 Low-Resource Language Processing
Khmer has been identified as a low-resource language, considering the lack of annotated corpora and
linguistic tools available for the language [ 8,9]. Unlike Latin-based scripts, there is no whitespace
specification for word boundaries, and the Khmer script follows an abugida system, consisting of
complex combinations of consonants and vowels [ 8]. Although there has been significant progress
in the development of word segmentation and part-of-speech tagging for the Khmer language,
achieving high accuracy results [ 10], there has been little research carried out on document-level
chunking for retrieval-based applications.
Current trends in Southeast Asian language modeling and code-mixed text processing have been
identified in recent research works [ 15,19]. This indicates the increased emphasis being placed on
low-resource languages. However, the chunking approaches for embedding-based retrieval in Khmer
agricultural documents have not been extensively explored [ 16].
3 Methodology
3.1 Architecture Overview
The proposed retrieval pipeline consists of six interconnected stages: data preparation, chunking,
embedding, indexing, retrieval, and evaluation [ 1,13]. Initially , documents are preprocessed by con-
verting PDF s to text and then normalizing the text into Markdown format. The text is subsequently
segmented using different chunking strategies, embedded with the BGE-M3 model [ 2], and indexed
using F AISS [ 3] for eﬀicient nearest neighbor retrieval. Finally , retrieval performance is quantita-
tively evaluated using ground-truth question–answer pairs under a 5-fold cross-validation [ 5].
4

3.2 Chunking Strategies
F our chunking strategies are considered:
• Khmer-A ware Recursive Splitting: A rule-based approach informed by Khmer script
characteristics [ 8,10].
• Recursive Splitting: Rule-based segmentation using fixed character (e.g., 300 characters)
to preserve hierarchical context [ 11].
• Sentence-Based Chunking: Segmentation along sentence boundaries, designed to preserve
semantic coherence at the sentence level [ 5].
• LLM-Based Semantic Chunking: Large language model (LLM) guided segmentation
using prompt-based semantic inference [ 17].
3.2.1 Embedding Generation
BGE-M3 Model Selection and Configuration: Each chunk is embedded into a dense vector
using BGE-M3 [ 2], a multilingual model effective for low-resource languages including Khmer. BGE-
M3 produces 1,024-dimensional vectors for input sequences up to 8,192 tokens.
3.2.2 V ector Indexing and Retrieval
F AISS Indexing and Similarity Search: Embeddings are indexed with F AISS [ 3], enabling fast
nearest neighbor search. L2 distance is used to identify nearest chunks, while cosine similarity is
applied to evaluate semantic relevance between retrieved chunks and queries [ 5].
3.2.3 Ground-T ruth Construction
Question–Answer Pair Development: Domain experts annotated the agricultural document
collection to produce 18 question–answer pairs, which serve as ground truth for retrieval evaluation
[8,10].
3.3 Chunking Strategy Examples
The following examples illustrate how different chunking strategies process Khmer agricultural text.
4 Evaluation Metrics
T o present an overview of the proposed chunking methods—Recursive, Khmer-A ware Recursive,
Sentence-Based, and LLM-Based four complementary evaluation metrics are adopted from recent
studies on text chunking and retrieval [ 6,17]. Evaluation is conducted using 5-fold cross-validation
on 18 question–answer pairs, with results reported as mean ± standard deviation.
4.1 Evaluation Setup
T o ground the metric definitions, consider the following representative example:
• Query ( Q):េតី្រតូវេ្របីជីអ្វីខ្លះស្រមាប ់េដីមសា្វយចន្ទី?(What fertilizers should be used for cashew trees?)
5

T able 1: Comparison of Chunking Strategies with Examples
Method Description Example
Khmer-A ware Splits text hierarchically by
paragraph ( \n\n ) and line
breaks ( \n ). Detects Khmer
sentence markers such as ( ។
៕) to preserve boundaries.Input: ដាំសា្វយចន្ទី។ ែថរក្សាស ួន៕ (Plant cashew trees. Maintain the
garden.)
Chunks: [ ដាំសា្វយចន្ទី។(Plant cashew trees.)]
[ែថរក្សាស ួន៕ (Maintain the garden.)]
Recursive
StandardHierarchical rule-based split-
ting using separators such as
\n , punctuation, and recur-
sive chunking with overlap.Input: Hello world. How are you?
Chunks: [Hello world.] [How are you?]
Sentence-
BasedDetects sentence boundaries
using punctuation (e.g., ។)
and groups sentences into
chunks of fixed size ( N= 5 )
with overlap.Sentences: S1–S8
Chunk 1: [S1–S5]; Chunk 2: [S5–S8]
LLM-Based Uses a large language model
to split text based on seman-
tic coherence rather than
fixed rules.Input: Plant cashew trees. W ater regularly . Protect crops. Spray
pesticide.
Chunks: [Plant cashew trees; W ater regularly]
[Protect crops; Spray pesticide]
• Ground-truth Answer ( A):េគ្រតូវេ្របីជីអ៊ុយេរ៉និងជីប៉ូតាស្យូមស្រមាប ់ដាំសា្វយចន្ទី។(One should use Urea
and Potassium fertilizers for planting cashews.)
• Retrieved Chunks ( R):
–Chunk 1: ការេ្របី្របាស់ជីអ៊ុយេរ៉ផ្តល់ផលល ្អដល់សា្វយចន្ទី។(The use of Urea fertilizer gives good results to
cashew nuts.)
–Chunk 2: ជីប៉ូតាស្យូមជួយឱ្យ្រគាប ់សា្វយចន្ទីមានទម ្ងន់ធ្ងន់។(Potassium fertilizer helps cashew nuts to be
heavy .)
4.2 Semantic Alignment Metrics
4.2.1 A verage Retrieval Score (L2 Distance)
This metric measures the average L2 distance between the query embedding and the top- kretrieved
chunk embeddings in vector space [ 3,5]. Lower values indicate stronger semantic proximity .
L2avg=1
kk∑
i=1∥eq−eci∥2 (1)
Interpretation: Smaller distances imply that retrieved chunks are closer to the query in embed-
ding space.
6

4.2.2 Answer Relevance
Answer Relevance quantifies the semantic alignment between the aggregated top- kretrieved chunks
(R) and the ground-truth answer ( A) using cosine similarity [ 5]:
AR= cos(eR, eA) (2)
where eRrepresents the mean embedding of the retrieved chunks.
Higher values indicate stronger semantic coverage of the expected answer.
4.3 Khmer Linguistic Fidelity
4.3.1 Khmer Coverage
T o maintain script integrity and reduce unnecessary noise such as metadata, Latin script, or encod-
ing artifacts, we measure the proportion of Khmer Unicode characters in the retrieved text. This
metric is motivated by the challenges of segmentation in Khmer script, as discussed in previous
studies on Khmer word segmentation and low-resource language processing [ 10].
Coverage =T otal Khmer Characters in R
T otal Non-whitespace Characters in R(3)
V alues closer to 1.0 indicate higher linguistic purity .
4.3.2 Khmer Intersection-over-Union (KIoU)
Standard word-level IoU is unreliable for Khmer due to the absence of explicit word boundaries and
the presence of zero-width characters. T o address this, we adopt a character-level IoU specifically
adapted for Khmer. Unique Khmer character sets are extracted from both the retrieved text ( R)
and the ground-truth answer ( A) to compute the intersection-over-union, providing a metric that
is robust to segmentation inconsistencies in low-resource scripts [ 8].
KIoU =|Rkh∩Akh|
|Rkh∪Akh|(4)
This metric measures textual overlap while remaining robust to segmentation inconsistencies
and script-specific characteristics.
5 Evaluation Results
In this section, we evaluate four distinct chunking strategies for a collection of Khmer agricul-
tural documents: an LLM-based method, a Khmer-A ware Recursive method, a Recursive method
with a 300-character limit, and a Sentence-based method. The results are presented as mean ±
standard deviation across 5-fold cross-validation. The evaluation framework follows prior work on
retrieval-augmented generation and embedding-based similarity measures [ 1]. Chunking strategies
are inspired by previous studies in supervised text segmentation and long-context embedding [ 4,11].
5.1 Chunk Distribution Analysis
Chunk distribution differs across methods, reflecting each method’s segmentation philosophy . The
Sentence-based method produced the most chunks (1,206), resulting in fragmented procedural steps,
7

T able 2: Comparison of Chunking Strategies on Khmer Agricultural Documents
Method Chunks QTY A vg Retr. (L2)↓ Khmer Cov.↑ Ans. Rel. (Cos)↑ Khmer IoU↑
LLM 318 0.4948 ± 0.0347 0.8737 ± 0.0184 0.8355 ± 0.0173 0.5766 ± 0.0141
Khmer-A ware 455 0.7727 ± 0.0223 0.8425 ± 0.0389 0.8530 ± 0.0249 0.6053 ± 0.0269
Recursive 1055 0.4295 ± 0.0461 0.8860 ± 0.0231 0.8663 ± 0.0199 0.6441 ± 0.0347
Sentence-based 1206 0.4721 ± 0.0470 0.8155 ± 0.0430 0.8556 ± 0.0230 0.6384 ± 0.0203
Note: F or A vg Retr. (L2) ↓, lower values are better; for Khmer Cov. ↑, Ans. Rel. (Cos) ↑, and Khmer IoU ↑, higher
values are better.
consistent with prior observations [ 11]. In contrast, the LLM-based method grouped content into
fewer chunks (318), consolidating information but reducing granularity .
The Recursive method with a 300-character window produced 1,055 chunks, offering a balance
between preserving hierarchical structure and capturing fine-grained semantic details. Compared
to larger recursive windows (e.g., 500 or 800 characters), the 300-character improves embedding
discrimination and retrieval performance [ 4].
5.2 Retrieval Performance Comparison
T able 2shows that the Recursive method performs best overall. It has the lowest L2 distance
(0.4295±0.0461 , where lower is better), and the highest Answer Relevance ( 0.8663±0.0199 ) and
Khmer IoU ( 0.6441±0.0347 , where higher is better) [ 1,7].
A paired t-test confirms that Recursive significantly outperforms Sentence-based chunking in
L2 distance ( p= 0.0121 ), indicating a statistically meaningful improvement.
Although Sentence-based chunking achieves competitive Khmer IoU ( 0.6384±0.0203 ), its re-
trieval distance ( 0.4721±0.0470 ) remains inferior to Recursive. The Khmer-A ware method shows
high Khmer IoU but substantially worse L2 distance ( 0.7727±0.0223 ), suggesting that linguistic
heuristics alone are insuﬀicient without structural segmentation awareness.
5.3 Qualitative Analysis
Qualitative analysis shows that Recursive (300-character) is effective in preserving procedural
integrity in Khmer agricultural guides. By retaining hierarchical structures such as headings,
paragraphs, and lists, prerequisite conditions and corresponding actions remain semantically com-
plete [ 1,11].
Sentence-based chunking often separates conditions from actions. Although it can match key
terms, it sometimes lacks context, reducing coherence [ 5,8].
The LLM-based method produced moderate results but occasionally over-generalized domain-
specific terminology , reducing alignment with queries for procedural instructions [ 2].
Overall, qualitative results indicate that the Recursive (300-character) method best balances
contextual completeness and semantic specificity in Khmer agricultural documents [ 10,16].
6 Discussion
6.1 Interpretation of Key Findings
The results show that structural preservation has a more decisive effect than sentence-level seg-
mentation in technical domains [ 11]. The Recursive (300-character) approach achieved the lowest
L2 distance (0.4295 ± 0.0461) and highest Answer Relevance (0.8663 ± 0.0199) and Khmer IoU
8

(0.6441 ± 0.0347). Moreover, a paired t-test confirmed that the improvement over Sentence-based
chunking is statistically significant for L2 distance at p = 0.0121.
This achievement is due to the Recursive approach’s ability to preserve “contextual bundles”
[4,6]. In agricultural manuals, instructions are often given within a hierarchical structure of headers,
paragraphs of explanations, and nested steps. The Recursive approach maintains this structural
relationship within a 300-character window. As a result, the conditional statements are kept within
a coherent embedding space with their corresponding actions [ 2].
The Sentence-based approach achieved a comparable Khmer IoU of 0.6384 ± 0.0203 but has a
disadvantage of breaking down procedural logic by disconnecting conditional statements from their
corresponding actions [ 5]. The Khmer-A ware approach achieved a significantly higher L2 distance
of 0.7727 ± 0.0223. Although it is a linguistically motivated approach, it is still far from a proper
solution. The lack of segmentation constraints is a limitation of this approach, as emphasized by
Chea [ 8].
6.2 T rade-offs and Strategy Selection Guidelines
The selection of an effective chunking strategy depends on a balance of four main factors: seman-
tic preservation, linguistic integrity , computational eﬀiciency , and structural preservation. These
factors have been discussed in various studies on text segmentation and chunking [ 4,11].
Khmer-A ware: This strategy be used when preserving Khmer punctuation markers and script-
specific cues is important. It produces a moderate chunk volume of 455 chunks, but in this study
it shows lower retrieval performance than the Recursive method.
Recursive (300-character): This chunking strategy may be used in documents with a hier-
archical structure in which the document structure reflects the semantic content of the text. It
provides the highest retrieval accuracy with a reasonable index size of 1,055 chunks.
LLM-Based: It provides the highest compression rate of 318 chunks. Hence, it may be used
in applications in which storage space is a concern.
Sentence-Based: It used in documents with a linguistic structure in which sentence boundaries
play a crucial role in the content. It provides the lowest retrieval accuracy with a chunk volume of
1,206 chunks.
The results show that the granularity of the segment may be determined based on the document
structure rather than linguistic factors.
6.3 Implications for Low-Resource Language Processing
The results show a guideline for designing retrieval systems in low-resource language documents. It
indicates that preserving the document structure may compensate for linguistic chunking errors [ 8].
The results may be used in designing retrieval systems in Khmer documents in which word
boundaries are not clearly defined. Sentence tokenization may not be effective in Khmer documents.
Hence, preserving the document structure may be used in designing retrieval systems.
The results may be used in designing retrieval systems in documents written in various Southeast
Asian scripts in which word boundaries are not clearly defined. It may be used in designing retrieval
systems in Lao and Thai documents [ 15].
6.4 Limitations
The research has some limitations. Firstly , the research only focuses on the evaluation of Khmer
agricultural manuals. It may not be generalized to conversational, literary , or multi-domain corpora
[1]. Secondly , the experiments only used one multilingual embedding model (BGE-M3) [ 2]. Other
9

embedding architectures may have different sensitivity to chunk length and structural segmentation.
Thirdly , the LLM-based method was only evaluated with one model configuration and prompt
design. It may have a higher performance after optimization [ 20]. Lastly , although using objective
quantitative metrics in the research helps in the objective comparison of results, using human
evaluation in the agricultural domain would have increased the validity of the research in assessing
the quality of retrieval [ 15].
7 Conclusion and F uture W ork
This paper provides a comprehensive evaluation of text chunking strategies in embedding-based
text retrieval in a low-resource language setting, Khmer agricultural documents [ 10]. A unified
evaluation framework is proposed to incorporate semantic-based text retrieval metrics (L2 distance
and cosine similarity) and language model-based fidelity measures (Khmer Coverage and Khmer
IoU) [ 1]. Performance is evaluated using 5-fold cross-validation to ensure statistical validity . Ex-
perimental results demonstrate that structure-preserving recursive chunking (300-character setting)
provides the most promising results in text retrieval in a low-resource language setting, achieving
minimal text retrieval distance and maximum semantic similarity , with statistically significant
improvements over Sentence-based segmentation [ 6].
These results confirm that text chunking strategies significantly impact text retrieval perfor-
mance in a low-resource language setting [ 15,16]. Document structure and contextual continuity
are key factors in text embeddings that provide more discriminative text embeddings compared
to sentence-based or script-based text segmentation [ 5,8]. Although text chunking using LLMs
provides compact text embeddings and adaptive topic boundaries [ 20], it is not as effective as
structure-preserving recursive splitting in text retrieval in a technical procedural document setting.
In summary , it is believed that document structure is a more important factor in text retrieval
performance compared to text chunking strategies in a low-resource language setting [ 10].
Possible future work in this area is to explore hybrid text segmentation strategies that integrate
structure-based recursive splitting and semantic text refinement [ 4]. More text embeddings and
reranking strategies will be investigated to assess text chunking granularity sensitivity [ 1,2]. The
proposed text segmentation and evaluation framework will also be extended to explore text retrieval
in other low-resource language and in other domains beyond agricultural documents.
References
[1] P . Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W.-T.
Yih, T. Rocktäschel, S. Riedel, and D. Kiela, “Retrieval-augmented generation for knowledge-
intensive nlp tasks,” in Advances in Neural Information Processing Systems (NeurIPS) , vol. 33,
2020, pp. 9459–9474.
[2] J. Chen, S. Xiao, P . Zhang, K. Luo, D. Lian, and Z. Liu, “M3-embedding: Multi-linguality ,
multi-functionality , multi-granularity text embeddings through self-knowledge distillation,”
2024. [Online]. A vailable: https://arxiv.org/abs/2402.03216
[3] J. Johnson, M. Douze, and H. Jégou, “Billion-scale similarity search with gpus,” 2017.
[Online]. A vailable: https://arxiv.org/abs/1702.08734
[4] R. Qu, R. T u, and F. Bao, “Is semantic chunking worth the computational cost?” 2024.
[Online]. A vailable: https://arxiv.org/abs/2410.13070
10

[5] N. Reimers and I. Gurevych, “Sentence-bert: Sentence embeddings using siamese bert-
networks,” in Proceedings of the 2019 Conference on Empirical Methods in Natural Language
Processing (EMNLP-IJCNLP) , 2019, pp. 3982–3992.
[6] M. Günther, I. Mohr, D. J. Williams, B. W ang, and H. Xiao, “Late chunking: Contextual
chunk embeddings using long-context embedding models,” 2025.
[7] K. Guu, K. Lee, Z. T ung, P . Pasupat, and M.-W. Chang, “Realm: Retrieval-augmented lan-
guage model pre-training,” arXiv preprint arXiv:2002.08909 , 2020.
[8] V. Chea, Y. Kyaw, C. Ding, M. Utiyama, A. Finch, and E. Sumita, “Khmer word segmentation
using conditional random fields,” 2015.
[9] R. Buoy , N. T aing, and S. Kor, “Khmer word segmentation using bilstm networks,” 2020.
[10] S. Sry and A. Nguyen, “A review of khmer word segmentation and part-of-speech tagging and
an experimental study using bidirectional long short-term memory ,” Ho Chi Minh City Open
University Journal of Science: Engineering and T echnology , vol. 12, pp. 23–34, 2022.
[11] O. Koshorek, A. Cohen, N. Mor, M. Rotman, and J. Berant, “T ext segmentation as a
supervised learning task,” 2018. [Online]. A vailable: https://arxiv.org/abs/1803.09337
[12] A. A. Alemi and P . Ginsparg, “T ext segmentation based on semantic word embeddings,” arXiv
preprint arXiv:1503.05543 , 2015.
[13] C. Merola and J. Singh, “Reconstructing context: Evaluating advanced chunking strategies for
retrieval-augmented generation,” 2025. [Online]. A vailable: https://arxiv.org/abs/2504.19754
[14] T. V u, D. Q. Nguyen, D. Q. Nguyen, M. Dras, and M. Johnson, “V ncorenlp: A vietnamese
natural language processing toolkit,” in Proceedings of NAACL-HL T Demonstrations . Asso-
ciation for Computational Linguistics, 2018, pp. 56–60.
[15] Y. Susanto, A. V. Hulagadri, J. R. Montalan, J. G. Ngui, X. B. Y ong, W. Leong, H. Rengarajan,
P . Limkonchotiwat, Y. Mai, and W. C. Tjhi, “Sea-helm: Southeast asian holistic evaluation
of language models,” 2025. [Online]. A vailable: https://arxiv.org/abs/2502.14301
[16] W. Q. Leong, J. G. Ngui, Y. Susanto, H. Rengarajan, K. Sarveswaran, and W. C. Tjhi,
“Bhasa: A holistic southeast asian linguistic and cultural evaluation suite for large language
models,” 2023. [Online]. A vailable: https://arxiv.org/abs/2309.06085
[17] I. Strauss, J. Y ang, T. O’Reilly , S. Rosenblat, and I. Moure, The Attribution Crisis in
LLM Search Results: Estimating Ecosystem Exploitation , Jun. 2025. [Online]. A vailable:
http://dx.doi.org/10.35650/AIDP.4114.d.2025
[18] X. Zhang, N. Thakur, O. Ogundepo, E. Kamalloo, D. Alfonso-Hermelo, X. Li, Q. Liu,
M. Rezagholizadeh, and J. Lin, “Making a miracl: Multilingual information retrieval across a
continuum of languages,” 2022. [Online]. A vailable: https://arxiv.org/abs/2210.09984
[19] A. F. Hidayatullah, R. Apong, D. T. C. Lai, and A. Qazi, “Pre-trained language model for code-
mixed text in indonesian, javanese, and english using transformer,” Social Network Analysis
and Mining , vol. 15, 2025.
[20] N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni, and P . Liang, “Lost in
the middle: How language models use long contexts,” arXiv preprint arXiv:2307.03172 , 2023.
11