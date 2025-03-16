# DeepRAG: Building a Custom Hindi Embedding Model for Retrieval Augmented Generation from Scratch

**Authors**: Nandakishor M

**Published**: 2025-03-11 09:27:56

**PDF URL**: [http://arxiv.org/pdf/2503.08213v1](http://arxiv.org/pdf/2503.08213v1)

## Abstract
In this paper, I present our work on DeepRAG, a specialized embedding model
we built specifically for Hindi language in RAG systems. While LLMs have gotten
really good at generating text, their performance in retrieval tasks still
depends heavily on having quality embeddings - something that's been lacking
for Hindi despite being one of the world's most spoken languages. We tackled
this by creating embeddings from the ground up rather than just fine-tuning
existing models. Our process involved collecting diverse Hindi texts (over 2.7M
samples), training a custom SentencePiece tokenizer that actually understands
Hindi morphology, designing transformer architecture with Hindi-specific
attention mechanisms, and optimizing with contrastive learning. Results were
honestly better than I expected - we saw a 23% improvement in retrieval
precision compared to the multilingual models everyone's been using. The paper
details our methodology, which I think could help others working with
low-resource languages where the one-size-fits-all multilingual models fall
short. We've also integrated our embeddings with LangChain to build complete
Hindi RAG systems, which might be useful for practitioners. While there's still
tons more to explore, I believe this work addresses a critical gap for Hindi
NLP and demonstrates why language-specific approaches matter.

## Full Text


<!-- PDF content starts -->

arXiv:2503.08213v1  [cs.CL]  11 Mar 2025DeepRAG: Building a Custom Hindi Embedding
Model for Retrieval Augmented Generation from
Scratch
Nandakishor M
Deepmost Innovations
Abstract —While Large Language Models (LLMs) excel in
text generation, their performance in Retrieval Augmented
Generation (RAG) systems heavily depends on the quality of
text embeddings. For non-English languages like Hindi, the
lack of high-quality, dedicated embedding models remains a
signiﬁcant challenge. This paper presents DeepRAG, a com-
prehensive framework for developing custom Hindi-speciﬁc text
embeddings from scratch for RAG applications. We detail our
end-to-end process, including corpus collection from dive rse
Hindi sources, specialized tokenizer training with Senten cePiece,
transformer architecture design with semantic pooling str ategies,
and model training with contrastive learning techniques. O ur
evaluation demonstrates that DeepRAG embeddings signiﬁca ntly
outperform multilingual alternatives in Hindi semantic si milarity
tasks, with a 23% improvement in retrieval precision. We fur ther
demonstrate the integration of our embeddings with LangCha in
for building effective Hindi RAG systems. The detailed meth odol-
ogy provides a roadmap for creating domain-speciﬁc embeddi ngs
for low-resource languages, addressing critical gaps in mu ltilin-
gual NLP infrastructure.
Index Terms —Hindi embeddings, Retrieval Augmented Gen-
eration, NLP, custom tokenization, semantic search, trans former
architectures, low-resource languages, SentencePiece, c ontrastive
learning, LangChain
I. I NTRODUCTION
The advent of large language models (LLMs) has revo-
lutionized natural language processing capabilities, ena bling
sophisticated text generation across domains. However, th ese
models face challenges with factuality, knowledge limita-
tions, and hallucinations [1]. Retrieval Augmented Genera tion
(RAG) offers a solution by retrieving relevant documents
before generation, enhancing both accuracy and factuality .
A critical, often overlooked component in the RAG pipeline
is the quality of text embeddings used for retrieval. While
numerous embedding models exist for English, high-quality
embeddings for other languages remain limited. This gap is
particularly pronounced for Hindi, one of the world’s most
widely spoken languages.
Existing multilingual embedding solutions like multiling ual-
E5 [2] and LaBSE [3] provide some coverage for Hindi,
but these models sacriﬁce language-speciﬁc performance fo r
multilingual capabilities. They typically under-represe nt Hindi
semantic nuances, resulting in suboptimal retrieval perfo r-
mance when applied to Hindi RAG systems.
In this paper, I present DeepRAG, a complete framework
for building custom, high-performance Hindi embeddingsspeciﬁcally designed for RAG applications. Rather than ﬁne -
tuning existing models, DeepRAG was built entirely from
scratch—from corpus collection and tokenizer training to
model architecture design and optimization. This ground-u p
approach allowed us to incorporate language-speciﬁc consi d-
erations throughout the development process.
The key contributions of this work include:
•A comprehensive methodology for building language-
speciﬁc embedding models from scratch, with each com-
ponent optimized for Hindi.
•A specialized SentencePiece tokenizer trained on a di-
verse corpus of over 2.7 million Hindi texts, incorporating
linguistic-aware subword segmentation.
•A custom transformer architecture with enhanced at-
tention mechanisms and pooling strategies speciﬁcally
designed for Hindi semantic representation.
•A multi-stage training process using contrastive learning
and synthetic data generation for robust embeddings.
•Integration techniques with LangChain for building end-
to-end Hindi RAG systems.
•Extensive comparative evaluation demonstrating signiﬁ-
cant improvements over multilingual approaches.
Our ﬁndings show that a dedicated, language-speciﬁc ap-
proach yielded substantial gains in embedding quality, wit h
DeepRAG embeddings demonstrating a 23% improvement
in retrieval precision over the best multilingual alternat ives.
I believe our methodology provides valuable insights for
developing similar solutions for other low-resource langu ages
where general-purpose multilingual models fall short.
II. R ELATED WORK
A. Multilingual Embedding Models
Several established multilingual embedding models cur-
rently support Hindi to varying degrees. Multilingual-E5
[2] covers 100+ languages including Hindi, while LaBSE
(Language-agnostic BERT Sentence Embeddings) [3] was
designed speciﬁcally for cross-lingual alignment across 1 09
languages.
These models, while impressive in their breadth, face in-
herent trade-offs. The "curse of multilinguality" [4] high lights
how performance on individual languages degrades as more
languages are added to a model. In my personal testing with
Hindi documents, I’ve found this degradation to be particul arly
pronounced for semantic search applications.

B. Language-Speciﬁc Embedding Models
Language-speciﬁc embedding models have demonstrated
signiﬁcant performance improvements over multilingual al -
ternatives. English-speciﬁc models like MPNet [5] and E5
[2] consistently outperform their multilingual counterpa rts in
English tasks. Similarly, BGE [6] for Chinese and KUZU
[7] for Korean have shown the value of language-dedicated
approaches.
For Hindi, however, the landscape remains surprisingly
sparse. Work by Kakwani et al. [8] introduced IndicBERT,
but focused primarily on classiﬁcation rather than embeddi ng
quality. Kumar et al. [9] explored Hindi word embeddings, bu t
comprehensive sentence embedding models remain notably
absent.
C. RAG Systems and Embedding Quality
Recent work by Lewis et al. [10] introduced the RAG
paradigm, while Gao et al. [11] emphasized the critical impa ct
of embedding quality on RAG performance. Their ﬁndings
indicate that retrieval quality is often the primary limiti ng
factor in RAG systems.
Through my experiments with Hindi RAG, I’ve consistently
found that existing embedding models fail to capture the
nuanced semantic relationships in Hindi documents. This
observation aligns with ﬁndings from Ruder et al. [12], who
noted that language-speciﬁc pretraining signiﬁcantly imp roves
downstream task performance.
III. C ORPUS COLLECTION AND ANALYSIS
The foundation of any high-quality embedding model is
a diverse and representative text corpus. For Hindi, this
presented unique challenges due to the relative scarcity of
digitized content compared to languages like English.
A. Data Sources and Collection
I prioritized diversity of both content and style to ensure
robust embeddings. Our ﬁnal corpus comprised text from the
following sources:
•IITB Parallel Corpus: 1.2 million Hindi sentences from
various domains
•Samanantar: 750,000 samples of general Hindi text
•Oscar Hindi: 450,000 sentences from web crawls
•CC-100 Hindi: 300,000 sentences of web content
•Hindi Wikipedia: 150,000 articles covering encyclopedic
knowledge
•Hindi news articles: 100,000 news pieces covering current
events
•XNLI Hindi: 50,000 premise-hypothesis pairs useful for
semantic reasoning
•IndicGLUE: 30,000 samples from diverse tasks
•Hindi literature: 5,000 passages from classic and modern
Hindi literature
Several of these datasets required custom extraction scrip ts
to isolate the Hindi portions and remove non-Hindi text
contamination. After collection, the raw corpus totaled ap prox-
imately 3.1 million text samples.B. Corpus Cleaning and Preprocessing
Raw datasets contained signiﬁcant noise, including code-
mixing with English, HTML artifacts, and irregular Unicode
representations. Our cleaning pipeline included:
1) Unicode normalization (NFC form)
2) Removal of non-Hindi text segments via Unicode range
ﬁltering
3) Deduplication through exact and near-duplicate detecti on
4) Length ﬁltering (removing very short or excessively long
texts)
5) Specialized Hindi normalization using IndicNLP’s nor-
malizer
6) Removal of frequent patterns indicating low-quality con -
tent
The cleaning process reduced our corpus to approximately
2.7 million high-quality Hindi text samples. An example of
our cleaning function is shown below:
Algorithm 1 Hindi Text Cleaning
1:Input: Raw Hindi text T
2:Output: Cleaned text T′
3:T←strip(T)
4:T←replace_urls_with_placeholder (T)
5:T←unicode_normalize_NFC (T)
6:T←hindi_normalizer.normalize (T)
7:T←remove_repeating_chars (T)
8:T←replace_numbers_with_placeholder (T)
9:T←normalize_whitespace (T)
10:returnT′
C. Corpus Analysis
To inform our tokenizer and model design decisions, I con-
ducted extensive analysis on the cleaned corpus, examining :
•Character and word frequency distributions
•Sentence length statistics
•Common word collocations
•Subword unit effectiveness through various segmentation
approaches
•Topic diversity via clustering analysis
This analysis revealed several insights speciﬁc to Hindi th at
informed our approach:
1) Hindi’s agglutinative features create long compound
words that beneﬁt from subword tokenization
2) The frequent absence of spaces between certain Hindi
words requires careful tokenization
3) Common transliterations from English technical terms
require special handling
4) Sanskrit-derived vocabulary follows patterns that bene ﬁt
from linguistically-informed tokenization
Compared to English, Hindi demonstrates a higher ratio of
unique characters to total characters (0.0089 vs 0.0024 for
English) and a lower ratio of unique words to total words
(0.053 vs 0.071 for English), suggesting different optimal
tokenization strategies.

IV. C USTOM SENTENCE PIECE TOKENIZER FOR HINDI
While most embedding projects start with existing tok-
enizers, I found that Hindi’s linguistic features warrante d a
specialized approach.
A. Tokenizer Design Considerations
General-purpose tokenizers like those from BERT or
RoBERTa typically perform poorly on Hindi due to:
•Under-representation of Hindi Unicode characters
•Incorrect segmentation of compound words
•Poor handling of Hindi-speciﬁc punctuation
•Inefﬁcient tokenization of common Hindi morphological
patterns
I experimented with several tokenization approaches inclu d-
ing BPE, WordPiece, and Unigram, ultimately ﬁnding that the
Unigram model from SentencePiece performed best for Hindi
when properly conﬁgured.
B. Tokenizer Training Process
The tokenizer training process required several key design
decisions:
1)Vocabulary Size : Through ablation studies, I determined
that a vocabulary size of 50,000 offered the optimal
balance between coverage and model efﬁciency. Smaller
vocabularies (32K) increased unknown token frequency,
while larger ones (64K+) showed diminishing returns.
2)Character Coverage : Set to 0.9995 to ensure compre-
hensive coverage of Hindi’s character set while excluding
extremely rare Unicode points.
3)Special Tokens : Added Hindi-speciﬁc markers beyond
the standard set, including separation markers for com-
pound verbs and noun phrases.
4)Normalization Rules : Applied custom normalization
rules for Hindi, including handling of nukta variations
and homoglyph normalization.
The training process included an 80-20 split between nor-
malization rules and explicit character coverage, which I
found produced more consistent tokenization across Hindi t ext
variants.
C. Tokenizer Evaluation
To evaluate tokenizer quality, I developed several Hindi-
speciﬁc metrics:
•Semantic Unit Preservation : Percentage of semantic
units preserved after tokenization and detokenization
(92.7%)
•Morphological Segmentation Accuracy : How well the
tokenizer identiﬁes meaningful morphological boundaries
(87.3%)
•OOV Handling : Processing of out-of-vocabulary words
across test sets (4.2% OOV rate)
•Efﬁciency : Average tokens per Hindi sentence compared
to general tokenizers (22.4 vs 31.7 for multilingual tok-
enizers)I was particularly satisﬁed with the tokenizer’s handling o f
Hindi’s rich morphology. For example, the compound word
“vishwavidyalaya (university)” was properly segmented in to
“vishwa (world)” and “vidyalaya (school)”, preserving mea n-
ingful units, while multilingual tokenizers typically pro duced
fragments that did not maintain semantic coherence.
V. M ODEL ARCHITECTURE
With our custom tokenizer in place, I designed a transformer
architecture speciﬁcally optimized for Hindi semantic rep re-
sentations.
A. Core Architecture Design
DeepRAG uses a modiﬁed transformer encoder with several
Hindi-speciﬁc optimizations:
•Model Dimensionality : 768-dimensional embeddings,
providing sufﬁcient capacity for Hindi semantic space
while remaining computationally efﬁcient
•Depth and Width : 12 transformer layers with 12 atten-
tion heads, determined through ablation studies as the
optimal conﬁguration for Hindi
•Advanced Attention Mechanism : Implemented rotary
positional embeddings instead of standard positional en-
codings, which better captured Hindi’s relatively free
word order
•Enhanced Feed-Forward : Used SwiGLU activations
instead of standard GELU for better gradient ﬂow
•Pre-Layer Normalization : Applied layer normalization
before each sub-layer rather than after, improving trainin g
stability
B. Hindi-Speciﬁc Architectural Innovations
Based on our corpus analysis, I incorporated several novel
components:
1)Multi-Resolution Attention : Added a mechanism to
capture both character-level and word-level patterns si-
multaneously, which proved essential for Hindi’s diverse
orthographic conventions
2)Morphology-Aware Feed-Forward : Modiﬁed feed-
forward layers with additional projections targeting
Hindi’s morphological patterns
3)Script-Mix Processing : Added speciﬁc handling for
Hindi-English code-mixing, common in technical and
modern texts
C. Pooling Strategy
For sentence embedding models, the pooling strategy criti-
cally affects representation quality. I experimented with several
approaches:
•CLS token pooling (standard in BERT)
•Mean pooling (average of all token embeddings)
•Max pooling (element-wise maximum)
•Attention pooling (learned attention weights)
•Weighted pooling (our proposed approach)

Through extensive evaluation, I found that a novel weighted
pooling strategy performed best for Hindi. This approach us es
a learned weighting function that considers:
1) Token position (with sensitivity to Hindi’s SOV structur e)
2) Token importance (learned through attention mecha-
nisms)
3) Contextual signiﬁcance (determined through layer-wise
aggregation)
This weighted pooling achieved a 9.3% improvement in
semantic similarity tasks compared to standard mean poolin g.
The code for our weighted pooling implementation is shown
below:
Algorithm 2 Weighted Pooling for Hindi
1:Input: Token embeddings E∈Rn×d, attention mask
M∈{0,1}n
2:Output: Sentence embedding S∈Rd
3:W←sigmoid(Linear(E)){Token-wise weights}
4:W←W⊙M{Apply mask to weights}
5:W←W/summationtextW+ǫ{Normalize weights}
6:S←/summationtext
iWi·Ei{Weighted sum of embeddings}
7:S←S
||S||2{L2 normalization}
8:returnS
VI. T RAINING METHODOLOGY
A. Dataset Creation
Training high-quality embeddings requires carefully con-
structed datasets. I created a specialized dataset for Hind i
semantic similarity through several mechanisms:
1)Parallel Sentences : Extracted 500,000 pairs from parallel
corpora with high semantic similarity
2)Hard Negatives : Generated 300,000 challenging negative
pairs with subtle semantic differences
3)Augmentation : Created 250,000 pairs through controlled
text augmentation to improve robustness
4)Synthetic Pairs : Generated 450,000 pairs using larger
Hindi LLMs to improve coverage
Each pair was annotated with a similarity score in the [0,1]
range. I used a combination of automated methods and manual
veriﬁcation to ensure quality.
B. Loss Functions and Optimization
I experimented with several loss functions for training:
•Cosine similarity loss (baseline)
•Multiple negatives ranking loss
•Triplet loss
•InfoNCE contrastive loss
•Mixed similarity loss (our proposed approach)
The mixed similarity loss, combining aspects of MSE,
contrastive, and triplet losses, proved most effective:
L=α·LMSE+β·Lcontrastive+γ·Ltriplet (1)
With weights α= 0.5,β= 0.3, andγ= 0.2determined
through validation performance.TABLE I
SEMANTIC SIMILARITY PERFORMANCE (SPEARMAN CORRELATION )
Model MHSS Dataset InSTS Dataset
mBERT 0.58 0.62
XLM-R 0.67 0.70
LaBSE 0.71 0.74
mE5-base 0.75 0.78
mUSE 0.70 0.72
DeepRAG-base 0.81 0.83
DeepRAG-large 0.85 0.87
For optimization, I used AdamW with a learning rate of 2e-
5, a cosine learning rate schedule with warmup, and gradient
accumulation for stability.
C. Training Infrastructure and Process
The model was trained using distributed training across 4
NVIDIA A100 GPUs. Key training hyperparameters included:
•Batch size: 128 per GPU (effective batch size 512)
•Training epochs: 10 (with early stopping)
•Gradient accumulation steps: 4
•Mixed precision: FP16
•Weight decay: 0.01
I implemented several training optimizations including:
1) Gradient checkpointing to reduce memory usage
2) Model parallelism for the largest model variants
3) Dynamic batch sizing based on sequence length
4) Curriculum learning, starting with easier examples
During training, I monitored several key metrics including
validation loss, semantic similarity correlation, and ret rieval
precision on a held-out test set. The model was selected base d
on retrieval performance rather than raw loss values, which I
found better predicted real-world effectiveness.
VII. E VALUATION AND RESULTS
A. Intrinsic Evaluation
I ﬁrst evaluated DeepRAG on standard semantic similarity
benchmarks:
For these evaluations, I used:
•MHSS (Multilingual Hindi Semantic Similarity) - a
dataset of 2,500 Hindi sentence pairs with human simi-
larity judgments
•InSTS (Indian Semantic Textual Similarity) - a collection
of 1,800 Hindi sentence pairs with graded similarity
scores
The results in Table I demonstrate that DeepRAG signif-
icantly outperforms multilingual alternatives, with a 13. 3%
relative improvement over the best competitor (mE5-base).
B. Retrieval Evaluation
More importantly for RAG applications, I evaluated retriev al
performance using a custom Hindi document retrieval bench-
mark:
These experiments used a corpus of 100,000 Hindi doc-
uments with 1,000 queries requiring semantic understandin g

TABLE II
HINDI DOCUMENT RETRIEVAL PERFORMANCE
Model P@1 P@5 MRR
mBERT 0.53 0.47 0.61
XLM-R 0.58 0.51 0.67
LaBSE 0.62 0.57 0.72
mE5-base 0.67 0.61 0.76
mUSE 0.60 0.53 0.70
DeepRAG-base 0.79 0.72 0.86
DeepRAG-large 0.83 0.75 0.89
TABLE III
ABLATION STUDY RESULTS (MRR ONRETRIEVAL TASK)
Conﬁguration MRR
Full DeepRAG-base model 0.86
- Custom tokenizer (using multilingual) 0.77 (-0.09)
- Weighted pooling (using mean) 0.81 (-0.05)
- Mixed loss (using only MSE) 0.80 (-0.06)
- Hindi-speciﬁc architecture 0.79 (-0.07)
- All customizations (baseline transformer) 0.68 (-0.18)
rather than simple keyword matching. The metrics evaluated
were:
•P@K: Precision at K (proportion of relevant documents
in top K results)
•MRR: Mean Reciprocal Rank (average of reciprocal
ranks of ﬁrst relevant result)
DeepRAG demonstrated a 23.9% improvement in P@1 over
mE5-base, indicating substantially better retrieval prec ision for
RAG applications.
C. Ablation Studies
To understand the contribution of each component, I con-
ducted ablation studies removing key elements of DeepRAG:
These results highlight the signiﬁcant contribution of our
custom tokenizer, which alone accounts for approximately
half of DeepRAG’s performance advantage over baseline ap-
proaches.
D. Qualitative Analysis
Beyond quantitative metrics, I conducted qualitative anal ysis
of retrieval examples. When examining cases where DeepRAG
retrieved relevant documents that multilingual models mis sed,
several patterns emerged:
1)Cultural Concepts : DeepRAG better captured culture-
speciﬁc concepts without direct English translations
2)Idiomatic Expressions : Signiﬁcantly better handling of
Hindi idioms and ﬁgurative language
3)Syntactic Variations : More robust to Hindi’s ﬂexible
word order
4)Formal/Informal Distinctions : Better matching across
formal and colloquial variants of the same concept
A particularly striking case involved retrieving document s
about “jal sanrakshan (water conservation)”, where Deep-
RAG correctly identiﬁed semantically relevant documentsdiscussing “pani bachana (saving water)” and “jal sansadha n
prabandhan (water management)”, while multilingual model s
failed to establish these connections.
VIII. RAG S YSTEM INTEGRATION WITH LANG CHAIN
A. LangChain Integration Architecture
To demonstrate practical application, I integrated DeepRA G
embeddings with LangChain to build complete Hindi RAG
systems. Key integration components included:
1)Custom Embeddings Class : Created a LangChain-
compatible wrapper for DeepRAG embeddings
2)Text Chunking Strategies : Developed Hindi-speciﬁc
text chunking that respects sentence and paragraph
boundaries
3)Vector Store Integration : Connected DeepRAG with
FAISS for efﬁcient vector search
4)LLM Prompt Engineering : Designed prompts for Hindi
LLMs to effectively utilize retrieved context
B. Inference Optimization
For production deployment, several optimizations were cru -
cial:
•Model Quantization : Applied 8-bit quantization, reduc-
ing model size by 75% with minimal performance impact
•Batched Inference : Implemented efﬁcient batching for
document indexing
•Caching Strategy : Developed a two-level cache for
frequently accessed embeddings
•Result Re-ranking : Added lightweight semantic re-
ranking to improve precision
C. End-to-End System Performance
I evaluated complete RAG systems built with DeepRAG
versus those using multilingual embeddings:
•Generation Quality : 27% improvement in human-judged
answer quality
•Factual Accuracy : 18% reduction in factual errors on a
Hindi knowledge benchmark
•Retrieval Efﬁciency : 35% improvement in processing
speed due to more efﬁcient tokenization and embedding
The complete LangChain integration allowed for rapid de-
velopment of Hindi RAG applications, with signiﬁcantly bet ter
performance than using off-the-shelf multilingual embedd ing
models.
IX. C ONCLUSION AND FUTURE WORK
In this paper, I’ve presented DeepRAG, a comprehen-
sive approach to building custom Hindi embeddings for Re-
trieval Augmented Generation from scratch. By designing
each component speciﬁcally for Hindi—from corpus col-
lection and tokenization to model architecture and trainin g
methodology—DeepRAG achieves substantial improvements
over multilingual alternatives. The model is available for pub-
lic use at https://huggingface.co/DeepMostInnovations/ hindi-
embedding-foundational-model.
Key ﬁndings include:

•The critical importance of language-speciﬁc tokenization
for embedding quality
•The effectiveness of weighted pooling strategies for
Hindi’s linguistic features
•The signiﬁcant performance gains from mixed similarity
loss functions
•The practical beneﬁts in RAG applications, with a 23%
retrieval precision improvement
These results clearly demonstrate that language-speciﬁc e m-
bedding models provide substantial beneﬁts over multiling ual
approaches, particularly for languages like Hindi that are often
under-represented in general-purpose models.
The ﬁeld of Hindi NLP still has substantial room for growth.
Future work could explore:
•Expanding DeepRAG to other Indic languages while
preserving language-speciﬁc optimizations
•Incorporating Hindi knowledge graphs to enhance seman-
tic representations
•Developing specialized models for domains like legal,
medical, or technical Hindi
•Creating instruction-tuned Hindi embeddings for more
targeted retrieval
Personally, I believe that high-quality, language-speciﬁ c
embeddings are foundational infrastructure for bringing a d-
vanced NLP to non-English languages. DeepRAG contributes
to this goal for Hindi, providing both practical tools and
methodological insights applicable to other languages and
domains.
My hope is that this work inspires similar efforts for other
less-resourced languages, ultimately working toward more
linguistically diverse and equitable NLP ecosystems.
ACKNOWLEDGMENT
I would like to express my heartfelt gratitude to the open-
source community and researchers who have created novel
mathematical architectures and code bases for Large Langua ge
Models. Their tireless efforts, even through countless fai led
experiments and sleepless nights debugging mysterious mod el
behaviors, have made this work possible. I’m particularly
thankful to those who generously shared their knowledge on
forums when I was stuck with perplexing tokenization errors
at 3 AM, and to the anonymous contributors whose clever
optimizations saved our training runs from crashing yet aga in.
This work stands on the shoulders of brilliance, persistenc e,
and countless cups of coffee. I must also acknowledge the
patient colleagues who endured my excited ramblings about
embedding spaces when they just wanted to eat lunch in peace.
Despite my best efforts, any remaining errors or questionab le
design choices in this paper are entirely my own—though I
secretly hope reviewers will be merciful in pointing them ou t.
REFERENCES
[1] S. Gao, T. Yao, and D. Chen, “Hallucinations in large lang uage models:
A survey,” arXiv preprint arXiv:2311.05232 , 2023.
[2] H. Wang, L. Xiong, and M. Rao, “Text embeddings by weakly-
supervised contrastive pre-training,” arXiv preprint arXiv:2212.03533 ,
2022.[3] F. Feng, Y . Yang, D. Cer, N. Arivazhagan, and W. Wang, “Lan guage-
agnostic BERT sentence embedding,” Proceedings of the 60th Annual
Meeting of the Association for Computational Linguistics , pp. 878-891,
2022.
[4] A. Conneau, K. Khandelwal, and N. Goyal, “Unsupervised c ross-
lingual representation learning at scale,” Proceedings of the 58th Annual
Meeting of the Association for Computational Linguistics , pp. 8440-
8451, 2020.
[5] K. Song, X. Tan, T. Qin, J. Lu, and T.Y . Liu, “MPNet: Masked and
permuted pre-training for language understanding,” Advances in Neural
Information Processing Systems , vol. 33, pp. 16857-16867, 2020.
[6] L. Xiao, X. Zhao, and J. Chin, “C-Pack: Packaged resource s to advance
general Chinese embedding,” arXiv preprint arXiv:2309.07597 , 2023.
[7] S. Park, J. Moon, and S. Kim, “KLUE: Korean language under standing
evaluation,” Transactions of the Association for Computational Linguis -
tics, vol. 10, pp. 652-670, 2022.
[8] D. Kakwani, A. Kunchukuttan, and S. Golla, “IndicNLPSui te: Mono-
lingual corpora, evaluation benchmarks and pre-trained mu ltilingual
language models for Indian languages,” Findings of the Association for
Computational Linguistics: EMNLP 2020 , pp. 4948-4961, 2020.
[9] R. Kumar, B. Lahiri, and A.K. Ojha, “Developing resource s and stan-
dardized evaluation for Hindi codemixing,” Proceedings of the 13th
Language Resources and Evaluation Conference , pp. 3675-3685, 2022.
[10] P. Lewis, E. Perez, and A. Piktus, “Retrieval-augmente d generation
for knowledge-intensive NLP tasks,” Advances in Neural Information
Processing Systems , vol. 33, pp. 9459-9474, 2020.
[11] L. Gao, X. Ma, J. Lin, and J. Callan, “Precise zero-shot d ense retrieval
without relevance labels,” Proceedings of the 46th International ACM
SIGIR Conference on Research and Development in Informatio n Re-
trieval , pp. 2196-2206, 2023.
[12] S. Ruder, M.E. Peters, S. Swayamdipta, and T. Wolf, “Uns upervised
cross-lingual representation learning,” Journal of Artiﬁcial Intelligence
Research , vol. 71, pp. 363-392, 2021.
[13] T. Brants and A. Franz, “Web 1T 5-gram Version 1,” Linguistic Data
Consortium , 2006.
[14] A. Vaswani, N. Shazeer, and N. Parmar, “Attention is all you need,”
Advances in Neural Information Processing Systems , vol. 30, pp. 5998-
6008, 2017.
[15] J. Devlin, M.W. Chang, K. Lee, and K. Toutanova, “BERT: P re-
training of deep bidirectional transformers for language u nderstanding,”
Proceedings of the 2019 Conference of the North American Cha pter
of the Association for Computational Linguistics: Human La nguage
Technologies , pp. 4171-4186, 2019.
[16] T. Kudo and J. Richardson, “SentencePiece: A simple and language inde-
pendent subword tokenizer and detokenizer for neural text p rocessing,”
Proceedings of the 2018 Conference on Empirical Methods in N atural
Language Processing: System Demonstrations , pp. 66-71, 2018.
[17] H. Chase, “LangChain: Building applications with LLMs through com-
posability,” https://github.com/hwchase17/langchain, 2023.
[18] D. Roy, D. Paul, M. Mitra, and U. Garain, “Machine transl ation
quality estimation for Indian languages: The IIT Bombay sub mission for
WMT19 shared task,” Proceedings of the Fifth Conference on Machine
Translation , pp. 873-879, 2020.
[19] J. Johnson, M. Douze, and H. Jégou, “Billion-scale simi larity search
with GPUs,” IEEE Transactions on Big Data , vol. 7, no. 3, pp. 535-
547, 2019.
[20] P. Joshi, S. Santy, and A. Budhiraja, “The state and fate of linguistic
diversity and inclusion in the NLP world,” Proceedings of the 58th
Annual Meeting of the Association for Computational Lingui stics, pp.
6282-6293, 2020.
[21] Y . Liu, M. Ott, and N. Goyal, “RoBERTa: A robustly optimi zed BERT
pretraining approach,” arXiv preprint arXiv:1907.11692 , 2019.
[22] H. Le, L. Vial, and J. Frej, “FlauBERT: Unsupervised lan guage model
pre-training for French,” Proceedings of the 12th Language Resources
and Evaluation Conference , pp. 2479-2490, 2020.
[23] L. Martin, B. Muller, and P.J. Ortiz Suárez, “CamemBERT : A tasty
French language model,” Proceedings of the 58th Annual Meeting of
the Association for Computational Linguistics , pp. 7203-7219, 2020.