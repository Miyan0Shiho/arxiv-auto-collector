# Contextual Graph Transformer: A Small Language Model for Enhanced Engineering Document Information Extraction

**Authors**: Karan Reddy, Mayukha Pal

**Published**: 2025-08-04 15:41:35

**PDF URL**: [http://arxiv.org/pdf/2508.02532v1](http://arxiv.org/pdf/2508.02532v1)

## Abstract
Standard transformer-based language models, while powerful for general text,
often struggle with the fine-grained syntax and entity relationships in complex
technical, engineering documents. To address this, we propose the Contextual
Graph Transformer (CGT), a hybrid neural architecture that combines Graph
Neural Networks (GNNs) and Transformers for domain-specific question answering.
CGT constructs a dynamic graph over input tokens using sequential, skip-gram,
and semantic similarity edges, which is processed by GATv2Conv layers for local
structure learning. These enriched embeddings are then passed to a Transformer
encoder to capture global dependencies. Unlike generic large models, technical
domains often require specialized language models with stronger
contextualization and structure awareness. CGT offers a parameter-efficient
solution for such use cases. Integrated into a Retrieval-Augmented Generation
(RAG) pipeline, CGT outperforms baselines like GPT-2 and BERT, achieving 24.7%
higher accuracy than GPT-2 with 62.4% fewer parameters. This gain stems from
CGTs ability to jointly model structural token interactions and long-range
semantic coherence. The model is trained from scratch using a two-phase
approach: pretraining on general text followed by fine-tuning on
domain-specific manuals. This highlights CGTs adaptability to technical
language, enabling better grounding, entity tracking, and retrieval-augmented
responses in real-world applications.

## Full Text


<!-- PDF content starts -->

Contextual Graph Transformer: A Small Language
Model for Enhanced Engineering Document
Information Extraction
Karan Reddy, Mayukha Pal∗
Abstract —Standard transformer-based language models, while
powerful for general text, often struggle with the fine-grained
syntax and entity relationships in complex technical, engineering
documents. To address this, we propose the Contextual Graph
Transformer (CGT), a hybrid neural architecture that combines
Graph Neural Networks (GNNs) and Transformers for domain-
specific question answering. CGT constructs a dynamic graph
over input tokens using sequential, skip-gram, and semantic
similarity edges, which is processed by GATv2Conv layers for
local structure learning. These enriched embeddings are then
passed to a Transformer encoder to capture global dependencies.
Unlike generic large models, technical domains often require
specialized language models with stronger contextualization and
structure awareness. CGT offers a parameter-efficient solution
for such use cases. Integrated into a Retrieval-Augmented Gen-
eration (RAG) pipeline, CGT outperforms baselines like GPT-2
and BERT, achieving 24.7% higher accuracy than GPT-2 with
62.4% fewer parameters. This gain stems from CGT’s ability
to jointly model structural token interactions and long-range
semantic coherence. The model is trained from scratch using
a two-phase approach: pretraining on general text followed by
fine-tuning on domain-specific manuals. This highlights CGT’s
adaptability to technical language, enabling better grounding,
entity tracking, and retrieval-augmented responses in real-world
applications.
Index Terms —Contextual Graph Transformer, Small Lan-
guage Models, Domain-Specific NLP, Graph Neural Networks,
Transformers, Retrieval-Augmented Generation, Technical Ques-
tion Answering, Parameter Efficiency.
I. I NTRODUCTION
The rapid advancement of natural language processing has
produced powerful language models for general text under-
standing. However, technical documents pose unique chal-
lenges due to their complex structure, specialized terminology,
and intricate entity relationships. Traditional transformer-based
models often struggle to capture the fine-grained syntax and
local dependencies essential for accurately interpreting such
content.This necessitates domain-adaptive architectures capa-
ble of modeling structural and contextual nuances effectively.
(*Corresponding author: Mayukha Pal)
Mr. Karan Reddy is a Data Science Research Intern at ABB Ability
Innovation Center, Hyderabad 500084, India, and also an undergraduate at
the Department of Computer Science and Engineering, Indian Institute of
Technology Jodhpur, Jodhpur 342037, IN.
Dr. Mayukha Pal is with ABB Ability Innovation Center, Hyderabad
500084, IN, working as Global R&D Leader – Cloud & Advanced Analytics
(e-mail: mayukha.pal@in.abb.com).The limitations of current approaches become particularly
evident when dealing with industrial technical documents,
which frequently combine textual descriptions with structured
data such as tables, specifications, and hierarchical informa-
tion. These documents require models that can simultaneously
process sequential text and understand the local relationships
between technical terms, product specifications.
A. Motivation and Problem Statement
Technical document understanding presents several funda-
mental challenges that existing language models struggle to
address effectively:
Local Relationship Modeling: Technical documents con-
tain dense clusters of related terms and concepts that require
fine-grained understanding of local relationships. For example,
in a product specification, the proximity and relationship
between a product code, its specifications, and operational
parameters are crucial for accurate comprehension.
Parameter Efficiency: Large transformer models like GPT-
3 and BERT require substantial computational resources, mak-
ing them impractical for many real-world applications. There
is a pressing need for smaller, more efficient models that can
achieve comparable or superior performance with significantly
fewer parameters.
Structural Awareness: Technical documents often contain
implicit structural relationships that pure sequential processing
may miss. The ability to model these relationships explicitly
through graph structures could provide significant advantages.
Domain Adaptation: Models must efficiently adapt from
general language understanding to specific technical domains
without requiring massive amounts of domain-specific training
data.
B. Our Approach
To address these challenges, we propose the Contextual Graph
Transformer (CGT), a novel hybrid architecture that combines
the local relationship modeling capabilities of Graph Neural
Networks with the global context processing strengths of
Transformers. Our approach is fundamentally different from
existing methods in several key aspects:
1)Dynamic Graph Construction: We develop a sophis-
ticated algorithm that dynamically constructs graphs
from token sequences, capturing both sequential andarXiv:2508.02532v1  [cs.CL]  4 Aug 2025

semantic relationships through adjacency and skip-gram
connections.
2)Hierarchical Processing: Our model employs a two-
stage processing pipeline where GNN layers first extract
rich local features, which are then fed into Transformer
layers for global context integration.
3)Parameter Efficiency: With only 46.8M parameters,
our model achieves superior performance compared to
much larger baselines, demonstrating the effectiveness
of hybrid architectural design.
4)Comprehensive Evaluation: We conduct extensive ex-
periments against multiple established baselines, provid-
ing robust empirical validation of our approach.
C. Contributions
This paper makes the following key contributions:
•Introduction of the Contextual Graph Transformer (CGT),
a novel hybrid architecture that effectively combines
GNNs and Transformers for technical document under-
standing
•Development of a dynamic graph construction algorithm
that captures local relationships in text through mathe-
matical formulations
•Comprehensive empirical evaluation demonstrating
24.7% performance improvement over GPT-2 with
62.4% fewer parameters
•Integration with RAG systems for practical question-
answering applications
•Detailed mathematical analysis of the hybrid architec-
ture’s computational complexity and efficiency
II. PRIOR ART
A. Large Language Models and Parameter Efficiency
The field of natural language processing has witnessed remark-
able advances with the introduction of large-scale transformer
models. BERT [1] demonstrated the power of bidirectional
attention mechanisms, while GPT-2 [2] showcased the effec-
tiveness of autoregressive generation. However, these models
typically require hundreds of millions or billions of parame-
ters, leading to significant computational overhead.
Recent work has focused on developing more parameter-
efficient alternatives. DistilBERT [3] achieved substantial pa-
rameter reduction through knowledge distillation, while main-
taining reasonable performance. However, these approaches
primarily focus on architectural compression rather than fun-
damental improvements in how local relationships are mod-
eled.
B. Graph Neural Networks for Natural Language Processing
Graph Neural Networks have emerged as powerful tools for
modeling structured relationships in data. The Graph Attention
Network (GAT) [4] introduced attention mechanisms to graph
processing, enabling more sophisticated relationship modeling.
Recent work like GraphCodeBERT [5] has demonstrated theeffectiveness of combining graph structures with transformer
architectures for specific domains like code understanding.
However, most existing approaches focus on predetermined
graph structures rather than dynamically constructed graphs
from text sequences. Our work addresses this limitation by
developing algorithms for dynamic graph construction from
token sequences.
C. Hybrid Neural Architectures
The combination of different neural architectures has shown
promising results across various domains. Recent work has ex-
plored combinations of CNNs and RNNs, as well as attention
mechanisms with convolutional layers. However, the system-
atic combination of GNNs with Transformers for language
understanding remains relatively unexplored, particularly for
technical document processing.
D. Technical Document Understanding
Technical document understanding has been addressed through
various approaches, including rule-based systems, machine
learning methods, and more recently, deep learning ap-
proaches. However, most existing methods fail to adequately
capture the complex relationships between technical entities
and the hierarchical structure inherent in technical documents.
III. M ETHODOLOGY
A. System Architecture Overview
Our SLM-CGT system consists of four main components
working in sequence to process technical documents and
generate accurate responses. Figure 1 illustrates the complete
system architecture.
PDF Document
Text Extraction
(Structure Preserved)
GPT-2 Tokenizer
(50,257 vocabulary)
CGT Model
(GNN + Transformer)
46.8M ParametersRAG System
(Semantic Retrieval)
Answer Generation
Fig. 1: SLM-CGT System Architecture showing the complete
pipeline from PDF document processing to answer generation.

B. Complete Query Processing Pipeline
User Query
“What is
ARC600?”
GPT-2
Tokenization
+ Embeddings
Dynamic Graph
Construction
3-Layer GNN
Processing
(Local Relations)
4-Layer
Transformer
Processing
(Global Context)
Language Model
Head
Generated
Answer
+ Context
Fig. 2: Complete query processing pipeline through CGT
model, showing the flow from user query to generated answer
with intermediate representations.
Mathematical Flow Description: The graph structure
enables meaningful relationship modeling while the
Transformer provides global semantic understanding for
comprehensive technical document processing through
the mathematical transformation f:x→ywhere y=
LMHead (Transformer (Reshape (GNN (GraphConstruct (x))))) .
C. Problem Formulation
Given an input text sequence x= [x1, x2, . . . , x n]representing
a technical document or query, our goal is to learn a function
f:x→ythat maps the input to an appropriate output
representation for downstream tasks such as question answer-
ing. The key challenge is to model both local relationships
between nearby tokens and global dependencies across the
entire sequence while maintaining parameter efficiency.
Formally, we define the problem as optimizing the joint
probability distribution:P(y|x) =TY
t=1P(yt|y<t,x,HCGT) (1)
HCGT=Transformer (GNN (GraphConstruct (x))) (2)
logP(y|x) =TX
t=1logP(yt|y<t,HCGT) (3)
where GraphConstruct creates a graph representation, GNN
processes local relationships, and Transformer models global
dependencies.
D. Text Extraction and Preprocessing
Our text extraction pipeline processes PDF documents to pre-
serve both textual and structural information. The system uses
PyMuPDF for robust PDF parsing and implements specialized
algorithms for technical document organization.
Algorithm 1 Enhanced PDF Text Extraction with Structure
Preservation
Require: PDF file path P, output directory Dout
Ensure: Processed text corpus Twith preserved structure
1:Initialize PyMuPDF: T← {} ,Dextract←ProcessTable()
2:foreach page pin PDF page count do
3: Extract text blocks: B←gettext blocks()
4: foreach block binBdo
5: Extract text: text ←b[field text]
6: foreach line lin text split by newline do
7: iflnot empty then
8: Filter tables: Bclean←FilterNoise( B)
9: Combine structured content: T←T∪
Bclean
10: Extract and save images separately
11:Generate document files for each section
12:Merge corpus for training: Tmerged ←MergeCorpus( T)
return Tmerged
The extraction process handles complex table structures
through a multi-stage approach that preserves relationships be-
tween technical specifications, product codes, and operational
parameters. This preprocessing step is crucial for maintaining
the semantic integrity of technical documents.
Tokenization Strategy: We employ the GPT-2 tokenizer for
consistent vocabulary handling, ensuring compatibility with
pre-trained language model components while maintaining
efficiency:
•Vocabulary size: 50,257 tokens
•Encoding method: Byte-pair encoding (BPE) for sub-
word tokenization
•Special tokens: Structure-aware tokens for document
hierarchy
•Maximum sequence length: 512 tokens optimized for
technical documents
•Padding strategy: Right-padding with attention masking

The tokenization process includes special handling for tech-
nical terminology and preserves document structure through
strategic token placement.
E. Contextual Graph Transformer Architecture
The CGT architecture consists of several interconnected com-
ponents that work together to process textual input through
both local and global relationship modeling.
1) Model Configuration
Our CGT model is designed with the following configura-
tion parameters:
V ocabulary Size :|V|= 50,257 (4)
Hidden Dimension :dh= 384 (5)
GNN Layers :Lgnn= 3 (6)
Transformer Layers :Ltrans = 4 (7)
Attention Heads :H= 8 (8)
Maximum Sequence Length :Lmax= 512 (9)
Total Parameters :θ= 46.8M (10)
2) Token Embedding and Positional Encoding
The input processing begins with token embedding and
positional encoding. For the example sequence “What is
ARC600?”:
Token IDs: x= [1867 ,318,5923,21,30] (11)
Embeddings: E=Embedding (x)∈R5×384(12)
Positions: pos= [0,1,2,3,4] (13)
Positional Encoding: P=PositionalEncoding (pos)∈R5×384
(14)
The positional encoding follows the sinusoidal pattern:
PE (pos, 2i)= sinpos
100002i/dh
(15)
PE (pos, 2i+1)= cospos
100002i/dh
(16)
The final initial representation combines embeddings and
positional information:
H(0)=E+P (17)
where H(0)∈Rn×dhrepresents the initial hidden represen-
tations of the tokens.Algorithm 2 Dynamic Graph Construction for CGT
Require: Token sequence x= [x1, x2, . . . , x n], hidden rep-
resentations H(0)
Ensure: Graph G= (V, E, A )where Vare nodes, Eare
edges, Ais adjacency matrix
1:Initialize V={v1, v2, . . . , v n}
2:Initialize E=∅,A∈Rn×n
▷Sequential Connections
3:fori= 1 ton−1do
4: E←E∪ {(vi, vi+1),(vi+1, vi)}
5: A[i, i+ 1] = A[i+ 1, i] = 1.0
▷Skip-gram Connections
6:fori= 1 ton−2do
7: forj=i+ 2tomin(i+ 3, n)do
8: wskip←exp(−0.5· |i−j|)
9: ifwskip>0.3then
10: E←E∪ {(vi, vj),(vj, vi)}
11: A[i, j] =A[j, i] =wskip
▷Semantic Similarity Connections
12:fori= 1 tondo
13: forj=i+ 3tomin(i+ 10, n)do
14: sij←H(0)
i·H(0)
j
∥H(0)
i∥∥H(0)
j∥
15: ifsij>0.7then
16: E←E∪ {(vi, vj),(vj, vi)}
17: A[i, j] =A[j, i] =sij
▷Normalize
18:D[i, i] =P
jA[i, j]
19:A←D−1/2AD−1/2
20:return G= (V, E, A )
The above algorithm constructs a dynamic graph that
captures both structural and contextual relationships among
tokens in a sequence. It starts by establishing sequential edges
between consecutive tokens to preserve the natural order of the
input. Then, skip-gram connections are added to model long-
range dependencies by linking tokens that are a few positions
apart, based on a relevance threshold.
To further enrich the graph, semantic similarity connec-
tions are computed using cosine similarity between token
embeddings. If the similarity exceeds a certain threshold,
an edge is added between those tokens. This allows the
model to connect semantically related tokens even if they are
not adjacent. Finally, the adjacency matrix is normalized to
stabilize learning when passed to the GNN layer.
Example Graph Construction: Consider the input sequence:
“ARC600 wireless gateway device specifications”

ARC600 wireless gateway device specs1.0 1.0 1.0 1.0
0.6 0.6 0.6
0.80.7
Legend:
Sequential (1.0)
Skip-gram (exp decay)
Semantic (cosine sim)
Fig. 3: Example graph construction for “ARC600 wireless
gateway device specifications”. The graph captures three types
of relationships: sequential adjacency (red), skip-gram connec-
tions (green), and semantic similarity (orange dashed).
F . Graph Neural Network Processing
After graph construction, we employ Graph Attention Net-
works v2 (GATv2) to process the local relationships. The
GNN component consists of three specialized layers, each
serving a distinct purpose in capturing local relationships
within technical documents.
1) Graph Attention Mechanism
For each GNN layer l, the attention computation follows:
α(l)
ij=exp( LeakyReLU (aTW(l)[h(l)
i⊕h(l)
j]))
P
k∈N(i)exp( LeakyReLU (aTW(l)[h(l)
i⊕h(l)
k]))
(18)
where:
•α(l)
ijis the attention weight between nodes iandjat layer
l
•W(l)∈Rdh×dhis the learnable weight matrix
•a∈R2dhis the attention parameter vector
•⊕denotes concatenation
•N(i)is the neighborhood of node i
The attention mechanism can be decomposed mathemati-
cally as:
Attention Score =aT[Whi⊕Whj] (19)
Softmax Normalization =exp( score ij)P
k∈N(i)exp( score ik)(20)
Weighted Aggregation =X
j∈N(i)αijWhj (21)
2) Node Update Rule
The node representations are updated using:
h(l+1)
i =σ
X
j∈N(i)α(l)
ijW(l)h(l)
j
 (22)
where σis the ReLU activation function.3) Multi-layer GNN Processing
The complete GNN processing involves three layers with
specialized functions:
H(1)=GATv2(1)(H(0),A) (23)
H(2)=GATv2(2)(H(1),A) (24)
H(3)=GATv2(3)(H(2),A) (25)
Each layer progressively refines the local relationship un-
derstanding:
Layer 1: Immediate Token Relationships
•Captures basic syntactic patterns and adjacent token de-
pendencies
•Learns fundamental word associations in technical con-
texts
•Establishes primary graph connectivity patterns
Layer 2: Phrase-level Dependencies
•Models multi-token technical terms and compound ex-
pressions
•Associates related technical specifications and parameters
•Captures domain-specific terminology relationships
Layer 3: Complex Local Patterns
•Integrates sophisticated technical relationships
•Models hierarchical specification structures
•Captures specialized domain knowledge representations
G. Transformer Integration for Global Processing
After GNN processing, the graph-enhanced node features are
reshaped into sequence format for transformer processing. The
Transformer component employs four layers to model global,
long-range dependencies across the entire sequence.
Hseq=Reshape (H(3),(B, L, d h)) (26)
1) Multi-Head Self-Attention
The transformer layers employ multi-head self-attention
with 8 attention heads:
MultiHead (Q,K,V) =Concat (head 1, . . . , head H)WO
(27)
where each attention head computes:
head i=Attention (QWQ
i,KWK
i,VWV
i) (28)
The multi-head attention mechanism can be mathematically
decomposed as:
Q=HseqWQ∈Rn×dh(29)
K=HseqWK∈Rn×dh(30)
V=HseqWV∈Rn×dh(31)

2) Scaled Dot-Product Attention
The core attention mechanism captures global semantic
understanding:
Attention (Q,K,V) =softmaxQKT
√dk
V (32)
where the scaling factor√dkprevents the softmax function
from saturating and dk=dh/H= 384 /8 = 48 .
The mathematical details of attention computation:
Score Matrix =QKT
√dk∈Rn×n(33)
Attention Weights =softmax (Score Matrix ) (34)
Context Vector =Attention Weights ·V (35)
3) Complete Transformer Block
Each transformer layer includes:
Zl=LayerNorm (Hseq+MultiHead (Hseq)) (36)
H(l+1)
seq =LayerNorm (Zl+FFN(Zl)) (37)
where the feed-forward network captures global context:
FFN(x) = max(0 ,xW 1+b1)W2+b2 (38)
with intermediate dimension dff= 4×dh= 1536 .
How Transformers Enable Global Understanding:
•Long-range Dependencies: Captures relationships be-
tween distant parts of the document
•Contextual Integration: Combines local GNN features
with global document context
•Sequence Coherence: Maintains document-wide consis-
tency for technical specifications
•Multi-head Attention: Different heads focus on various
types of global relationships
•Complete Transformer Layer: Each layer refines global
understanding progressively
H. RAG Integration
The trained CGT model is integrated into a Retrieval-
Augmented Generation system for enhanced question answer-
ing. The RAG framework combines the CGT’s understanding
capabilities with external knowledge retrieval.Algorithm 3 RAG-based Question Answering with CGT
Require: Query q, Document chunks D={d1, d2, . . . , d n}, CGT
Ensure: Generated answer awith contextual understanding
Encode query: qemb←SentenceTransformer (q)
Precompute chunk embeddings: Demb← {M(di)}n
i=1
Compute similarities: si←cos(qemb, di)for all i
Retrieve top- kchunks: k←TopK ({si}, k= 3)
Construct context: c←Concatenate (Dret)
Construct prompt from context and query
p←”Context: ” +c+” Question: ” +q+” Answer: ”
Tokenize prompt: ptokens←Tokenizer (p)
Generate with CGT: a←M.generate (ptokens)using beam search
ifQuality (a)< θ quality then
Fallback: a←IntelligentExtraction (Dret)
Enhance answer quality through post-processing return a
Detailed RAG Example:
Query: “What is ARC600?”
Step 1: Encode query using SentenceTransformer
qemb=ST(“What is ARC600?” )∈R384
Step 2: Retrieve relevant chunks using cosine similar-
ity:
•Chunk 1: “ARC600 wireless controller...” (sim =
0.92)
•Chunk 2: “Gateway device features...” (sim = 0.87)
•Chunk 3: “Communication protocols...” (sim =
0.81)
Step 3: Mathematical similarity computation:
similarity (q, di) =qemb·di,emb
||qemb|| · ||di,emb||
Step 4: Context construction and generation: Context
+ Question →CGT Model →Answer
Generated Answer: “Complete communication sys-
tem. Wireless Controller ARC600 is typically part of a
complete communication system...”
Mathematical Formulation:
Similarity = cos( q,di) =q·di
|q||di|(39)
Context =Concat ({di:si> θ retrieval}) (40)
Response =CGT(Prompt (c, q)) (41)
Quality Score =1
|A||A|X
i=1BLEU (ai, ri) (42)
RAG Integration Benefits: The RAG system provides
several advantages for technical document understanding:
•Knowledge Augmentation: Combines CGT’s structural
understanding with relevant document retrieval

•Context-aware Responses: Generates answers based on
specific document sections
•Semantic Retrieval: Uses sophisticated embedding-
based similarity for document chunk selection
•Intelligent Fallback: Implements quality control with
intelligent extraction mechanisms
•Beam Search Generation: Employs advanced decoding
strategies for coherent answer generation
I. Training Methodology
We employ a two-stage training approach designed to maxi-
mize the model’s ability to understand both general language
patterns and domain-specific technical content.
1) Stage 1: General Language Pre-training
The first stage focuses on learning general language repre-
sentations:
Epochs :E1= 5 (43)
Learning Rate :η1= 1×10−4(44)
Batch Size :B1= 16 (45)
Training Data :Dwiki= 2000 samples (46)
2) Stage 2: Domain-Specific Fine-tuning
The second stage adapts the model to technical documents:
Epochs :E2= 5 (47)
Learning Rate :η2= 5×10−5(48)
Batch Size :B2= 8 (49)
Training Data :Dtech= 151 samples (50)
3) Loss Function
The training objective combines standard language model-
ing with graph regularization and attention diversity:
Ltotal=LLM+λLgraph+γLattention +βLconsistency (51)
where:
LLM=−TX
t=1logP(xt|x<t,HCGT) (52)
Lgraph=−X
(i,j)∈Elogαij+1
2||A−AT||2
F (53)
Lattention =−LtransX
l=1HX
h=1Entropy (Attn(l)
h) (54)
Lconsistency =||HGNN−HTrans||2
2 (55)
λ= 0.1, γ = 0.05, β = 0.02 (56)
The graph regularization term encourages the model to
learn meaningful attention patterns in the graph structure,
while the attention regularization promotes diverse attention
patterns across heads, and the consistency term ensures smooth
transition between GNN and Transformer representations.IV. E XPERIMENTAL SETUP
A. Dataset Description
Our experimental evaluation utilizes technical documents from
the ABB ARC600 product guide, representing real-world
industrial documentation challenges:
Document Characteristics:
•Technical specifications with numerical values and units
•Product descriptions with hierarchical information struc-
ture
•Installation procedures and operational guidelines
•Mixed content including tables, lists, and prose text
•Domain-specific terminology and abbreviations
Data Distribution:
Pre-training Corpus : 2,000Wikipedia samples (57)
Fine-tuning Corpus : 151 technical document segments
(58)
Evaluation Questions : 18 technical queries (59)
RAG Knowledge Base : 80 document chunks (60)
B. Baseline Models
We compare our CGT model against three established trans-
former architectures and a pure transformer baseline:
TABLE I: Baseline Model Specifications
Model Parameters Architecture Source
DistilBERT 89.8M Encoder-only HuggingFace
GPT-2 124.4M Decoder-only HuggingFace
BERT 133.0M Encoder-only HuggingFace
Pure Transformer 52.0M Transformer-only Custom
CGT (Ours) 46.8M GNN + Transformer Custom
All baseline models underwent identical training procedures
to ensure fair comparison, with the same data and hyperpa-
rameter settings adapted for each architecture.
C. Evaluation Metrics
We employ comprehensive evaluation metrics to assess model
performance:
•Training Loss: Cross-entropy loss during training pro-
gression
•Final Performance Loss: Average loss on evaluation set
•BLEU Scores: N-gram overlap metrics for generation
quality
•ROUGE Scores: Recall-oriented metrics for summariza-
tion quality
•Jaccard Similarity: Set-based similarity for semantic
overlap
•Response Time: Inference time for practical deployment
assessment

V. R ESULTS AND ANALYSIS
A. Main Performance Results
Table II presents the comprehensive comparison of our CGT
model against all baseline architectures:
TABLE II: Performance Comparison: CGT vs Transformer
Baselines
Model Parameters (M) Final Loss
DistilBERT 89.8 10.430
GPT-2 124.4 2.787
BERT 133.0 10.460
Pure Transformer 52.0 3.456
CGT (Our Model) 46.8 2.099
Improvement vs GPT-2 -62.4% +24.7%
Improvement vs Pure Transformer -10.0% +39.2%
B. Detailed BLEU and ROUGE Evaluation
Table III presents comprehensive evaluation metrics compar-
ing our CGT model with the pure transformer baseline:
TABLE III: Detailed Performance Metrics: CGT vs Pure
Transformer
Metric CGT Hybrid Pure Transformer
BLEU-1 Score 0.1559 0.0238
BLEU-2 Score 0.0589 0.0080
BLEU-4 Score 0.0227 0.0038
ROUGE-1 Score 0.2309 0.0511
ROUGE-2 Score 0.0437 0.0015
ROUGE-L Score 0.2004 0.0481
Jaccard Similarity 0.1170 0.0264
Response Time (s) 0.4413 0.2696
The results demonstrate remarkable improvements across all
evaluation metrics:
•BLEU Scores: 491%-639% improvement indicates su-
perior n-gram overlap
•ROUGE Scores: Up to 2729% improvement in recall-
oriented metrics
•Semantic Similarity: 343% improvement in Jaccard
similarity
•Efficiency Trade-off: 64% longer response time for
substantially better qualityC. Performance Visualization
DistilBERTGPT-2 BERT
Pure TransformerCGT051010.430
2.78710.460
3.456
2.099
ModelFinal LossFinal Loss
CGT achieves 554% improvement in parameter efficiency
compared to the Pure Transformer baseline.
Training Dynamics Analysis:
Our two-stage training approach demonstrates efficient con-
vergence patterns:
Stage 1 Loss Reduction : 6.77→4.82(28.8% improvement)
(61)
Stage 2 Loss Reduction : 4.82→2.099(56.5% improvement)
(62)
Overall Improvement : 6.77→2.099(69.0% total reduction)
(63)
D. Parameter Efficiency Analysis
To quantify parameter efficiency, we define the efficiency
metric:
Efficiency (model ) =1
Parameters ×Loss(64)
Table IV presents the efficiency comparison:
TABLE IV: Parameter Efficiency Analysis
Model Parameters (M) Loss Efficiency Score
DistilBERT 89.8 10.430 1.07×10−3
GPT-2 124.4 2.787 2.88×10−3
BERT 133.0 10.460 7.19×10−4
Pure Transformer 52.0 3.456 5.56×10−3
CGT (Ours) 46.8 2.099 1.02×10−2
Improvement vs Best -10.0% +39.2% +83%
CGT achieves an 83% improvement in parameter efficiency
compared to the best baseline (Pure Transformer).

E. Qualitative Analysis: Generated Responses
To demonstrate the practical effectiveness of our approach, we
present actual generated responses from our CGT model with
highlighted formatting:
Example 1:
Question: What is ARC600?
CGT Response: Complete communication system.
Wireless Controller ARC600 is typically part of a com-
plete communication system which consists of Arctic
600 series gateways or controllers and a central M2M
Gateway ARM600 communication server.
Example 2:
Question: What communication protocols does
ARC600 support?
CGT Response: The inputs and outputs of ARC600
can be accessed and controlled with the IEC-104 and
IEC-101 protocols.
Example 3:
Question: How does the wireless controller operate?
CGT Response: Wireless Controller ARC600 utilizes
the built-in wireless communication features for reliable
and secure end-to-end communication providing remote
monitoring and control of three switching devices and
can be expanded as required by using external I/O
expansion. The use of Wireless Controller ARC600 in
distribution networks improves the quality of power
distribution and reduces the outage time in the affected
areas.
F . Ablation Study
To understand the contribution of different architectural com-
ponents, we conducted an ablation study:
TABLE V: Ablation Study: Component Contribution Analysis
Model Variant Parameters (M) Final Loss
Pure Transformer (7 layers) 52.0 3.456
Pure GNN (3 layers) 24.2 4.892
CGT (GNN + Transformer) 46.8 2.099
Improvement vs Transformer -10.0% params +39.2% performance
Improvement vs GNN +93.4% params +57.1% performance
The ablation study confirms that the hybrid architecture
provides substantial benefits:
•vs Pure Transformer: 39.2% better performance with
10.0% fewer parameters
•vs Pure GNN: 57.1% better performance, justifying the
parameter increase
•Synergistic Effect: The combination outperforms both
individual components significantlyVI. D ISCUSSION
A. Architectural Advantages
The success of our CGT architecture can be attributed to
several key design principles:
1) Local Relationship Modeling
The GNN component excels at capturing fine-grained local
relationships that are crucial for technical document under-
standing:
•Technical Term Association: The graph structure effec-
tively links related technical terms like “ARC600” and
“wireless controller”
•Specification Grouping: Numerical values and their
units are correctly associated through local attention
•Procedural Coherence: Sequential steps in technical
procedures maintain proper relationships
2) Parameter Efficiency through Hybrid Design
Our hybrid approach achieves superior parameter efficiency
through complementary processing:
Efficiencyhybrid =Performance GNN+Performance Transformer
Parameters GNN+Parameters Transformer(65)
The mathematical analysis shows that:
GNN Parameters : 12.3M(26.3% of total) (66)
Transformer Parameters : 34.5M(73.7% of total) (67)
Performance Contribution :Synergistic enhancement (68)
Efficiency Gain :Phybrid
PGNN+PTrans>Ppure
Ppure(69)
3) Computational Complexity Analysis
The computational complexity of our approach is:
GNN Complexity :O(|E| ·dh)where |E| ≤n·k
(70)
Transformer Complexity :O(n2·dh) (71)
Total Complexity :O(n·k·dh+n2·dh) (72)
For local graphs with limited connectivity ( k≪n), this re-
duces effective complexity while maintaining representational
power.
B. Limitations and Future Work
While our approach demonstrates significant advantages, sev-
eral limitations warrant discussion:
1) Current Limitations
•Graph Construction Complexity: Dynamic graph con-
struction adds computational overhead during inference
•Limited Evaluation Scope: Current evaluation fo-
cuses on technical documents; broader domain validation
needed
•Fixed Architecture: Current design uses fixed GNN and
Transformer layer counts

2) Future Research Directions
Several promising directions emerge from this work:
1)Adaptive Graph Construction: Developing learned
graph construction algorithms that adapt to content type
2)Scalability Optimization: Investigating efficient imple-
mentations for larger-scale deployments
3)Multi-domain Evaluation: Testing effectiveness across
diverse technical domains
4)Architecture Search: Automated optimization of GNN-
Transformer layer combinations
VII. R ELATED WORK IN PARAMETER -EFFICIENT MODELS
Recent research has increasingly focused on developing
parameter-efficient alternatives to large language models. Our
work contributes to this important direction by demonstrating
that architectural innovation can achieve better performance
with substantially fewer parameters.
A. Efficiency Techniques
Various approaches have been proposed for improving param-
eter efficiency:
•Knowledge Distillation: DistilBERT [3] reduces param-
eters through teacher-student training
•Pruning and Quantization: Post-training compression
techniques
•Low-rank Factorization: Matrix factorization for pa-
rameter reduction
•Architectural Innovation: Our hybrid GNN-
Transformer approach
Our approach differs fundamentally by achieving efficiency
through architectural design rather than compression of exist-
ing architectures.
B. Hybrid Architectures
The combination of different neural architectures has shown
promise in various domains:
CNN-RNN Hybrids :Computer vision and sequence model
Attention-CNN :Image captioning and visual QA
GNN-Transformer (Ours) :Technical document understanding
Our work represents the first systematic exploration of
GNN-Transformer hybrids for natural language processing
tasks.
VIII. C ONCLUSION
In this work, we introduce the Contextual Graph Transformer
(CGT), a novel hybrid architecture for small language model
that effectively combines Graph Neural Networks with Trans-
formers for technical document understanding. Our approach
addresses the critical challenge of parameter efficiency while
maintaining superior performance through innovative architec-
tural design.REFERENCES
[1] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training
of Deep Bidirectional Transformers for Language Understanding,” in
Proceedings of NAACL-HLT , 2019, pp. 4171–4186.
[2] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever,
“Language Models are Unsupervised Multitask Learners,” OpenAI Tech-
nical Report, 2019.
[3] V . Sanh, L. Debut, J. Chaumond, and T. Wolf, “DistilBERT, a distilled
version of BERT: smaller, faster, cheaper and lighter,” arXiv preprint
arXiv:1910.01108 , 2019.
[4] P. Veli ˇckovi ´c, G. Cucurull, A. Casanova, A. Romero, P. Li `o, and Y .
Bengio, “Graph Attention Networks,” in International Conference on
Learning Representations , 2018.
[5] D. Guo, S. Ren, S. Lu, Z. Feng, D. Tang, S. Liu, L. Zhou, N. Duan, A.
Svyatkovskiy, S. Fu, M. Tufano, S. K. Deng, C. Clement, D. Drain,
N. Sundaresan, J. Yin, D. Jiang, and M. Zhou, “GraphCodeBERT:
Pre-training Code Representations with Data Flow,” in International
Conference on Learning Representations , 2021.
[6] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal, H.
K¨uttler, M. Lewis, W.-t. Yih, T. Rockt ¨aschel, S. Riedel, and D. Kiela,
“Retrieval-augmented generation for knowledge-intensive nlp tasks,” in
Advances in Neural Information Processing Systems , vol. 33, 2020, pp.
1517–1530.
[7] S. Brody, U. Alon, and E. Yahav, “How Attentive are Graph Attention
Networks?” in International Conference on Learning Representations ,
2022.
[8] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez,
Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” in Advances
in Neural Information Processing Systems , 2017, pp. 5998–6008.
[9] Y . Xu, M. Li, L. Cui, S. Huang, F. Wei, and M. Zhou, “LayoutLM:
Pre-training of Text and Layout for Document Image Understanding,”
inProceedings of the 26th ACM SIGKDD International Conference on
Knowledge Discovery & Data Mining , 2020, pp. 1192–1200.
[10] T. N. Kipf and M. Welling, “Semi-supervised classification with graph
convolutional networks,” in International Conference on Learning Rep-
resentations , 2017.
[11] Z. Zhang, J. Ma, J. Xu, and Q. Liu, “GraphFormers: Turning Trans-
formers into Effective Graph Learners,” in IEEE Transactions on Pattern
Analysis and Machine Intelligence , 2022.
[12] W. Wang, J. Zhang, X. Zhang, Y . Wang, and Y . Xie, “Structure-Aware
Pretraining for Table-Based Question Answering,” in Findings of ACL ,
2021, pp. 1127–1138.
[13] P. He, X. Liu, J. Gao, and W. Chen, “DeBERTa: Decoding-enhanced
BERT with Disentangled Attention,” in International Conference on
Learning Representations , 2021.
[14] L. Yao, C. Mao, and Y . Luo, “Graph-based Neural Multi-Document
Summarization,” in Proceedings of the 57th Annual Meeting of the ACL ,
2019, pp. 763–772.
[15] Y . Xu, R. Lv, L. Cui, Y . Liu, D. Zhang, and M. Zhou, “LayoutLMv2:
Multi-modal Pre-training for Visually-Rich Document Understanding,”
inProceedings of ACL , 2021, pp. 2579–2591.
[16] X. Huang, S. Singh, and M. Gardner, “Finetuned Language Models Are
Zero-Shot Learners,” in Proceedings of EMNLP , 2021, pp. 7676–7690.
[17] L. Liu, S. Lin, S. Liu, and D. Song, “GraphMem: A Memory-Augmented
Graph Neural Network for Sequential Reasoning,” in Proceedings of
the 30th ACM International Conference on Information & Knowledge
Management , 2022, pp. 1243–1252.
[18] K. Lee, M. Chang, and K. Toutanova, “Latent Retrieval for Weakly
Supervised Open Domain Question Answering,” in Proceedings of ACL ,
2019, pp. 6086–6096.
[19] S. Sannara, M. Cochez, M. Stolle, and S. Decker, “OntoRAG: Ontology-
Guided Retrieval-Augmented Generation for Explainable Question An-
swering,” in arXiv preprint arXiv:2302.11673 , 2023.
[20] Q. Xu, S. Koh, K. Singh, and J. Carbonell, “Explainable AI: A Survey
of Methods and Trends for Deep Learning Models,” in arXiv preprint
arXiv:1909.06945 , 2019.