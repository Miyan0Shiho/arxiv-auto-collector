# Ghost Vectors: Soft-Deleted Embeddings Remain Reconstructible in HNSW Vector Databases

**Authors**: Chandranil Chakraborttii, Jackeline García Alvarado, Sitora Abdulofizova, Shivanshu Dwivedi

**Published**: 2026-06-16 21:15:49

**PDF URL**: [https://arxiv.org/pdf/2606.18497v1](https://arxiv.org/pdf/2606.18497v1)

## Abstract
Retrieval-augmented generation (RAG) allows large language models to access external and private corpora for factual, domain-specific responses. Modern RAG pipelines use hierarchical navigable small world (HNSW) vector databases for efficient similarity search. When a user requests data deletion, the systems typically only mark the record as deleted, leaving the embedding on disk physically unchanged. This soft-delete operation raises compliance concerns under data-erasure and retention requirements such as GDPR Article 17 and HIPAA. Analysis on three HNSW implementations confirms that deleted vectors remain physically recoverable by accessing the raw index files at the storage layer, bypassing API access. Using the Vec2Text inversion model without domain-specific fine-tuning, we show this vulnerability on multiple real-world datasets and data modalities. On Wikipedia biographical living persons dataset (BLP), we successfully recover 25.5% of exact person names and 46.4% of geographic locations (ROUGE-L 0.185). Recovery reaches 100% for both patient age and gender markers (ROUGE-L 0.290) on highly structured, sensitive data (NIH Synthea dataset). On soft-deleted image embeddings, we show 100% tissue classification on histopathology patches (p=1.02e-07) and top-1 identity recovery reaches 99% on facial embeddings (p<0.01). This work introduces Epoch Key Rotation, which encrypts vectors and discards the key upon deletion. Epoch key rotation reduces observed PII recovery to 0% and completes in 2.5 ms for 500 deleted vectors (approximately 0.005 ms/record). Additionally, it generates an ECDSA-signed cryptographic proof as an auditable record of the deletion event.

## Full Text


<!-- PDF content starts -->

Ghost Vectors: Soft-Deleted Embeddings Remain
Reconstructible in HNSW Vector Databases
Chandranil Chakraborttii, Jackeline Garc ´ıa Alvarado, Sitora Abdulofizova, and Shivanshu Dwivedi
Trinity College, USA
Abstract—Retrieval-augmented generation (RAG) allows large
language models to access external and private corpora for
factual, domain-specific responses. Modern RAG pipelines use
hierarchical navigable small world (HNSW) vector databases
for efficient similarity search. When a user requests data dele-
tion, the systems typically only mark the record as deleted,
leaving the embedding on disk physically unchanged. This soft-
delete operation raises compliance concerns under data-erasure
and retention requirements such as GDPR Article 17 and
HIPAA. Analysis on three HNSW implementations confirms
that deleted vectors remain physically recoverable by accessing
the raw index files at the storage layer, bypassing API access.
Using the Vec2Text inversion model without domain-specific
fine-tuning, we show this vulnerability on multiple real-world
datasets and data modalities. On Wikipedia biographical living
persons dataset (BLP), we successfully recover 25.5% of exact
person names and 46.4% of geographic locations (ROUGE-L
0.185±0.062). Recovery reaches 100% for both patient age and
gender markers (ROUGE-L0.290±0.033) on highly structured,
sensitive data (NIH Synthea dataset). On soft-deleted image
embeddings, we show 100% tissue classification on histopathology
patches (p= 1.02×10−7) and top-1 identity recovery reaches
99% on facial embeddings (p <10−12). This work introduces
Epoch Key Rotation, which encrypts vectors and discards the
key upon deletion. Epoch key rotation reduces observed PII
recovery to 0% and completes in 2.5 ms for 500 deleted vectors
(approximately 0.005 ms/record). Additionally, it generates an
ECDSA-signed cryptographic proof (π) as an auditable record
of the deletion event.
Index Terms—Vector databases, Soft-delete, HNSW, Privacy,
Cryptographic Erasure
I. INTRODUCTION
Retrieval-Augmented Generation (RAG) technologies are
being rapidly deployed across highly regulated domains (such
as finance or healthcare) using HNSW-based vector databases
in the backend to store and retrieve sensitive data. Regu-
lations such as Article 17 of the General Data Protection
Regulation (GDPR) allow users to request immediate and full
removal of their data [1], [2] from client systems. However,
these implementations raise urgent regulatory concerns. Vector
databases typically implement deletion as a ‘soft deletion’
at the metadata level by design to preserve performance.
Under data minimization obligations [3], user data must not
be retained beyond its intended purpose [3], [4] — soft-delete
is potentially inconsistent with these regulations indefinitely.
Treating no response from an API as confirmation of deletion
assumes that data has been removed, even though it may
still physically exist. Query-side embedding transformations
have been introduced to reduce the risk of inversion attacks;however, it applies only to active queries and does not provide
any protection once a vector enters secondary storage.
Our threat model operates at the secondary storage layer,
where active query defenses are no longer effective. We as-
sume an adversary has access to the raw storage data, bypass-
ing all API restrictions. Since soft-delete only modifies API
metadata, the vectors remain physically present in the HNSW
index (invisible to queries, but intact on disk). The adversary
reads the HNSW binary index (e.g., ChromaDB’s index.bin
file), extracts soft-deleted vectors usingget_items(), and
feeds them into a pre-trained Vec2Text corrector [5] without
any fine-tuning. The recovered vectors are inverted back
into plaintext, recovering sensitive information at higher rates
than accepted GDPR and HIPAA thresholds. We show the
full attack pipeline in Figure 1 and describe it in detail in
Section IV. This work introducesEpoch Key Rotationas
a storage-layer solution to this vulnerability. The approach
encrypts vectors with AES-256-CTR and discards the key,
reducing PII recovery to 0% in our experiments. A signed
cryptographic proof is also generated, providing an auditable
record of the deletion event. This paper makes the following
five contributions:
(1) Weempirically showthat soft-deleted HNSW vectors
remain physically present, even after deletion requests are
confirmed via API on three independent systems (ChromaDB,
FAISS, and Weaviate).
(2) On the Wikipedia BLP dataset, 25.5% of person names
and 46.4% of locations were recovered (n=98, ROUGE-L =
0.185 ± 0.062). On the NIH-standard Synthea dataset, recovery
reached 100% for age and gender markers under corrector-
aligned clinical text. In the synthetic clinical dataset, patient
surname recovery reached 49.9% (CI: 46.8–53.0), indicating
significant re-identification risk in our evaluation. On MIMIC-
III, reconstruction remained low (ROUGE-L<0.010), sug-
gesting that text reconstruction quality depends on corrector
alignment. However, physical persistence is universal, repre-
senting a latent threat when stronger domain-adapted inversion
models become available.
(3)The threat is indefinite, since a backup snapshot can be
used for attack with identical reconstruction quality (ROUGE-
L=0.207). Vec2Text reconstructions transfer across three struc-
turally different surrogate models (ROUGE-L≥0.19, cosine
similarity 0.86–0.90), confirming the attackers does not require
knowledge of the victim’s embedding model. The threat also
shows low variance across the tested scales since ROUGE-L
variance across 1K, 5K, 10K, 50K, 100K remains at 0.016.arXiv:2606.18497v1  [cs.CR]  16 Jun 2026

(4)The threat extends to image embeddings and structural
metadata.Soft-deleted facial images result in 99% top-1 iden-
tity accuracy. Similarly, histopathology patches (PathMNIST,
N= 1000) are recovered at 100% tissue-class accuracy
(p= 1.02×10−7). Additionally, we show that ghost nodes
remain structurally active in the HNSW graph, degrading live
search results and creating a timing side-channel that reduces
query latency by 4.3% under true deletion (Table IX).
(5)Epoch key rotation reduces all PII recovery to 0%
at a rate of 0.005ms/record and generates an ECDSA-signed
cryptographic proof supporting auditable erasure workflows
in regulated settings (e.g, GDPR Art. 17 or HIPAA). For
deployments requiring strict isolation, we later show that
standard SQLite key deletion leaves key data recoverable
from free pages, motivating hardware-isolated key storage
(TPM/TrustZone).
II. BACKGROUND
I. HNSW Vector Indexes and Soft-Deletion:In vector
databases, HNSW graphs are the most common index structure
for approximate nearest neighbor search [6], [7]. HNSW builds
a multi-layer graph where each node represents a vector, edges
connect it to itsMnearest neighbors, and layer-by-layer
traversal takesO(logn)search time. To minimize latency,
systems built on HNSW handle deletion with metadata flags,
instead of erasing data from disk directly. For example, Chro-
maDB callsdelete(ids=[...])to update an SQLite
metadata table, with subsequent queries skipping the flagged
entries [8]. However, the rawindex.binfile is not modified.
FAISSIndexHNSWFlatprovides nodelete()method,
so all vectors accumulate indefinitely [9], [10]. Weaviate [11]
and Pinecone [12] follow the same pattern. Thus, soft-delete
represents a deliberate design trade-off, not a bug in imple-
mentation.
II. Text Embedding Inversion (Vec2Text):Inversion mod-
els such as Vec2Text [5] have been developed to reconstruct
embeddings back into semantically equivalent text. Starting
from a candidate sequence produced by beam search, the
corrector runs T iterative steps, each reducing the embedding
distance between the candidate and the target vector. A pre-
trained corrector (gtr-base) learns to associate token-level
modifications using embedding residuals, with the weights
staying frozen at inference. Since Vec2Text generates plain
text than model-specific artifacts, it can invert embeddings
across different architectures (without knowing the victim’s
exact embedding model). Thegtr-t5-base[13] retriever
used in this work is a pretrained dense retrieval model [14].
Prior work reports high reconstruction accuracy when training
and evaluation are performed on closely matched text distri-
butions. However, they have not been studied on soft-deleted
records inside a live vector database.
III. GDPR Article 17 and Verifiable Erasure:GDPR
Art. 17(1) grants data subjects the right to erasure ‘without
undue delay’ [1]. EDPB Guidelines 05/2019 [2] specify that
erasure must be bothverifiableandirreversible; suppress-
ing records from query results alone does not satisfy thisrequirement. Art. 5(1)(e) additionally limits retaining data
beyond its intended purpose [1]. Current soft-delete designs
may violate this indefinitely, regardless of stated policy. In
these systems, enforcement is treated as a self-certification.
However, from a systems perspective, functional deletion (i.e.,
an API returns no record) and storage-layer erasure are not
equivalent. After a soft-delete command, a controller may treat
the API response as sufficient evidence of deletion without
checking the underlying state. Even with vectors physically
present on the disk, the controller can still truthfully report
that the API has no records for that ID. This paper identifies
these two gaps and proposes a storage-layer solution to address
them.
III. THREATMODEL
A. System and Adversary Model
Threat Model: Three actors participate in our threat model:
(i) aData Subjectthat exercises the right to erasure under
modern data complaince standards; (ii) aData Controller
operating a RAG system on top of an HNSW-based vector
store (Weaviate, FAISS, or ChromaDB) who issues a soft-
delete in response to deletion requests and treats the no API
response as evidence of erasure; and (iii) aBreach Adversary
with read access to raw storage who acts after the deletion has
been recorded.
The Data Controller embeds documents using the
gtr-t5-baseencoder (d= 768) to build an HNSW index
for RAG queries. After soft-delete, the API correctly returns
no records for the removed IDs — but the controller performs
no check at the storage layer, treating API silence as evidence
of deletion.
Adversary Model:The breach adversary hasread access
to the raw storage directory (e.g, insider access, cloud storage
mis-configuration, or backup data access). The adversary has
knowledgeof the embedding model that was used (often
publicly available from system docs). Itsgoalis to reconstruct
the semantic content of deleted data, i.e., records that the
controller has declared erased. The plaintext, decryption keys,
and API endpoints remain inaccessible.
Backup systems keep HNSW binary files regardless of API
state, and entire database directories have been exposed in
prior storage breaches [15]. A malicious insider (such as a
database administrator or a cloud provider employee) can
access raw files without causing API audit logs to be triggered.
Copyingindex.binonce is sufficient. The threat is not
limited by deletion time, as soft-deleted vectors can remain
unchanged in a backup for months after deletion. The attacker
takes advantage of the indefinite window that starts at deletion
and ends only when the storage media is either physically
destroyed or cryptographically overwritten. We empirically
validate the indefinite threat window and scope of this threat
model in Section V-A.
B. Formal Security Analysis
We formalize two main claims: (1) soft-delete leaves recov-
erable semantic information in storage, and (2) epoch key ro-

Vector Database 
(ChromaDB/FAISS) 
HNSW Index API Layer 
(Returns 0 records) 
Raw HNSW 
Binary 
(index.bin) Vec2Text 
Corrector 
(gtr-base) Reconstructed Text 
ROUGE-L = 0.207 
BERTScore =0.858 
PII Recorded: 
49.9% surnames 
73.4% gender 
Vector Database 
(ChromaDB/FAISS) HNSW 
index Epoch Key Rotation 
AES-256-CTR AES Ciphertext 
(float32 noise) 
+
Key discarded Vec2Text Fails 
ROUGE-L = 0.00 
0% PII Recorded 
(Complete Protection) Soft-delete 
(vector marked, 
not removed) 
Storage layer access 
adversary recovers 
PII here 
On deletion: 
encrypt + 
discard key Key
destroyed Inversion 
attempted hnswlib.get_items() 
bypass` ATTACK PIPELINE 
DEFENSE (EPOCH KEY ROTATION) vs.Fig. 1. Ghost vector attack pipeline (top) and epoch key rotation defense (bottom)
tation removes that information under standard cryptographic
assumptions. Both are expressed in terms of the Semantic
Recovery Advantage (SRA). We useDandEto represent a
deletion process and an embedding function respectively. We
define SRA of adversaryAas:
Advsem
A(λ) = Pr [EntityRecovery(A(D(E(x))))≥θ]
whereθ= 0.2is the re-identification risk threshold, consistent
with HIPAA Safe Harbor [16] to denote meaningful semantic
recovery, andxrepresents a plaintext record.
Theorem 1(Soft-Delete Risk).Advsem
A(λ)is empirically non-
negligible under soft-delete (D=SoftMark), due to the
persistence of the original embedding vector in raw storage
(e.g., 0.499 surname recovery in Table IV).
Theorem 2(Epoch Rotation Security).Assuming AES-256-
CTR is used with securely managed and irrecoverably deleted
keys, (D=EpochRotate),Advsem
A(λ)is negligible inλfor
any PPT adversaryA.
Proof.AssumeAachieves non-negligible semantic advantage
over epoch key rotation. We construct a distinguisherBagainst
AES-256-CTR as follows:
The AES-256-CTR challenger sends a challenge ciphertext
c∗toB, which converts it into a float32 embedding vector
and sends it toA.Boutputs 1 if entity recovery fromA(c∗)
exceedsθ; otherwise 0.Ashould not recover meaningful
semantic content with non-negligible probability whenc∗is
truly random. By contradiction, supposeAcould still recover
meaningful semantic content from ciphertext produced under
epoch rotation with non-negligible probability. Because AES-
CTR is widely considered as IND-CPA secure under the
standard PRP assumptions [17]–[19], this resultant gap gives
Ba non-negligible advantage in distinguishing AES-256-CTR
output from random. Therefore, under these assumptions, no
PPTAachieves non-negligible semantic recovery advantage
under epoch rotation.The bound in Theorem 2 does not depend on Vec2Text or
any specific corrector. Future inversion models cannot weaken
it, because the ciphertext carries no semantic structure for
them to recover. This argument depends on the security of
the underlying AES-256 encryption scheme and on correct
key disposal. However, the argument fails if the system leaves
the decryption key on disk. If the decryption key remains
accessible on disk somewhere, every guarantee in Theorem 2
no longer applies. Theorem 2 requires erasing epoch keys
permanently and irrevocably. SQLite’s standardDELETEsim-
ply marks rows for overwrite without zeroing disk sectors,
leaving key material recoverable from free pages. Hardware
isolation (ARM TrustZone, HSM) or strict memory zeroing
before deletion (OPENSSL_cleanse()) in deployments is
needed for full-proof secure key disposal.
IV. GHOSTVECTORATTACKSETUP
A. Threat Environment and Setup
The attack is evaluated across three HNSW implementations
and multiple text and multimodal datasets. Figure 1 shows
the full (attack and defense) pipeline, including storage-layer
extraction, Vec2Text inversion, and epoch key rotation. Vector
extraction runs entirely from Python scripts that read the on-
disk index files directly, never touching the database query
API or any defense layered on top of it. All experiments were
conducted with PyTorch 2.0 on a dedicated node equipped
with AMD Ryzen 7960X CPU with 96GB system RAM, and
a NVIDIA RTX A6000 GPU with 48 GB VRAM.
B. Evaluated Systems
We evaluate ChromaDB (usinghnswlib), FAISS (using
IndexHNSWFlat), and Weaviate (in embedded mode).
For each system, we insert 5,000 embedded records, and
soft-delete 10% of each dataset usingdefault_rng(43).
Eitherindex.reconstruct_n()(for FAISS) or
hnswlib.get_items()(for ChromaDB) are used to
extract vectors directly at the storage layer.

C. Datasets
1) Text Datasets::(1) The baseline dataset used in Section
3 consists of the first 5,000 articles from wikimedia/wikipedia
20231101.en, trimmed to 512 characters and embedded using
gtr-t5-basewith mean pooling and L2 normalization. Re-
construction is evaluated acrossN∈1K,5K,10K,50K,100K,
soft-deleting 10% per scale.
(2) Real-world privacy (PII) leakage is measured on a Bio-
graphical Living Persons (BLP) subset ofN= 98records.
Wikipedia baseline above is used to confirm scale.
(3) Since real-world healthcare data is sensitive, we instead
use 1,000 synthetic clinical records generated with Faker [20]
— each with complaints, diagnoses, and drug bundles. 58.8%
of records received one of five clinical abbreviation transfor-
mations (e.g., ‘Acute myocardial infarction’→‘AMI’, ‘twice
daily’→‘BID’) before the embedding step, simulating real-
world distortions. We also include two clinical stress tests:
1,000 Synthea EHR records with multi-year patient histories
and 100 MIMIC-III Demo [21] intensive-care notes containing
physician shorthand. For the cryptographic-erasure baseline,
we encrypt raw float32 bytes with AES-256-CTR and imme-
diately discard the key.
2) Multimodal Datasets::Two image benchmarks comple-
ment the text experiments. Biometric identity recovery is tested
on Labeled Faces in the Wild (LFW,N= 4,324across
158 identities) [22]. Medical tissue-class recovery is evaluated
on PathMNIST withN= 1,000images [23]. Both are
embedded with CLIP ViT-H/14 (d= 1,024). Ak-NN classifier
(k= 5, cosine metric) handles downstream evaluation. LFW
results are reported as mean ± std across 5 random seeds
{42,43,44,99,100}.
D. Inversion Model & Evaluation Metrics
For the Vec2Text inversion attack (described earlier),
we useload_pretrained_corrector(‘gtr-base’)
and setbeam_width=4,n_steps=20(where 10 steps
reach a plateau). Clinical entity recovery requires an exact
string match after lemmatization. For verifying real data ex-
traction instead of simple keyword matching, ROUGE-L [24]
is used that tracks the longest common subsequence between
the reconstruction and the ground-truth text, penalizing struc-
tural reformulation highly. We also use BERTScore F1 [25] to
measure semantic preservation (catches the cases ROUGE-L
misses, i.e, where wording changes but meaning holds).
V. EMPIRICALFINDINGS
A. Storage-Layer Persistence
Soft-delete works at the metadata layer instead of the
storage layer. We useN= 5,000Wikipedia records
from three databases to initialize HNSW instances
using the standard API of each system. Then, as
mentioned earlier, we soft-delete 10% of each dataset
using a fixed random seeddefault_rng(43).
Eitherindex.reconstruct_n()(for FAISS) or
hnswlib.get_items()(for ChromaDB) were
used to extract vectors. For Vec2Text, we useload_pretrained_corrector(‘gtr-base’)
withbeam_width=4,n_steps=20. AES-256-CTR
is applied to raw float32 bytes with instant key discard in
the cryptographic-erasure baseline. The results confirm that
removed vectors still remain extractable (Table I).
ChromaDB:A ChromaDB instance withN= 5,000
Wikipedia embedded records was created using gtr-t5-base
(d= 768). After soft-deleting 500 records using the standard
delete()API, the directory size remained unchanged. Soft-
deleted vectors take up identical disk space as live vectors.
Deleted records are correctly hidden by the API layer, with
col.get(ids=[‘42’])returning no results. The hnswlib
API, on the other hand, exposesget_items(list_ids),
which returns vectors for each stored ID, independent of
deletion state (for the storage layer, shown above). All 500
soft-deleted vectors are extracted using theget_items()
method, and cosine similarity>0.999is verified using the
original embeddings. This confirms in ChromaDB, soft-delete
is implemented as a metadata operation, not as a physical
storage operation.
FAISS and Weaviate:FAISS does not support deletion.
Since FAISS IndexHNSWFlat [9] provides nodelete()
orremove_ids()methods,RuntimeError: not
implementedis raised whenindex.remove_ids()
is called. Since the HNSW graph traversal algorithm
depends on all nodes remaining connected, all vectors
added to IndexHNSWFlat are kept indefinitely. Using
index.reconstruct_n(0, N), we recover all
N= 5,000vectors, including the 500 ‘soft-deleted’
subset. Vec2Text inversion on FAISS-extracted vectors results
in ROUGE-L = 0.167, slightly below ChromaDB (0.207) but
still indicative of non-trivial text recovery in our evaluation.
The difference is expected:reconstruct_npulls raw floats
directly from the flat index, bypassing the proximity-graph
structure that ChromaDB’s hnswlib backend maintains. The
vectors come out slightly noisier with lower reconstruction
quality. Weaviate 1.26.6 behaves differently at the storage
layer — disk size unexpectedlygrowsafter logical deletion
(52.5MB→55.4MB), rather than contracting. HNSW slots
are not reclaimed; deletion appends metadata instead. The
API correctly returns no results for removed IDs. Weaviate’s
proprietary Go-based storage format blocks direct hnswlib
extraction, but the observed disk growth is consistent with
persistence rather than physical removal, aligning with FAISS
and ChromaDB results.
B. Reconstruction Results (Text)
1) Baseline Extraction (BLP and Wikipedia):Soft-deleted
Wikipedia records retain substantial recoverable semantic
content. We tested 500 soft-deleted records and recovered
original text with a ROUGE-L score of 0.207 (±0.07) and
BERTScore F1 of 0.858 (Table III). 97.8% of reconstructed
records scored above 0.1 on ROUGE-L. Hard deletion elim-
inated recoverable semantic content. Under the AES en-
cryption and key-discard baseline, reconstruction produced
ROUGE-L = 0.000 for all 500 records, showing that the

original semantic information was no longer recoverable. To
contextualize the semantic similarity scores, we also used
a random baseline computed from 500 unrelated Wikipedia
phrase pairs, which produced a BERTScore of 0.606. This
value is used as a random-reference baseline for semantic
similarity. In contrast, soft-deleted records produced higher
scores, with BERTScore values concentrated around 0.85, as
shown in Figure 2. A Wilcoxon signed-rank test confirms
that the difference between soft deletion and hard deletion
was statistically significant (p= 3.95×10−13). The same
threat appears in the Wikipedia BLP dataset, which contains
real biographical information about living individuals. On
98 BLP records, Vec2Text recovered multiple categories of
identifying information from soft-deleted embeddings without
any additional fine-tuning or distribution alignment. Recovery
rates reached 46.4% for geographic locations (GPE), 44.7%
for affiliations (NORP), 25.5% for person names (PERSON),
and 6.2% for organizations (ORG), with a ROUGE-L score
of0.185(±0.062), as shown in Table VII. In contrast, the
hard-delete baseline produced 0% recovery across all entity
categories. To ensure that the recovered information was
coming from deleted records rather than from similar live
records, we also tested the 50 most unique deleted records.
These were records whose entities did not appear in the
remaining live data. Vec2Text achieves 18% entity recovery,
identifying entities from 9 of these 50 records. The deleted
vector itself was the only available source of that information,
creating a temporal gap risk window, in which an attacker
can recover identity-related information from records that were
supposed to have been removed from the system.
Database Soft-Delete Persist ROUGE-L
ChromaDB hnswlib mark YES 0.207
FAISS No remove() YES 0.167
Weaviate SQLite YES†N/A
TABLE I
SOFT-DELETE PERSISTENCE ACROSS EVALUATED SYSTEMS.
Quantization ROUGE-L % Baseline Production Use
Float32 (baseline) 0.207 100% Research
SQ8 (int8) 0.193±0.048 93% Most common
PQ32 (product) 0.162±0.043 78% Memory-optimized
Binary (1-bit) 0.123±0.036 59% Rare
TABLE II
RECONSTRUCTIONQUALITY VS. QUANTIZATIONSCHEME
2) Clinical Extraction on Structured Data:The synthetic
clinical dataset shows that soft-deleted records can expose
medically relevant identifying information. We analyze 1,000
Faker-generated clinical records, designed to resemble real
clinical data. To make them more realistic, the records included
noise such as abbreviations, shorthand, and typos; 58.8% of
the records contained at least one of these variations typical
of clinical language. The results show high recovery rates for
various types of personal identification information. Gender
was the easiest feature to recover, with a recovery rate of
73.4%. Patient first names were recovered in 65.0% of cases,Dataset N Condition ROUGE-L BERTScore
Wikipedia
(orig)200 Soft-Delete 0.207
(±0.07)0.798
Wikipedia
(orig)200 Hard-Delete 0.000
(±0.00)0.566
Wikipedia
(n= 500)500 Soft-Delete 0.207
(±0.07)0.858
Wikipedia
(n= 500)500 Hard-Delete 0.00 (±0.00) 0.566
Clinical (synth) 1000 Soft-Delete 0.085
(±0.04)0.748
Clinical (synth) 1000 Hard-Delete 0.000
(±0.00)0.566
Synthea EHR 1000 Soft-Delete 0.290
(±0.033)0.837
Synthea EHR 1000 Hard-Delete 0.000
(±0.00)0.566
FAISS HNSW 500 Soft-Delete 0.167
(±0.03)N/A
FAISS HNSW 500 Hard-Delete 0.002
(±0.01)N/A
TABLE III
TEXTRECONSTRUCTION: SOFT-DELETE VS. AES.
and last names in 49.9%. Diagnosis information was retrieved
in 28.0% of cases, and physicians’ names in 6.2%. Although
some of these have recovery rates, they remain significant
because any retrieval of patient identity or medical information
still indicates privacy risks under HIPAA and the GDPR.
Table IV provides the full quantitative breakdown.
Cryptographic-erasure/hard deletion baseline eliminated re-
covery entirely, with all reported PII categories dropping
to 0%. Some fields, such as medical record numbers, drug
codes, and dates of birth, could not be recovered by the
corrector model. A Wilcoxon signed-rank test resulted in
p= 5.71×10−12for the clinical comparison, confirming
statistical significance. To test whether recovered identities
resulted from random text generation, we ran Vec2Text on
200 L2-normalized, 768-dimensional Gaussian noise vectors
and extracted entities with spaCy NER. Although 42% of the
random outputs contained somePERSONentity, none matched
the true surname of any individual in the held-out dataset (95%
confidence interval upper bound: 3%). In contrast, Vec2Text
recovered the correct patient surname in 49.9% of the soft-
deleted clinical records, a17×increase over the random
baseline.
3) The Corrector Bottleneck:The comparative lower per-
formance of the reconstruction on real clinical notes is due to
the limitations of the correction model and not to the storage
system itself. Using the zero-shot ‘gtr-base’ corrector on 100
records from the MIMIC-III ICU demo produced very weak
reconstruction results, with ROUGE-L scores close to 0.005
for structured records and free-text clinical notes. The Faker
clinical dataset approximates an upper bound on recovery
for structured clinical text under better corrector alignment.
Reconstruction quality improved when the model was adapted
to the clinical setting by26.6×. Fine-tuning on 43,324 pairs of
MTSamples with oversampled pairs from MIMIC-III resulted

0.4 0.5 0.6 0.7 0.8 0.9 1.0
BERTScore F102468101214DensitySoft-delete
Hard-delete (AES)
Random baseline (0.606)Fig. 2. BERTScore F1 distribution for AES-encrypted versus soft-deleted
embeddings (Wikipedia,n= 500).
Entity Type Soft-Delete (%) Hard-Delete (%)
Gender 73.4 (70.6–76.1) 0
Patient Name 65.0 (62.0–67.9) 0
Last Name 49.9 (46.8–53.0) 0
Diagnosis 28.0 (25.2–30.8) 0
Physician 6.2 (4.7–7.7) 0
Hospital 4.9 (3.6–6.3) 0
Drug 3.4 (2.3–4.6) 0
DOB 0.0 0
MRN 0.0 0
TABLE IV
CLINICALPII RECOVERY: SOFT-DELETE VS. AES.
in an increase in ROUGE-L from 0.005 to 0.077. Beam-search
decoding (num_beams=8,no_repeat_ngram_size=3)
increased ROUGE-L to 0.232 on the retained records (Table
V). As a result, 95% of the retained records exceeded a
ROUGE-L score of 0.10, and 85% exceeded a ROUGE-L
score of 0.15. The storage risk remains independent of cur-
rent corrector’s reconstruction quality. Soft-deleted MIMIC-III
vectors still preserve their exact geometric structure inside the
HNSW index and remain physically recoverable from disk.
Because these indexes preserve structural dependencies, ex-
tracting even a single distinct entity, such as a patient surname,
is sufficient to create meaningful re-identification risk. We
also tested whether the reconstructed text remained seman-
tically similar across different embedding models. The top
50 Wikipedia reconstructions were evaluated using MiniLM,
MPNet, and multilingual-MPNet, which are different from
the originalgtr-t5-basetarget model. Across all three
models, ROUGE-L remained stable at 0.190, while cosine
similarity ranged from 0.86 to 0.90 (Table VI). This confirms
that an attacker does not need access to the victim’s exact
embedding architecture to recover meaningful information.
Recent zero-shot inversion methods, such as Zero2Text [26]
and ZSinvert [27] suggest even stronger cross-model trans-
ferability, indicating that these surrogate results represent a
conservative baseline for future exploit capabilities.Corrector ROUGE-
LImprovement
Zero-shotgtr-base 0.009 —
MIMIC-only fine-tuned (n=
80)0.005 —
Mixed fine-tuned (MTSamples +
MIMIC 10×)0.077 8.6×
Mixed + beam search
(num_beams=8)0.232 26.6×
TABLE V
DOMAINADAPTATION ONMIMIC-III RECONSTRUCTION
Surrogate Model Cosine Sim ROUGE-L
MiniLM-L6 0.86 0.190
MPNet 0.88 0.190
Multilingual-MPNet 0.90 0.190
TABLE VI
CROSS-ARCHITECTURETRANSFERABILITY USINGG T R-B A S E
C. Secondary Risks
1) Persistence under quantization:Vector quantization is
commonly used in production systems to reduce storage and
memory usage. To evaluate the reconstruction quality, we
tested FAISS quantized indexes using binary (1-bit), PQ32
(product quantization), and SQ8 (scalar int8) quantization.
The results are summarized in Table II. The attack remains
partially effective under all tested compression methods. SQ8
and PQ32, two of the most commonly used quantization
approaches, preserved between 78% and 93% of the orig-
inal reconstruction quality. These results suggest that PQ-
based [28] compression alone does not eliminate storage-layer
recovery. Even after PQ32 compression, ghost vectors preserve
significant reconstruction quality, indicating a defense built for
live API queries does not address raw index access.
2) Cross-Modality Persistence:In our experiments, any
type of data stored in an HNSW index as float32 vectors
remain recoverable after soft-deletion. We use CLIP ViT-
H/14 embeddings (d= 1,024) along with ChromaDB to test
biometric and medical image data protected under Article 9
of the GDPR. The same extraction process was used as in
the text-based attack. The downstream evaluation uses ak-NN
classifier (k= 5, cosine similarity), trained on live embeddings
and tested on ghost embeddings (Table VIII).
Tier 1: Facial identity recovery -For biometric data,
ghost embeddings show 99.17% top-1 identity recovery on the
Labeled Faces in the Wild (LFW) dataset, consisting of 158
individuals and 4,324 images. After soft-deletion, the cosine
similarity of ghost embeddings remains at 1.000 (Figure 3a).
The attack outperforms the 0.63% random baseline by 156.7×
(Wilcoxonp <10−12). Since GDPR Article 9 requires explicit
consent for processing biometric data used for identification,
persistence of deleted embeddings raises significant privacy
and compliance concerns.
Tier 2: Histopathology class preservation -Using PathM-
NIST histopathology images, tissue class information re-
mained recoverable at 100% accuracy, with ghost cosine

Dataset Type ROUGE-
LBERTScore Notes
Wiki
(n= 500)Real 0.207 0.858 Primary
result
Wiki BLP Real 0.185 0.585 Living
persons
Synth Clinical
(n= 1,000)Synth 0.085 0.748 Dist-aligned
upper bound
MIMIC-III
TextReal 0.005 —†OOD correc-
tor
MIMIC-III
EHRReal 0.005 —†OOD correc-
tor
MIMIC + Clin-
icalReal 0.232 —‡Fine-tuned +
beam search
OpenAI
EmbedsReal 0.020 —†Arch
mismatch
TABLE VII
ATTACKEFFECTIVENESSACROSSTEXTDISTRIBUTIONS.
†BERTSCORE OMITTED WHEREROUGE-L<0.01
similarity again remaining at 1.000 (Figure 3b). The2.0×
lift represents the mathematical ceiling for balanced binary
classification. Additional tests with domain-specialized models
(torch-xrayvision DenseNet-121 and BiomedCLIP [29]) pro-
duced downstream classifiers that failed to separate classes
effectively because inter-class cosine similarity remained ex-
tremely high (≥0.98). This negative result is still informative
because it shows that ghost vector persistence is independent
of the strength of a downstream classifier. Soft-deleted chest
radiographs remain physically present. The storage-layer threat
may extend to multimodal systems built on HNSW indexes.
The same storage-layer threat affects text, biometric, and
medical image embeddings.
D. Structural Persistence and Side Channels
Soft-delete operates at the metadata layer, instead of the
graph layer. In the HNSW graph, deleted nodes remain linked
to their neighbors. The greedy walk still visits ghost nodes
during search—they are filtered from output but not from
traversal. This structural persistence creates a secondary risk,
that can appear as a query-level side channel.
1) Ghost Traversal Cost:UsingN= 1,000Wikipedia em-
beddings (gtr-t5-base,d= 768,M= 32), we construct
an FAISSIndexHNSWFlatinstance and isolate 50 target
vectors for deletion. We then generate 100 probe queries per
target using a tight Gaussian distribution (σ= 0.01). Under
soft deletion, FAISS reported 672.7 distance computations per
query, which was nearly identical to the pre-deletion baseline.
A Wilcoxon test also showed no statistically meaningful
difference (p= 0.97). Query latency stayed the same as well,
at34.8µs before and after soft deletion. On the other hand, as
seen in Table 10, true deletion reduced latency to33.3µs,
a 4.3% reduction (p= 0.025). In disk-based indices such
as DiskANN [32], this cost could be higher because each
wasted ghost-node traversal may require additional storage
access. Overall, the evaluation shows the HNSW greedy walk
processes soft-deleted nodes with the same computational cost
as live nodes.2) Result-Set Drift:Soft deletion also impacts search re-
sults. As shown in Table IX, 95% of top-Kneighbor sets
differ between the soft-delete and true-deletion conditions.
Ghost nodes can still function as structural hubs that guide
the greedy walk through the embedding space. In this sense,
the deleted record still ‘votes’, or influences the search process.
Soft deletion can change results for live queries. Records that
users expected to be removed can influence other users’ search
results due to deleted nodes remaining structurally present.
3) API-Layer Side Channel:The difference in latency
between soft deletion and true deletion also creates a timing
side channel. Since true deletion reduced latency by 4.3%, an
attacker with API query access can compare latency against a
pre-deletion baseline and attempt to infer whether a node had
been soft-deleted or actually removed. This extends the ghost
vector threat beyond raw storage access and into the query
API layer.
4) Defense Requirements:As detailed in the following
section, proposed Epoch key rotation prevents reconstruction
by encrypting the ghost vector’s contents and destroying the
key. However, this only addresses the content of the vector.
The ghost node can still remain in the HNSW graph. A
more complete defense requires two steps: 1) content erasure
through epoch key rotation and 2) graph repair through neigh-
bor re-linking. In graph repair, the deleted node’s predecessors
would be connected directly to its successors.
5) Adversary Cost:A single inference step during eval-
uation achieves 98.4% of plateau reconstruction quality. On
n= 50Wikipedia records, we perform a step-count sweep
at step counts 1, 5, 10, 20, and 50. ROUGE-L = 0.204 is
recovered in a single step at 30.7 ms/record. Atn_steps=10
(0.2073), ROUGE-L reaches a plateau near 0.207, and ad-
ditional steps have no effect (Figure 4). On a single GPU,
a single-pass inference can invert one million soft-deleted
records (extrapolating from 30.7 ms/record) in less than 9
hours (106×30.7ms≈8.5hours). This verifies the attacker
requires neither specialized hardware nor iterative refinement.
VI. EPOCHKEYROTATIONDEFENSE
A. Formal Specification
Definition 1(Primitive).LetE= (KeyGen,Enc,Dec)
be a symmetric encryption scheme. An epoch key rota-
tion primitive for a vector databaseDconsists of a tuple
(EpochKeyGen,Rotate,Prove)where:
•EpochKeyGen(user id, epoch id)→(k e, metadata):
Generates a fresh 256-bit epoch key, stores only non-
secret (user id, epoch id, creation timestamp) in SQLite
and returns a key handle (not the key itself).
•Rotate(user id,D u)→(D′
u, π): Loads the current
epoch key for useru. AES-256-CTR encrypts all vectors
and generates a new epoch key for subsequent writes and
returns the encrypted embeddings together with proofπ.
•Prove(rotation event)→π: Generates SHA-256 event
hash of(user id+old epoch id+new epoch id+

20
 10
 0 10 20
UMAP 120
10
0102030UMAP 2(a) LFW Faces (N=4,324, 158 persons)
Live embeddings
Ghost vectors (soft-deleted)
2.5
 0.0 2.5 5.0 7.5 10.0 12.5 15.0
UMAP 1456789UMAP 2(b) PathMNIST (N=500, adipose vs. adenocarcinoma)
Adipose (normal)
Adenocarcinoma (cancer)
Ghost vectors (soft-deleted)
Fig. 3. UMAP projections of CLIP ViT-H/14 embeddings.(a)LFW faces.(b)PathMNIST patches. Gold stars mark ghost vectors of soft-deleted images,
perfectly overlapping their original clusters.
Tier Dataset Embedding Modeld NimgNdeleted Ghost Sim Recovery Lift
1 LFW Faces [30] CLIP ViT-H/14 1,024 4,324 1581.000 99.17% 156.7×
2 PathMNIST [31] CLIP ViT-H/14 1,024 1000 501.000 100.0% 2.0×†
†2.0×represents the mathematical ceiling for balanced binary classification.
TABLE VIII
MULTIMODALGHOSTVECTORRECOVERYRESULTS.
0 10 20 30 40 50
Vec2T ext Step Count0.000.050.100.150.200.250.30ROUGE-L Mean
ROUGE-L
Last Name Recovery
020406080100
Recovery Rate (%)
Vec2T ext Step Count vs Reconstruction Quality
Fig. 4. Vec2Text step count vs reconstruction quality. A single inference step
achieves 98.4% of plateau quality at 30.7ms/record.
Metric A (before) B (soft-del) C (true del)
ndis per query 672.7 672.7 669.9
Latency (µs) 34.8 34.8 33.3
Wilcoxonp(vs A) — 0.97 0.025
Result set drift B vs. C:95.0%of neighbors differ
TABLE IX
GHOSTTRAVERSAL: THREE-CONDITIONCOMPARISON.
timestamp), signs with ECDSA-SHA256 using the con-
troller’s private key (SECP256R1). It finally returns a
JSON proof object verifiable with registered public key.Epoch granularity is per-user (eachuser idhas an indepen-
dentepoch idcounter, incremented on deletion request). The
epoch key store must be present in a separate security domain
from the HNSW binary (e.g., KMS, HSM). Standard cloud
architectures meet this by separate IAM-controlled services for
storage and key management. Any vector encrypted under a
discardedk eis computationally indistinguishable from random
noise. The ciphertext reinterpreted as float32 does not preserve
semantic structure for an inversion model to exploit.
B. Failure of Prior Defenses
Prior defenses against inversion attacks mainly function at
query-time and are ineffective at the storage layer. We em-
pirically show that query-side embedding transformations (a
representative query-side defense proposed against embedding
inversion) fail at the storage layer. To guard against embed-
ding inversion, current works [33] propose using a random
orthogonal transformationT. However, any adversary with
access to the server-side configuration can trivially bypass this
mitigation by extractingTand applyingT⊤.
To confirm, we generate a random orthogonal768×768
matrixTusing QR decomposition of a Gaussian matrix.
We then transform 500 soft-deleted embeddings and apply
Vec2Text to the converted vectors. Before reversing the trans-
formation withT⊤, ROUGE-L score was at 0.073. Afterward,
the score rose to 0.215, nearly matching the baseline inversion
attack score of 0.207 (Table X). This shows that embedding
transformation defenses fail against storage-layer adversaries
since the transformation keyTmust remain a persistent secret
that can be extracted directly from server configuration files.

Epoch key rotation prevents this architectural vulnerability
since the AES key is cryptographically destroyed on rotation,
leaving no recoverable secret recoverable.
Condition ROUGE-L vs. Baseline
No transformation (soft-delete) 0.207 —
After applyingT(withoutT⊤) 0.073−65%
After applyingT+ invertingT⊤0.215+4%
Epoch rotation (AES, key destroyed) 0.000−100%
TABLE X
ORTHOGONAL TRANSFORMATION DEFENSE
Approach PII Rec. ROUGE-
LR@10 Time Proof
Soft-delete High
(49.9/
73.4%)0.2073 100% ∼0ms No
Full rebuild High 0.2073 100% 2.25s
(N=500)No
Per-rec.
AES0% 0.0000 90.4% 8ms No
Epoch rot. 0% 0.0000 90.4% 2.5ms Yes
TABLE XI
DEFENSECOMPARISON. SPEEDUP RELATIVE TO FULL REBUILD,
N=100,000.
C. Evaluation
Epoch key rotation shows approximately linear scaling. We
measure total runtime atN= 100,N= 500, andN= 5,000
records, obtaining 0.51 ms atN= 100, 2.48 ms atN= 500,
and 24.93 ms atN= 5,000respectively. These measurements
correspond to an amortized AES throughput of approximately
0.005 ms per record. Extrapolating toN=100,000 gives about
550 ms for vector encryption. This confirms linear scaling of
AES-256-CTR throughput. ECDSA proof generation adds∼3
ms, bringing the full protocol to∼553 ms. Structural graph
repair (neighbor re-linking) is discussed in Section VIII.
In the real world, epoch-encrypted vectors face minimal
retrieval degradation when they remain in the graph. Since
retrieval degradation is proportional to the ratio of encrypted
to live vectors, the degradation remains within tolerance for
most RAG deployments. Systems needing higher retrieval
precision should schedule frequent physical cleanup of
encrypted vectors. The cryptographic proofπis generated
at rotation time and remains independently verifiable even
if encrypted vectors are physically removed later. We
empirically compare four deletion strategies below on the
same 500 Wikipedia soft-deleted vectors. The results are
summarized in Table XI.
i) Soft-delete: Time:∼0 ms, ROUGE-L=0.207, without
proof (baseline).
ii) Full index rebuild: Time: 2,250ms, ROUGE-L=0.207
(same as soft-delete), without proof. Index rebuild is not
retroactive. An adversary who copies the binary before rebuild
achieves identical reconstruction quality.iii) Per-record AES: Time: 8 ms, ROUGE-L=0.000, without
proof. Effective but 3.2×slower than epoch rotation and
produces no verifiable evidence.
iv) Epoch key rotation: Time: 2.5 ms at N = 500, ROUGE-
L=0.000, cryptographic proofπ(∼300×faster than full
index rebuild and 53×faster thanan insert-onlyrebuild at
N= 100,000). End-to-end runtime is about 553 ms including
proof generation. This approach provides verifiable erasure.
ϵ σ ROUGE-
L95% CI R@10 Cos.
Sim
100 0.048 0.0830 [0.082,
0.084]55.8% 0.950
50 0.097 0.0760 [0.075,
0.077]26.6% 0.850
20 0.242 0.0747 [0.074,
0.076]4.8% 0.550
10 0.484 0.0756 [0.075,
0.077]0.4% 0.079
5 0.969 0.0749 [0.074,
0.076]0.1% 0.025
2 2.419 0.0734 [0.072,
0.074]0.0% 0.008
1 4.839 0.0748 [0.074,
0.076]0.0% 0.005
0.5 9.678 0.0749 [0.074,
0.076]0.0% 0.004
0.1 48.39 0.0730 [0.072,
0.074]0.0% 0.001
0.05 96.78 0.0730 [0.072,
0.074]0.0% 0.000
0.01 483.9 0.0730 [0.072,
0.074]0.0% 0.000
Epoch
rot.— 0.000 [0, 0] 0.0% N/A
TABLE XII
DIFFERENTIALPRIVACY VS. VEC2TEXTRECOVERY
D. Why Differential Privacy Fails
Our findings show Gaussian DP noise creates a poor
privacy-utility tradeoff for deleted embeddings. Starting from
500 soft-deleted Wikipedia embeddings, we apply 11 privacy
budgets:ϵ∈ {100,50, ...0.05,0.01}(Table XII), withδ=
10−5. The Gaussian mechanism usesσ=p
2 ln(1.25/δ)·
∆f/ϵwhere sensitivity∆f= 1for L2-normalized embed-
dings. We apply Vec2Text on each noise-perturbed embedding
(n= 10steps) and calculate ROUGE-L recovery and cosine
similarity with the original embedding. Cosine similarity falls
to 0.079 atϵ= 10(weak privacy), indicating that the
embedding is semantically destroyed, while ROUGE-L stays at
0.082. Atϵ= 0.1, ROUGE-L remains around 0.073 while co-
sine similarity drops to 0.001 (confirming structural geometry
is completely lost). The mechanism of failure is demonstrated
by the decoupling between ROUGE-L and cosine similarity:
the embedding space is collapsed by DP noise, but Vec2Text
smoothly degrades into fluent generic text that still has overlap
with the target. ROUGE-L remains around 0.073–0.083 across
the testedϵvalues. Thus, DP suppresses semantic recovery,

but does not eliminate it. Anyϵstrong enough to prevent
reconstruction also makes the index unusable for its primary
retrieval purpose.
The Utility-Privacy Trade-Off:The utility-privacy trade-
off fails for deleted data. ROUGE-L scores below 0.05 would
require noise levels that destroy retrieval utility. Evenσ= 0.48
(ϵ= 10) adds a noise magnitude of√
768·0.48≈13.4
in 768-dimensional space, rotating the embedding signifi-
cantly (cosine similarity≈0.08). Preserving retrieval utility
(recall@10≥0.8) requires cosine similarity>0.85and
σ <0.02, corresponding toϵ >200—a budget that offers no
privacy protection. For protecting deleted records, DP intro-
duces noise to live records. When the record is live, the noise
must be small enough to preserve retrieval. This guarantees the
noise is too small to protect the record once deleted. Epoch
rotation cryptographically renders deleted data unrecoverable
under the stated assumptions rather than attempting to obscure
it. A deleted record becomes completely irrecoverable by
construction. Epoch key rotation results in ROUGE-L = 0.000
at minimal utility cost for remaining records (Figure 5).
0.1 0.5 1 10
Differential Privacy Epsilon (log scale,  more private)
0.000.020.040.060.08ROUGE-L (text recovery)
Epoch Rotation
ROUGE-L
Epoch Rotation
Cosine Similarity0.000.010.020.030.040.050.060.070.08
Cosine Similarity (embedding utility)
Fig. 5. Differential Privacy vs. Retrieval Utility Tradeoff. x-axis showsϵ(log
scale; privacy increases right-to-left). Left y-axis: ROUGE-L (text recovery).
Right y-axis: cosine similarity (embedding utility)
VII. DISCUSSION
GDPR Compliance and Legal Implications:
This paper shows data persistence at the storage layer raises
serious GDPR compliance concerns. Article 17 requires verifi-
able erasure, not just functional deletion. An API returning no
records can mean either deletion or suppression. Soft-delete,
therefore, provides no real proof of deletion. Three explicit
regulatory requirements include Art. 5(1)(e) storage minimiza-
tion (data cannot be kept past its intended use), Art. 17(1)
erasure right (deleted data should no longer remain recover-
able), and Art. 5(2) accountability (proofπprovides auditable
evidence). As per EDPB guidelines (05/2019), erasure must
be ‘verifiable and irreversible’ [2]. Thus, soft deletion raises
compliance concerns when deleted vectors remain recoverable
from storage. Epoch key rotation supports this requirement by
destroying the cryptographic keys needed to recover the data
and generating ECDSA proofs that can be audited.Scope, Controller/Processor, and HIPAA:Article 17
becomes applicable when a RAG corpus contains personal
data, including consumer-facing applications and SaaS deploy-
ments (Article 28). The compliance gap is more severe for
multimodal RAG systems handling Article 9 special-category
data, such as facial biometrics and histopathology images.
Within U.S. healthcare context, NIST SP 800-88 categorizes
cryptographic erasure as a PHI sanitization technique. The
storage-layer persistence finding remains completely inde-
pendent of transient reconstruction quality. This structural
survival serves as the main compliance argument for free-form
clinical notes, where the immediate re-identification threat
depends on corrector availability. Under EDPB Guidelines
05/2019, deletion must extend to all storage layers, including
backup snapshots [2]. Effective compliance requires a breach-
resistant erasure mechanism. By producingπin response to an
Article 17 audit, a data controller satisfies the demonstrability
requirement regardless of attack success, providing a stronger
step toward cryptographic proof of key destruction.
Compliance debt:We introduce the concept of compliance
debt to describe the accumulated compliance burden: each
deletion request can leave one more recoverable record at
the storage layer. Epoch key rotation addresses this burden
by rendering deleted vectors unrecoverable under the stated
assumptions and by generating a signed proofπ.
VIII. LIMITATIONS ANDFUTUREWORK
Corrector Generalization and Scale:Two main factors
limit the current work. First, the clinical evaluation uses
synthetic records, and validation on raw MIMIC-III clini-
cal notes showed a ROUGE-L score of 0.009. Because the
gtr-t5-basecorrector was trained on Wikipedia-style text,
it does not generalize well to clinical terminology. We argue
that the bottleneck is due to the corrector rather than the
storage. On disk, the soft-deleted MIMIC-III vectors remain
physically retrievable. Near-term extensions include clinical
corrector training. Initial fine-tuning onN= 43,324MTSam-
ples pairs improves MIMIC-III ROUGE-L from 0.009 to 0.052
(5.9×). Additional domain-specific training on MIMIC-III or
i2b2 may improve recovery further. Second, our experiments
span103to105records, which is smaller than production de-
ployments that typically handle between107and1010records.
Computational constraints prevented larger trials, although
the geometric properties of HNSW suggest that the attack
should remain feasible at larger scales. Future research should
evaluate additional cloud-hosted vector databases, including
Pinecone and Qdrant, to determine whether comparable per-
sistence behavior appears under their storage designs.
Multimodal ghost vectors also raise intellectual-property
risks. Source code embeddings (e.g., CodeBERT, GraphCode-
BERT) may introduce analogous ghost vector IP theft. Soft-
deleted proprietary function logic remains as semantic ghost
embeddings in any HNSW-backed code search or copilot
system, extending the privacy threat beyond personal data
(GDPR) to IP protection. Some Multimodal RAG systems in
production may use a shared HNSW index to combine text,

image, and code embeddings. In a single storage breach, a
ghost vector adversary could also recover deleted text PII,
biometric identity, and proprietary code, further compounding
the threat. Epoch key rotation extends naturally to this case
since AES-256-CTR is modality-blind.
Hardware Isolation and Computational Storage:Our
evaluation of epoch key rotation is implemented as a software
prototype. This creates a trust dependency on the host OS. In
the event of a breach, a root-level adversary with access to
the host CPU can potentially extract active AES-256 epoch
keys from RAM before rotation. Mitigating this dependency
requires shifting the cryptographic boundary to hardware via
Computational Storage Drives (CSDs) with embedded ARM
cores or FPGAs, such as the Samsung SmartSSD [34]. By of-
floading the index to the drive, the host OS never sees the raw
float32 vectors or the epoch keys, because HNSW traversal and
AES decryption are executed directly on the drive controller.
A hardware-assisted design thus could move key invalidation
closer to the storage device. This creates a path towards
stronger isolation for vector-database deployments and ensures
cryptographic erasure even when the host kernel is fully com-
promised. One limitation that no corrector advance can address
is forward secrecy. Similar to TLS sessions recorded before a
key rotation, vectors extracted before epoch rotation remain
invertible. This represents a fundamental design limitation,
rather than a flaw in the current implementation.
IX. RELATEDWORK
Prior research showed that dense embeddings can leak
semantic information [35]. Follow-up works developed both
zero-shot [26], [27] and iterative [5], [28] inversion methods.
Zhuang et al. [28] presented a detailed study of inversion risks
and query-layer defenses, including embedding transforma-
tions [36]. Most earlier studies assume an adversary at the
API layer targetingliveembeddings. When this threat model
is extended to the storage layer, the semantic vulnerability can
remain even after data is logically removed, and query-side
transformations remain reversible viaT⊤.
To avoid the severe I/O penalties of graph restructuring [37],
vector databases such as ChromaDB [8], FAISS [9], [10],
Weaviate [11] or Pinecone [12] use metadata-driven soft-
deletion. In some systems, physical removal is deferred until
an asynchronous batch cleanup. This can create a window
during which ghost vectors remain reconstructible on disk
under the paper’s threat model. Multiple modern system-level
defenses address related parts of this problem. MemTrust [38]
has been proposed to protect AI memory systems, which is
related to our work but addresses a different systems boundary
than post-deletion erasure. Alqithami [39] explores privacy-
aware memory architectures that are adjacent to graph-level
deletion concerns, but not identical to the storage-layer cryp-
tographic erasure problem studied here. Our epoch key rotation
protocol can complement such architectures as a storage-layer
cryptographic mechanism for deleted embeddings. Unlike
asynchronous batch cleanup alone, the protocol generates a
signed proof object (π) at deletion time.Machine unlearning studies the impact of removing training
influence on neural network model weights [40]–[42]. These
approaches need high computation and operate at the model
layer, but do not address the persistence of ghost vectors in
an HNSW index. Similarly, membership inference attacks [43]
target a different privacy threat surface based on model outputs
or query behavior that a breach-level storage adversary cannot
access. Prior works also explored secure data expiration. Van-
ish [44] and Perlman’s Ephemerizer [45] used time-bounded
cryptographic keys to enable data expiration. By adapting this
idea to contemporary vector databases, epoch key rotation
bridges the gap between the physical realities of storage [46]
and secure deletion compliance requirements [4]. Finally, so-
lutions such as indirect prompt injection defenses [47], clinical
de-identification [48], [49], and Cloaked AI [50] protect active
RAG pipelines, but they are orthogonal to our threat model.
They do not address storage-layer persistence or cryptographic
proof of deletion under the threat model studied here.
X. CONCLUSION
This paper reviews how soft-deletion in modern vector
databases can create an illusion of compliance. Logical dele-
tion in RAG-based applications (such as ChromaDB, FAISS,
and Weaviate) leaves high-dimensional embeddings physically
intact within the HNSW index. The persistence lies entirely
at the storage layer, not prevented by API-level access control
under the threat model. Basic inversion models successfully
recovered sensitive data from the raw storage directory, includ-
ing patient surnames, biometric facial identities, and clinical
attributes from ‘deleted’ records in the evaluated datasets.
The threats extends across modalities and bypasses API-level
access since float32 persistence is a basic property of the
system. The evaluated alternatives do not fully address this
problem. If an adversary copies the binary before a rebuild
runs, full index rebuild later provide no retroactive protection.
Differential privacy showed a poor privacy-utility: noise levels
high enough to suppress recovery also destroys retrieval utility.
Epoch key rotation closes this gap at the storage layer. Under
the paper’s assumptions, epoch key rotation completes in 2.5
ms for 500 deleted vectors and reduces observed recovery to
0% while generating a signed proof objectπat rotation time.
When data controllers rely on soft-delete, deleted data may
remain physically recoverable even after it disappears from
the API. Epoch key rotation addresses this problem at the
storage layer by converting silent suppression into erasure
that can be verified cryptographically. Whether the data is
text, images, or special-category medical data under GDPR
Article 9, epoch key rotation replaces assumed compliance
with verifiable proof.
XI. ACKNOWLEDGMENT
The authors take full responsibility for the technical content
of this paper. AI-assisted writing tools were used only for
limited language editing in accordance with conference policy.

REFERENCES
[1] “Regulation (eu) 2016/679 of the european parliament and of the council
(general data protection regulation),” 2016. Article 17: Right to erasure.
[2] E. D. P. Board, “Guidelines 05/2019 on the right to erasure,” 2019.
[3] E. M. Villaronga, P. Kieseberg, and T. Peyr, “Humans forget, machines
remember: Artificial intelligence and the right to be forgotten,”Com-
puter Law & Security Review, vol. 34, no. 2, pp. 304–313, 2018.
[4] E. Politou, E. Alepis, and C. Patsakis, “Forgetting personal data and
revoking consent under the gdpr: Challenges and proposed solutions,”
Journal of Cybersecurity, vol. 4, no. 1, 2018.
[5] J. X. Morris, W. Zhao, J. T. Chiu, V . Shmatikov, and A. M. Rush,
“Transformers can invert their inputs: Joint embeddings and inverse
transformers for text reconstruction,”arXiv preprint arXiv:2305.12556,
2023.
[6] Y . A. Malkov and D. A. Yashunin, “Efficient and robust approxi-
mate nearest neighbor search using hierarchical navigable small world
graphs,”IEEE Transactions on Pattern Analysis and Machine Intelli-
gence, vol. 42, no. 4, pp. 824–836, 2018. arXiv:1603.09320.
[7] A. Babenko and V . Lempitsky, “Efficient indexing of billion-scale
datasets of deep descriptors,” inIEEE Conference on Computer Vision
and Pattern Recognition, pp. 2055–2063, 2016.
[8] “Chromadb,” 2023. Accessed: 2026-02-21.
[9] J. Johnson, M. Douze, and H. J ´egou, “Billion-scale similarity search
with gpus,”IEEE Transactions on Big Data, vol. 7, no. 3, pp. 535–547,
2019.
[10] M. Douze, A. Guzhva, C. Deng, J. Johnson, G. Szilvasy, P. Mazar ´e,
M. Lomeli, L. Hosseini, and H. J ´egou, “The faiss library,”arXiv preprint
arXiv:2401.08281, 2024.
[11] “Weaviate,” 2023. Accessed: 2026-02-21.
[12] “Pinecone vector database,” 2023. Accessed: 2026-02-21.
[13] J. Ni, C. Li, and J. McAuley, “Large dual encoders are generalizable
retrievers,”arXiv preprint arXiv:2112.09118, 2022.
[14] N. Reimers and I. Gurevych, “Sentence-bert: Sentence embeddings using
siamese bert-networks,” pp. 3982–3992, 2019.
[15] S. Potluri, “A deep learning-driven framework for detecting anomalous
data breaches in distributed cloud storage infrastructures,”International
Journal of Artificial Intelligence, Data Science, and Machine Learning,
vol. 5, no. 3, pp. 80–87, 2024.
[16] U.S. Department of Health & Human Services, Office for Civil Rights,
“Guidance regarding methods for de-identification of protected health
information in accordance with the HIPAA privacy rule,” tech. rep.,
U.S. Department of Health & Human Services, Nov. 2012.
[17] M. Bellare, A. Desai, D. Pointcheval, and P. Rogaway, “Relations
among notions of security for public-key encryption schemes,” inAnnual
International Cryptology Conference, pp. 26–45, Springer, 1998.
[18] N. I. of Standards and Technology, “Fips pub 197: Advanced encryption
standard (aes),” tech. rep., U.S. Department of Commerce, 2001.
[19] M. Bellare and P. Rogaway, “Authenticated encryption in the public-key
setting: Security notions and analyses,” inCRYPTO, 2000.
[20] F. S. Hameed, P. Fatima, M. P. Venkataprasad, J. Subramanian, and
S. Ganesan, “Generating classes of test data using fake data generation
and static data manipulation,” inAIP Conference Proceedings, vol. 3279,
p. 020088, AIP Publishing LLC, 2025.
[21] A. E. Johnson, T. J. Pollard, L. Shen, L.-w. H. Lehman, M. Feng,
M. Ghassemi, B. Moody, P. Szolovits, L. A. Celi, and R. G. Mark,
“Mimic-iii, a freely accessible critical care database,”Nature Scientific
Data, vol. 3, no. 1, pp. 1–9, 2016.
[22] G. B. Huang, M. Ramesh, T. Berg, and E. Learned-Miller, “Labeled
faces in the wild: A database for studying face recognition in uncon-
strained environments,” Tech. Rep. 07-49, University of Massachusetts,
Amherst, Oct. 2007.
[23] J. Yang, R. Shi, D. Wei, Z. Liu, L. Zhao, B. Ke, H. Pfister, and B. Ni,
“Medmnist v2: A large-scale lightweight benchmark for 2d and 3d
biomedical image classification,”Scientific Data, vol. 10, no. 1, p. 41,
2023.
[24] C.-Y . Lin, “Rouge: A package for automatic evaluation of summaries,”
inText Summarization Branches Out, pp. 74–81, 2004.
[25] T. Zhang, V . Kishore, F. Wu, K. Q. Weinberger, and Y . Artzi, “Bertscore:
Evaluating text generation with bert,” inInternational Conference on
Learning Representations, 2020.
[26] Anonymous, “Zero2text: Zero-training embedding inversion via la-
tent space optimization,”arXiv preprint, 2025. Concurrent work onzero-training inversion; higher cross-architecture transferability than
Vec2Text.
[27] Anonymous, “Zsinvert: Zero-shot text reconstruction from embeddings,”
arXiv preprint, 2025. Concurrent work on zero-shot inversion; validates
threat model assumption.
[28] S. Zhuang, B. Koopman, X. Chu, and G. Zuccon, “Understanding and
mitigating the threat of vec2text to dense retrieval systems,” inPro-
ceedings of the 47th International ACM SIGIR Conference on Research
and Development in Information Retrieval (SIGIR ’24), pp. 1234–1243,
ACM, 2024.
[29] H. Damm, T. M. Pakull, H. Becker, B. Bracke, B. Eryilmaz, L. Bloch,
R. Br ¨ungel, C. S. Schmidt, J. R ¨uckert, O. Pelka,et al., “Overview
of imageclefmedical 2025–medical concept detection and interpretable
caption generation,”Proceedings of the CLEF , Madrid, Spain, pp. 9–12,
2025.
[30] G. B. Huang, M. Ramesh, T. Berg, and E. Learned-Miller, “Labeled
faces in the wild: A database for studying face recognition in uncon-
strained environments,” Tech. Rep. 07-49, University of Massachusetts,
Amherst, 2007. Publicly available face benchmark for unconstrained
recognition.
[31] J. Yang, R. Shi, D. Wei, Z. Liu, L. Zhao, B. Ke, H. Pfister, and B. Ni,
“Medmnist v2 – a large-scale lightweight benchmark for 2d and 3d
biomedical image classification,”Scientific Data, vol. 10, no. 1, p. 41,
2023.
[32] J. Sun, G. Li, J. Pan, J. Wang, Y . Xie, R. Liu,et al., “GaussDB-Vector:
A large-scale persistent real-time vector database for LLM applications,”
inProceedings of the 2025 International Conference on Very Large Data
Bases (VLDB), 2025.
[33] e. a. Zhang, “Random orthogonal transformation as a defense against
embedding inversion,”arXiv preprint arXiv:2402.12784, 2024.
[34] C. Chakraborttii,Improving performance of solid state drives using
machine learning. PhD thesis, University of California, Santa Cruz,
2021.
[35] C. Song and A. Raghunathan, “Information leakage in embedding
models,” inProceedings of the 2020 ACM SIGSAC conference on
computer and communications security, pp. 377–390, 2020.
[36] J. X. Morris, W. Zhao, J. T. Chiu, and V . Shmatikov, “Understanding
and mitigating the threat of vec2text to dense retrieval,”arXiv preprint
arXiv:2402.12784, 2024. Vec2Text attack on live dense retrieval sys-
tems; query-side embedding transformation defense.
[37] e. a. Wang, “Hnsw index deletion evaluation: Persistent storage ar-
tifacts,”arXiv preprint arXiv:2512.06200, 2025. Evaluates HNSW
deletion behavior across implementations.
[38] X. Zhou, D. Ustiugov, H. Shang, and K. Lin, “Memtrust: A zero-
trust architecture for unified ai memory system,”arXiv preprint
arXiv:2601.07004, 2026.
[39] S. Alqithami, “Forgetful but faithful: A cognitive memory architecture
and benchmark for privacy-aware generative agents,”arXiv preprint
arXiv:2512.12856, 2025.
[40] Y . Cao and J. Yang, “Towards making systems forget with machine
unlearning,” in2015 IEEE Symposium on Security and Privacy, pp. 463–
480, 2015.
[41] A. Ginart, M. Y . Guan, G. Valiant, and J. Zou, “Making ai forget you:
Data deletion in machine learning,” inAdvances in Neural Information
Processing Systems 32 (NeurIPS 2019), 2019.
[42] W. Li and J. Zhao, “Beyond data erasure: The eu’s convoluted approach
to model deletion,”Available at SSRN, 2025.
[43] R. Shokri, M. Stronati, C. Song, and V . Shmatikov, “Membership
inference attacks against machine learning models,” inIEEE Symposium
on Security and Privacy, pp. 3–18, 2017.
[44] R. Geambasu, T. Kohno, A. Levy, and H. M. Levy, “Vanish: Increasing
data privacy with self-destructing data,” inUSENIX Security Symposium,
2009.
[45] R. Perlman, “The ephemerizer: Making data disappear,”Sun Labs
Technical Report, 2005.
[46] P. Gutmann, “Secure deletion of data from magnetic and solid-state
memory,” in6th USENIX Security Symposium, pp. 77–90, 1996.
[47] K. Greshake, S. Abdelnabi, S. Mishra, C. Endres, T. Holz, and
M. Fritz, “Not what you’ve signed up for: Compromising real-world
llm-integrated applications with indirect prompt injection,” inAISec
Workshop at ACM CCS, 2023.
[48] A. Stubbs, C. Kotfila, and ¨O. Uzuner, “Automated systems for the
de-identification of longitudinal clinical narratives: Overview of 2014

i2b2/uthealth shared task track 1,”Journal of Biomedical Informatics,
vol. 58, pp. S11–S19, 2015.
[49] F. Dernoncourt, J. Y . Lee, O. Uzuner, and P. Szolovits, “Identification
of protected health information in clinical notes using neural networks,”
Journal of the American Medical Informatics Association, vol. 24,
no. e1, pp. e114–e118, 2017.
[50] I. Labs, “Cloaked ai: Encryption sdk for protecting ai embeddings in
vector databases.”
urlhttps://ironcorelabs.com/products/cloaked-ai/, 2023. Client-side en-
cryption for embeddings before storage; protects live data at rest but
does not address post-deletion cryptographic erasure or proof of erasure.