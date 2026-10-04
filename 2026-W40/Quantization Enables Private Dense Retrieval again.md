# Quantization Enables Private Dense Retrieval against Malicious Service Providers

**Authors**: Louis Tremblay Thibault, Sofiane Azogagh, Marc-Olivier Killijian, Ulrich Aïvodji

**Published**: 2026-09-28 23:07:39

**PDF URL**: [https://arxiv.org/pdf/2609.36376v1](https://arxiv.org/pdf/2609.36376v1)

## Abstract
Dense retrieval, the key component of Retrieval Augmented Generation (RAG), retrieves the most relevant documents by comparing dense vector representations of queries and passages from a large corpus. In privacy-sensitive applications, the server observes the query and controls which evidence is returned, creating both confidentiality and integrity risks. We formulate private dense retrieval as providing query privacy and retrieval integrity against a malicious server, and develop a two-round cryptographic protocol that provides both guarantees. Our protocol reduces private and verifiable retrieval to multiplication of a committed matrix by an encrypted vector and uses low-bit quantization to make this computation practical. We evaluate the resulting trade-off between cryptographic cost, retrieval quality, and downstream RAG accuracy across six embedding models, four language models, and corpora of up to 2.68 million passages. Our results show that, with a clipped quantizer, three-bit quantization largely preserves retrieval quality and downstream accuracy, while a private query over a corpus the size of a clinical reference requires one to three minutes of server time. These results suggest that private dense retrieval is already practical for moderately sized, privacy-sensitive corpora when minute-scale latency is acceptable.

## Full Text


<!-- PDF content starts -->

Preprint.
QUANTIZATIONENABLESPRIVATEDENSERETRIEVAL
AGAINSTMALICIOUSSERVICEPROVIDERS
Louis Tremblay Thibault Sofiane Azogagh
École de technologie supérieure and Mila Eurecom
Marc-Olivier Killijian Ulrich Aïvodji
Université du Québec à Montréal École de technologie supérieure and Mila
ABSTRACT
Dense retrieval, the key component of Retrieval Augmented Generation (RAG),
retrieves the most relevant documents by comparing dense vector representations
of queries and passages from a large corpus. In privacy-sensitive applications, the
server observes the query and controls which evidence is returned, creating both
confidentiality and integrity risks. We formulate private dense retrieval as provid-
ing query privacy and retrieval integrity against a malicious server, and develop
a two-round cryptographic protocol that provides both guarantees. Our protocol
reduces private and verifiable retrieval to multiplication of a committed matrix by
an encrypted vector and uses low-bit quantization to make this computation prac-
tical. We evaluate the resulting trade-off between cryptographic cost, retrieval
quality, and downstream RAG accuracy across six embedding models, four lan-
guage models, and corpora of up to 2.68 million passages. Our results show that,
with a clipped quantizer, three-bit quantization largely preserves retrieval quality
and downstream accuracy, while a private query over a corpus the size of a clinical
reference requires one to three minutes of server time. These results suggest that
private dense retrieval is already practical for moderately sized, privacy-sensitive
corpora when minute-scale latency is acceptable.
1 INTRODUCTION
Dense retrieval is a central paradigm for large-scale information access. Bi-encoder mod-
els (Karpukhin et al., 2020; Reimers & Gurevych, 2019; Ni et al., 2022) represent documents and
queries as vectors in a shared embedding space and answer queries by nearest-neighbor search. This
is the mechanism underlying semantic search, open-domain question answering and the retrieval
stage of retrieval-augmented generation (RAG) (Lewis et al., 2020). For exact retrieval by inner-
product scoring, the interaction can be expressed as a matrix-vector product: the server holds an
embedding matrixA∈Rm×n, where each row represents a document, and the user submits a
query embeddingx∈Rn. In such a setting, retrieval reduces to computing the score vectorAxand
fetching the highest-scoring documents.
However, this interaction exposes query embeddings to the server in the clear, potentially revealing
sensitive information about the user’s intent. In particular, these embeddings can be linked across
queries and inverted to recover information about the underlying text (Morris et al., 2023; Song &
Raghunathan, 2020). Consider clinical decision support, where AI-assisted synthesis of the medical
literature is increasingly common1and physicians submit queries to tools such asUpToDateand
OpenEvidence, which report reaching over three million clinicians worldwide and a majority of US
physicians, respectively (Wolters Kluwer, 2026a; NBC News, 2026). Every such query is visible to
the provider, to a compromised insider or to anyone who breaches the service or its logs.2By cross-
referencing with a clinician’s appointment schedule, an adversary could link individual patients to
139%of US physicians report incorporating summaries of medical research and standards of care into
practice, up from13%two years earlier (American Medical Association, 2026).
2Medical IT providers are attractive targets for cyberattacks in practice, as in the 2026 Cegedim Santé
incident that exposed 15 million medical files (Agence France-Presse, 2026).
1
arXiv:2609.36376v1  [cs.CR]  28 Sep 2026

Preprint.
Preprocessing: Commit to CorpusRound 1: Request scoresVerify then decryptEncrypt the indices
Service ProviderClientquantized embedding matrixquantized content matrixCorpusAmnDmℓmCommitcA,cDcA
"..ﬁrst line therapy for.."Encypt the requestEncodexnEncryptc1c1Public digest
Round 2: Request documentscDc1A⋅=c′ 1c′ 1,π1Prove(cA,c1,c′ 1)=Decryptπ1Verifyc′ 1ScoressTop-kks1…skDTOne-hot encodemEncryptc2si*ei*c2⋅=c′ 2π1
π2Prove(cD,c2,c′ 2)=c2c′ 2,π2Verify then decryptDecryptπ2Verifyc′ 2Most relevant content
Figure 1: Private dense retrieval. Round 1 returns encrypted scores, from which the client selectsi∗;
Round 2 fetchesD i∗. Each proof is verified against the committedAorDbefore decryption.
the topics their physician searched and reconstruct a working diagnosis. Such linkage can disclose
protected health information subject to the HIPAA Privacy Rule (Office for Civil Rights, 2002).
Beyond query privacy, the client also needs guarantees on the integrity of the retrieved evidence.
Indeed, the server currently decides what evidence reaches the user or the downstream language
model. One that silently rescores documents or substitutes the returned passage can steer diagnoses
or generated answers, and an encrypted-but-unverified protocol gives the clinician no way to notice.
Existing systems provide one of these two guarantees but not both, and those that encrypt the query
assume an “honest-but-curious” server, which contradicts the threat model that motivates encryption
in the first place (§3). Retrieval integrity also boundsindirect prompt injection(Greshake et al.,
2023) through the retrieval channel: once a corpus has been audited and committed, neither the
provider nor an attacker with write access to the corpus can slip adversarial instructions into the
passages that reach the reader, since the client accept only content consistent with the audited digest
(§2).
Background.Homomorpic encryption (HE) (Gentry, 2009) is a form of encryption that allows
computations on encrypted data without decrypting it first. The party holding the encryption key
can decrypt the result of the computation, but the party performing the computation learns nothing
about the underlying data. Over nearly two decades, HE has evolved from a theoretical topic to
a vast field of privacy-preserving applications. For example, HE has been used to enable Private
Information Retrieval (PIR) (Chor et al., 1998), where a client can retrieve an item from a server’s
database without revealing to the server which item it is has retrieved. It is crucial to understand that
homomorphically encrypted ciphertexts must containnoisefor security. If not carefully managed,
noise accumulation over homomorphic operations can lead to incorrect computation results. As
such, smaller noise allows for more operations over the ciphertexts, but guarantees a lower security
level. This trade-off is central to our work and we perform in §4.2 the careful balancing act of
choosing cryptographic parameters that provide adequate levels of both security and correctness for
the tasks at hand.
Our approach.Our starting point is the observation that the two stages of dense retrieval, scoring
and document fetch, reduce to a product between a public matrix held by the server (e.g.,provider)
and a private, encrypted vector held by the client (e.g.,clinician). We instantiate both with a single
maliciously secure primitive, verifiable matrix-vector multiplication under HE (Tremblay Thibault
et al., 2026). The server cryptographically commits to a matrix, multiplies its committed matrix by
the client’s encrypted vector and returns the encrypted result with a succinct proof, which the client
verifies before decrypting. The primitive guarantees integrity by construction, and we show in §2
2

Preprint.
that verifying before decrypting also closes the reaction-attack channel (Chillotti et al., 2016) that
honest-but-curious designs leave open. The resulting two-round protocol, illustrated in Figure 1,
is exact (every document is scored, with no clustering or hashing approximation), secure against a
malicious server (any manipulation of scores or of the returned document is detected except with
negligible probability), client-lightweight (per round the client encrypts one vector, verifies one
proof and decrypts one result) and modular (both rounds use one primitive behind one interface, so
any faster instantiation may be swapped in directly).
Setting.The naive way to hide a query is to send the corpus to the client and have it search locally.
That option is not available here: the clinical references that physicians consult are proprietary
and licensed per seat, so the provider grants query-time access rather than a copy. As such, the
setting we consider is a server-held corpus queried by a client that must not reveal its queries. We
target deferred-response settings, such as literature requests to medical librarians or tumor-board
preparation, where users expect a researched answer rather than an immediate response and minute-
scale latency is acceptable; we evaluate the protocol’s cost in this regime (§4.3).
Contributions.
1. We formalize private dense retrieval as requiring both query privacy and retrieval integrity
against a malicious server, and reduce it to two instances of verifiable matrix-vector multi-
plication (§2). We give a two-round protocol realizing this functionality, with exact retrieval
and a verify-before-decrypt mechanism that provably blocks reaction attacks (§2).
2. Using the protocol as an instrument, we experimentally locate the operating point of private
dense retrieval atb=3bits of quantization: retrieval quality and downstream RAG accuracy
hold there for six encoders and four readers, on corpora of up to2.68M passages (§4.1,
§4.4).
3. We measure the cost of a private, verifiable query in time and dollars at corpus sizes up to
106chunks, and show that bit-width buys performance, with a≈2×increase betweenb=2
andb=3(§4.2, §4.3). Atb=3a corpus the size of a clinical reference is served in one to
three minutes, while the literature behind it takes hours to serve.
2 PRIVATEDENSERETRIEVAL
The setting comprises two interactive parties. The server holds a corpus ofmdocuments, repre-
sented by an embedding matrixA∈Zm×n
p (rowiis the quantized embedding of documenti) and a
content matrixD∈Zm×ℓ
p (rowiis the content of documenti, padded to lengthℓ). The client holds
a private query embeddingx∈Zn
p. Embeddings are quantized fromRtoZ pby scalar quantization
(Appendix A.6). This is further discussed in §4. This abstracts a common deployment pattern in
RAG and enterprise search: the client trusts the embedding model that producedx, but does not trust
the retrieval service with the confidentiality of the query or the integrity of the returned evidence.
Definition 1(Private Dense Retrieval).Aprivate dense retrieval protocolallows the client to (1) ob-
tain the similarity scoress=A·x∈Zm
p, and (2) retrieve the documentD i∗fori∗= arg max isi,
subject to:
1.privacy: the server learns no information aboutxori∗beyond public parameters; and
2.integrity: the client accepts only scores and documents consistent with the server’s com-
mitted database, except with negligible probability.
The server is actively malicious: it may deviate arbitrarily from the protocol to learn the query or
to alter the result. As in standard PIR, we do not require database privacy against the client; our
confidentiality goal concerns the client’s query. The definition extends to top-kretrieval (§2).
One primitive suffices.Both requirements of Definition 1 reduce to verifiable multiplication of a
public plaintext matrix by an encrypted vector. The primitive provides privacy (ciphertexts reveal
nothing about the encrypted vector by semantic security under the GLWE assumption), soundness
(no server strategy makes the client accept anything but the product with the committed matrix,
except with negligible probability) and succinctness (digest and proof are small, and verifying is far
cheaper than computing the product). Formal definitions are in Appendix A.4; Tremblay Thibault
3

Preprint.
et al. (2026) give a concrete instantiation, which our protocol treats as a module. Improvements to
the primitive thus transfer directly to our protocol.
Protocol.The protocol runs two rounds of the same sub-protocol (Figure 1). Once per database,
the server commits toAandDand publishes the digests, which remain fixed for the database’s
lifetime. Updates require recomputing only the affected commitment, a cost amortized over many
queries. In Round 1, the client sendsc 1←Enc(sk1,x)and the server returnsc′
1←A·c 1with a
proofπ 1against the committedA. From this the client decrypts the exact score vector and computes
i∗←arg max isilocally. In Round 2, the client sendsc 2←Enc(sk2,ei∗), fore i∗∈ {0,1}mthe
the one-hot encoding ofi∗. The server returnsD⊤·c2with a proofπ 2against the committedD, from
which the client decryptsD i∗. Each round uses a fresh key and the client aborts unless verification
succeeds. The rounds are linked only throughi∗, which is computed locally. The server sees two
semantically secure ciphertext vectors under independent keys and learns no information aboutxor
i∗beyond public parameters. The client’s Round-1 work, namely downloadingmciphertext slots,
decrypting them and scanning for the maximum, is linear inm. We discuss scaling in §5.
Verification before decryption.That the client verifies before decrypting is essential to security.
If it decrypted an unverified response, its observable behavior (abort, retry, or which document it
fetches in Round 2) would become a function of the plaintext. This is the side channel that reaction
attacks exploit (Chillotti et al., 2016). Verifying first makes that behavior depend only on the proof’s
validity, which the server already knows.
Theorem 1(Informal; proof sketch in Appendix C).If the primitive satisfies the properties of §2,
the protocol realizes Definition 1: an actively malicious server learns no information aboutxori∗
beyond public parameters, and cannot cause the client to accept scores or a document inconsistent
with the committed database, except with negligible probability.
Top-kretrieval.RAG pipelines typically consumek >1passages. After Round 1 the client
knows the full score vector, so it can select anykindices and fetch each with an independent Round-
2 query, multiplying the second round’s workload byk. Batching thekqueries into one server
request multiplies communication bykbut barely increases latency, since the online server work
of Tremblay Thibault et al. (2026) is dominated by a step independent of the number of batched
queries (Appendix D).
3 RELATEDWORK
Encrypted and verifiable retrieval.Table 1 situates our protocol relative to prior work. We take
dense retrieval by maximum inner product search (Karpukhin et al., 2020; Reimers & Gurevych,
2019; Ni et al., 2022) as given and make the query private and the result verifiable. The natural tool
for hiding the query is HE, as PPMI (Bae et al., 2025) does for private LLM interaction. Tiptoe (Hen-
zinger et al., 2023) and PIR-RAG (Wang et al., 2025) instead search published cluster centroids lo-
cally and fetch documents by private information retrieval (PIR) (Chor et al., 1998), for which prac-
tical single-server lattice-based constructions include XPIR (Aguilar-Melchor et al., 2016). Other
systems perform the search itself privately: SANNS (Chen et al., 2020), Servan-Schreiber et al.
Table 1: Private and verifiable retrieval systems.Exactness: scores are exact inner products rather
than approximate search.Malicious security: the stated privacy and/or integrity guarantees hold
against an actively deviating server.
Privacy Integrity Exactness Malicious security
Tiptoe (Henzinger et al., 2023)✓× × ×
PIR-RAG (Wang et al., 2025)✓× × ×
PPMI (Bae et al., 2025)✓×✓×
Compass (Zhu et al., 2024)✓× × ×
VeriRAG (Lin et al., 2026)×✓×✓
zkRAG (Jiang et al., 2026)×✓×✓
Ours✓ ✓ ✓ ✓
4

Preprint.
(2022), Compass (Zhu et al., 2024) and BiSON (Das et al., 2026) run approximatek-NN under two-
party computation or oblivious RAM, at second-to-millisecond latencies on million- to billion-scale
corpora. These systems assume an honest-but-curious server, or two non-colluding servers, and do
not provide integrity against an actively deviating server. In interactive encrypted protocols, such
deviations can additionally enable reaction attacks that compromise privacy (Chillotti et al., 2016).
Conversely, VeriRAG (Lin et al., 2026) and zkRAG (Jiang et al., 2026) prove in zero knowledge
that the server ran an approximate search over a committed database, but send the query in the clear.
Our work instead follows the verifiable-computation-over-ciphertext paradigm (Bois et al., 2021;
Ganesh et al., 2023; Chatel et al., 2024; Tremblay Thibault et al., 2026), enabling query privacy and
retrieval integrity while preserving exactness.
Low-bit embeddings and the retrieval-metric/downstream gap.Several methods have been
proposed in recent years for compressing embeddings for more efficient downstream tasks, including
binary hashing (Yamada et al., 2021), jointly learned product quantization (Zhan et al., 2021), Ma-
tryoshka representation learning (Kusupati et al., 2022), int8-trained commercial encoders (Reimers,
2024) and quantization-aware open ones (Yu et al., 2024). Clipping the quantization range to a per-
centile of the values, rather than to their maximum, is standard in post-training quantization (Banner
et al., 2019). That literature primarily optimizes memory footprint and search efficiency, with rep-
resentation size scaling linearly with bit-width. In our encrypted setting, every bit consumes a fixed
share of a noise budget that only larger cryptographic parameters can compensate for (§4.2), which
imposes a different cost model. Separately, ranking metrics are known to predict end-task RAG
accuracy poorly, as retrieval and generation quality decouple (Salemi & Zamani, 2024; Cuconasu
et al., 2024). Section 4.4 studies this decoupling as a function of bit-width, which also sets cryp-
tographic cost, allowing us to select parameters based on downstream rather than retrieval quality
alone.
4 EXPERIMENTALANALYSIS
The protocol preserves exact dense retrieval over quantized embeddings: the only approximation is
theb-bit quantization itself, while all subsequent arithmetic is computed exactly inZ p, withp≥
2NB2. In fact, the GLWE plaintext modulusp(cf. §A.2) must be large enough so thatZ pcontains
the quantized embeddings and their inner products, without overflow. A largerbrequires a larger
p, which drives up cryptographic parameters. Holding the retrieval task and security target fixed,
the bit-widthbis therefore the key parameter coupling retrieval quality and cryptographic cost. Our
experiments therefore ask how farbcan be reduced before the cryptographic savings cease to justify
the loss in retrieval and downstream quality. We measure this trade-off holistically: §4.1 measures
retrieval quality across bit-widths and corpus sizes, §4.2 gives the conversion from bit-width to
cryptographic cost, §4.3 the concrete costs at scale and §4.4 the downstream performance.
Corpora.We evaluate on two biomedical retrieval corpora, chosen to match the setting of §1:
SciFact(Wadden et al., 2020), scientific claim verification over biomedical abstracts (m=5,183
documents,300test claims), andNFCorpus(Thakur et al., 2021), medical retrieval pairing nutrition
and health queries with linked PubMed documents (m=3,633,323queries). Both are BEIR (Thakur
et al., 2021) tasks, so floating-point baselines are comparable with published numbers. The retrieval
unit of the protocol is a chunk of at most1,536bytes:512matrix cells each containing3bytes
(cf. Table 3). This allows more than half of documents to fit into a single chunk, all the while
keeping the cost of the second round low. We cut at sentence boundaries with the title repeated in
every chunk. This accommodates the clinical references of §1, which are typically several pages
long. The downstream evaluation of §4.4 addsBioASQ(Krithara et al., 2023) for biomedical factoid
question answering, and the scale study of §4.3 usesNatural Questions(Kwiatkowski et al., 2019),
which also serves §4.4 as an out-of-domain control. BioASQ and Natural Questions are passage
collections already fitting within one chunk. Retrieval quality in §4.1 is measured on the chunked
corpora, ranked per chunk and scored per document, so that the relevance labels and the floating-
point baselines remain the published document-level ones; Appendix F gives the same table on
whole abstracts.
5

Preprint.
4.1 RETRIEVAL QUALITY UNDER QUANTIZATION
We evaluate six embedding models of similar performance on both corpora, so that training for low-
bit readout is the variable. Two were trained for low-bit readout: Cohereembed-english-v3.0,
trained for int8 and binary output (Reimers, 2024), and Snowflakearctic-embed-l-v2.0,
whose training was quantization-aware (Yu et al., 2024; Snowflake, 2024). Four were not:
bge-large-en-v1.5(Xiao et al., 2023),e5-large-v2(Wang et al., 2022),gte-large(Li
et al., 2023) andmxbai-embed-large-v1(Li & Li, 2023; Mixedbread AI, 2024). The last is
a control: it advertises int8 and binary use without claiming quantization-aware training. For each
b∈ {2, . . . ,8}we run the full two-round protocol. Table 8 reports, forb∈ {2,3,4,8}, the frac-
tion of floating-point nDCG@10 (Järvelin & Kekäläinen, 2002; Manning et al., 2008) each encoder
retains. Appendix F gives every bit-width on whole abstracts (Table 12) and top-1 agreement at
b∈ {3,4}(Table 9). At a high level, the results show that down tob=4, all encoders retain a high
percentage of retrieval quality compared to their floating-point counterparts.
Appropriate quantization.Ab-bit symmetric quantizer maps a chosen magnitude, the scaleτ,
to the top levelq max= 2b−1−1and rounds every coordinate to the nearest multiple of the step
τ/qmax, so every coordinate within[−τ, τ]is off by at most half a step. Max-abs scaling, the usual
choice, sets the scale to the largest coordinate of the corpus matrix. BERT-family encoders carry
one coordinate holding over5%of the squared norm of every embedding, and that one coordinate
makes the step up to4×coarser than the other1023need. Appendix F shows the effect: BGE-large
clipped atb=3retains97and99%of floating-point nDCG@10 on SciFact and NFCorpus, where
max-abs retains77and73%atb=3and only reaches97and96%atb=4. We thus use clipping
quantization for our experiments.
Quality at scale.A larger index gives every query more near-misses to lose to (Reimers &
Gurevych, 2021), so we repeat the sweep on Natural Questions from104passages to the full2.68M
with five encoders, one of them trained for low-bit readout (Table 13, Appendix F). Atb=4four
of the five retain at least98.8%of floating-point Recall@5 at every size, and e5 retains97.3%at
2.68M. Atb=3arctic, BGE and mxbai retain at least98.4%while gte and e5 fall to94.5and92.0%
at2.68M. Atb=2the loss grows with the corpus, from1to4%at104to10to23%at2.68M. The
lead of the quantization-aware encoder grows with the corpus: arctic retains0.8to2.7points more
Recall@5 than the conventional four at104and2.1to12.8points more at2.68M. As such, retrieval
quality places the operating point atb=3. There, five of the six encoders lose1to3%of nDCG@10
(e5 loses6%on NFCorpus), and for arctic, BGE and mxbai the loss stays small as the corpus grows.
4.2 QUANTIZATION ROBUSTNESS LOWERS CRYPTOGRAPHIC COST
In this section we formalize how the bit-widthbsets the cryptographic parameters, and thus the
cost, of Round 1. For both rounds, we fix the ciphertext modulusq= 264−232+ 1, at which the
primitive of Tremblay Thibault et al. (2026) is most efficient, and targetλ≥128bits of security
and a decryption failure probability of2−64per coefficient.
Theorem 2(Correctness of Round 1).LetA∈Zm×Nandx∈ZNhave entries of absolute value
less thanB, letp≥2NB2and letc←Enc(sk,x)be a GLWE encryption with ciphertext modulus
q, plaintext moduluspand Gaussian noise of standard deviationσ. For everyr >0, provided
q/σ≥2r√
N B p, each coordinate ofDec(sk,A·c)equals that ofA·xover the integers, except
with probabilityε(r) := 2 exp(−r2/2)/(r√
2π).
We defer the proof to Appendix A.3. In Round 1 the inner dimensionNis the embedding dimension
d= 210, the quantized entries lie in[−(2b−1−1),2b−1−1]soB= 2b−1, and we takep= 2NB2=
22b+9. Atq= 264−232+ 1andr= 9.16, for whichε(r)≤2−64, Theorem 2 thus makes Round 1
correct providedσ≤σ max(b)where
log2σmax(b) = log2q−log2(2r)−1
2log2N−(b−1)−(2b+ 9)≈46.8−3b.(1)
Every added bit of quantization thus costs three bits of noise budget, one through the entry bound
Band two through the plaintext modulusp. Security pulls in the other direction and calls for more
noise. It admits no closed form, so we evaluate it with the lattice estimator (Albrecht et al., 2015)
atσ=σ max(b), for a ternary secret and GLWE dimensions210and211(Table 2). Dimension210,
6

Preprint.
Table 2: Quantization-cryptography exchange rate: Round 1 parameters atq= 264−232+ 1,
κ= 1, ternary secret and2−64failure probability.λ 10andλ 11are the lattice-estimator security at
σ=σ max(b)for LWE dimensionsd= 210and211, anddthe smallest dimension at128bits.
b p σ max λ10 λ11 d
2213240.8138284210
3 215237.8122 253 211
4 217234.8110 227 211
5 219231.8100 206 211
8 225222.879 161 211
which matches the embedding dimension, reaches128bits atb= 2only; everyb≥3must double
the GLWE dimensionκd, by takingd= 211or rankκ= 2. As such, the cryptography admits two
regimes rather than a cost per bit: the smaller ring atb=2, and one doubling, paid once, fromb=3
on. We measure what this doubling costs in §4.3.
4.3 CONCRETE COSTS OF PRIVATE DENSE RETRIEVAL
In what follows we quantify the cost of a private and verifiable query, isolate the share of that price
paid for malicious security and identify the corpus sizes and workflows the protocol can serve today.
Measured cost.Table 3 reports the cost of the full protocol on the evaluation corpora and at scale,
at both parameter sets of §4.2. Atb=2the server time is of1.7s on SciFact and on NFCorpus, and
of2.5s fromb=3on, once the ring is doubled. At106chunks the two are of3.0and5.4minutes.
Round 2 multipliesD⊤∈Z512×m
p where Round 1 multipliesA∈Zm×1,024
p . Both matrices grow
linearly inmand the prover’s cost follows this trend. As such, Round 2 costs≈0.5×Round 1
at every corpus size atb=2. Per round, the client uploads⌈N/d⌉(κ+1)dlog2qbits of ciphertext
and downloads⌈m′/d⌉(κ+1)dlog2qbits plus the proof, wherem′is the output length of that
round. Over both rounds, communication is of528KiB on SciFact and496KiB on NFCorpus at
b=2, proofs included, and20KiB more once the ring is doubled. Most of the server time is spent
computing the proof of correct computation. On the SciFact Round-1 shape atd= 210, the matrix-
vector product over ciphertexts takes15ms without proof generation and1.18s with it. Detecting a
malicious server rather than trusting it thus costs a factor of77.
Table 3: Measured cost on the evaluation corpora and at scale, atb=2(Round 1 atd= 210) and
fromb=3on (Round 1 atd= 211); Round 2 runs atd= 212withℓ= 512(Table 5). Server time
is measured on48AMD EPYC 9654 cores; client time is measured on a single core. Prices at the
on-demand rate of a48-corehpc7a.24xlargeEC2 instance ($7.20/h).
b=2/b≥3
Corpus Round 1:ARound 2:D⊤Client Server $/query
NFCorpus (5,761chunks)5,761×1,024 512×5,761 510ms /914ms1.7s /2.5s0.003/0.005
SciFact (7,510chunks)7,510×1,024 512×7,510 510ms /914ms1.7s /2.5s0.003/0.005
m= 104104×1,024 512×104907ms /1.7s3.0s /5.0s0.006/0.010
m= 105105×1,024 512×1056.3s /12.5s21.7s /33.8s0.043/0.068
m= 106106×1,024 512×10650.4s /1.7min3.0min /5.4min0.364/0.650
The deployable frontier.Table 3 presents the cost of a private dense retrieval query under our
protocol for bothb∈ {2,3}. By §4.1,b=3is the smallest bit-width that keeps retrieval quality,
so the following assumesb=3, i.e. the larger GLWE parameters. Today, UpToDate, the curated
reference clinicians consult first, holds over12,000topics (Wolters Kluwer, 2026b), each the length
of a review article, i.e., of the order of2to5×105chunks in all. A private query on a corpus this
size costs one to three minutes and is affordable today. On the other hand, PubMed, the literature
behind UpToDate, holds over40million abstracts (National Library of Medicine, 2026), one or
two chunks each. A query on such a large corpus would take4to6hours of server time at this
throughput and incur communication costs of approximately1GB. We judge this is too costly for
7

Preprint.
the clinical setting, even in the deferred-response context. Two things can bring the cost down:
faster cryptographic machinery, and encoders that keep their quality atb=2. The former lies with
the cryptographic community. The latter is a matter of representation learning, and by §4.2 it is
worth the one ring doubling, i.e. the factor of1.5to1.8between the two columns of Table 3. As
such, atb=3the protocol serves a corpus of a few hundred thousand chunks in one to three minutes
and106in five, which covers a curated clinical reference and not the literature behind it.
4.4 DOWNSTREAM TASK QUALITY UNDER QUANTIZATION
In the RAG deployments we target, what matters is the accuracy of the answer a language model
produces from the retrieved passages rather than the relevance ranking of these passages. The two
need not degrade together, since a reader consumingkpassages tolerates a reordering provided the
evidence it needs appears somewhere in its context (Salemi & Zamani, 2024; Cuconasu et al., 2024).
We measure downstream accuracy as a function ofband locate the smallest bit-width it tolerates.
Methodology.We follow the standard end-task protocol for retrieval-augmented genera-
tion (Karpukhin et al., 2020; Lewis et al., 2020; Petroni et al., 2021): the retriever returns the top-k
passages, a fixed reader answers from them, and only the retrieved set changes between the condi-
tions we compare. A question-answering response is correct if it contains a reference answer (Mallen
et al., 2023; Asai et al., 2024), and a claim-verification response if it matches the label. We test each
bit-width against floating point with an exact McNemar test on paired per-query outcomes. Since
a non-significant difference is not evidence of equivalence, we further test equivalence within±5
points by two one-sided tests (TOST) on the same paired differences (Appendix E).
Setup.We use three endpoints and two encoders, one from each group of §4.1.SciFactis eval-
uated on its original task: the reader receives a claim and the top-kretrieved abstracts and outputs
SUPPORT, CONTRADICTor NOINFO, scored against the human labels on the188claims with ev-
idence in the corpus, atk=5andk=1.BioASQis biomedical factoid QA over a corpus of35,454
passages, scored by containment of a reference answer.Natural Questionsis the out-of-domain con-
trol at scale, answered over the full2.68M-passage corpus of §4.1. Each endpoint has300queries.
We test the performance limits of the reader by measuring its performance in the closed book setting,
where it answers from its own knowledge with no passages, and in the gold context setting, where
it receives the labelled relevant passages in place of the retrieved ones. Retrieval uses the clipped
quantizer with BGE and arctic, and only the retrieved set varies withb, so any change in accuracy
is attributable to quantization. The reader isphi-4(14B) (Abdin et al., 2024). Appendix E repeats
every measurement withOLMo-2-7B(OLMo Team, 2025) andQwen2.5at7B and72B (Qwen
Team, 2025), and gives the prompts and corpora.
Downstream quality holds tob=3and not tob=2.Table 4 shows the result. On closed book,
the reader reaches19%on BioASQ,31%on NQ and59%on SciFact, with gold passages68,74
and81%, and with floating-point retrieval it sits between the two, close to the gold row. The effects
of quantization below are to be read against that span of23to49points. No cell shows a significant
drop from floating point atb=4orb=3, and every cell is equivalent to floating point within5points
at both bit-widths. This holds at2.68M passages. The reason is that the RAG endpoint consumes
recall rather than rank. More precisely, atb=3the top-5set of BGE on NQ agrees with the floating-
point one on82%of positions, yet it contains a relevant passage for70%of questions, as in floating
point. Further, §4.1 showed Recall@5 retained at every corpus size. The three other readers agree:
over all four,32of32reader-endpoint-encoder combinations are equivalent to floating point atb=4
and30of32atb=3, with a single significant drop of3.3points (Table 6).
Atb=2, accuracy drops significantly in20of32combinations, by3to9points, and only9remain
equivalent to floating point within5points. Again, poor recall is responsible for this drop in per-
formance. On NQ, the top-5set of BGE contains a relevant passage for60%of questions atb=2
against70%atb=3, and the reader answers UNKNOWNmore often rather than answering wrongly.
The loss grows with the corpus: Recall@5 retained atb=2falls from98to86%for BGE and from
99to90%for arctic between104and2.68M passages (Table 13). Interestingly, the loss is uneven.
It is largest atk=1, where a single reordering at the top removes the only passage the reader sees,
at4to9points for both encoders. Atk=5it depends on the encoder: on SciFact, BGE loses3to
8

Preprint.
Table 4: Downstream accuracy versus bit-width with the clipped quantizer (readerphi-4,n=300
queries per endpoint). SciFact: accuracy on the188evidence-bearing claims; BioASQ and NQ:
answer containment, NQ retrieving over the full2.68M-passage corpus. Closed book: the reader
sees no passages; gold context: it sees the labelled relevant passages (at mostk).Boldmarks a
significant drop from floating point (exact McNemar,p <0.05). Other readers: Appendix E.
SciFactk=5SciFactk=1BioASQk=5NQk=5,2.68M
bBGE arctic BGE arctic BGE arctic BGE arctic
closed book0.585 0.585 0.187 0.313
gold context0.814 0.809 0.680 0.740
float0.734 0.713 0.766 0.739 0.633 0.647 0.643 0.643
8 0.734 0.718 0.771 0.739 0.633 0.643 0.657 0.637
6 0.729 0.713 0.771 0.739 0.633 0.640 0.650 0.640
4 0.723 0.718 0.771 0.739 0.637 0.637 0.650 0.627
3 0.707 0.713 0.761 0.755 0.643 0.640 0.623 0.640
20.6760.6970.686 0.6810.6130.607 0.5730.600
6points where arctic loses at most2.1, with no significant drop for any of the four readers. This
shows that the loss atb=2is a property of how the encoder was trained.
Chunking the SciFact corpus as in §4 costs the conventional encoder one bit and the quantization-
aware one none (Table 7, Appendix E). With BGE atk=5the loss reaches4.3points atb=3, where
arctic stays within5points of floating point down tob=2on this smaller corpus.
As such, downstream accuracy places the operating point of private RAG atb=3at every corpus
size we measured, one bit above theb=2at which §4.2 halves the GLWE parameters required for
sufficient security. That frontier bit is decided by the encoder: with a conventional encoder the loss
atb=2is of3to9points, with a quantization-aware one it is within2.1points atk=5on SciFact.
Whether private RAG runs on the smaller GLWE parameters is thus a property of the encoder.
5 DISCUSSION
Limitations.The protocol is a drop-in retrieval layer for deployments in which a licensed corpus
is hosted by an untrusted service, the queries are sensitive and an answer is expected in minutes.
Importantly, our contribution is not an end-to-end privacy-preserving RAG pipeline: we assume the
embedding model and the readers run locally or in a trusted environment, and leave a fully private
pipeline, which current technology does not yet allow, to future work. In terms of scale, three
constraints bound our protocol. The first is the client: the Round-1 response, the Round-2 query and
the client’s decryption and argmax are all linear inm. Atm= 106, Table 3 gives one to two minutes
of sequential client work and roughly16MB in each direction. The client is kept single-threaded
to model a computationally weak device. The second is the encoder, which decides whether a task
can go belowb=3. Our evidence is a controlled comparison of six encoders, of which only two
were trained for low-bit readout. What they change is the score margin relative to the rounding
step (§4.1); whether a training objective can widen it enough forb=2at scale is an open problem
we hand to representation learning. Very recent advances in this regard are promising (Wang et al.,
2026). The third constraint is the cryptographic primitive: verifiable HE throughput limits web-scale
deployment, but our modular design directly absorbs any improvement to it.
6 CONCLUSION
We have formulated private dense retrieval as a problem of privacy and integrity against a malicious
server. We have reduced the underlying computations to verifiable multiplication of a committed
matrix by an encrypted vector and used the resulting protocol to measure what privacy costs. With
a clipped quantizer, retrieval quality and downstream accuracy hold atb=3for six encoders and
four readers on corpora of up to2.68M passages. A private, verified query over a corpus the size
of a clinical reference is then of one to three minutes. At128bits of security, onlyb=2admits the
smaller GLWE parameters;b≥3costs a factor of1.8in server time. Which encoders keep their
9

Preprint.
quality below three bits is set by their score margin relative to the quantization step, and only those
trained for low-bit readout do so atb=2. Lowering the cost of privacy in dense retrieval is thus not
only a cryptography problem, but also one of representation learning.
AI USESTATEMENT
In this work, we used generative AI tools for feedback on research methodology. We have not used
generative AI tools to generate synthetic data sets, help develop theoretical models or conceptual
frameworks, formulate mathematical claims, provide critical ingredients for proving mathematical
claims, assist in the writing of proofs, propose or refine hypotheses, implement methods, assist with
translation, clean and reformat datasets, support qualitative and thematic data analysis, or interpret
results. Additionally, we used generative AI tools to edit software code, generate tables and identify
relevant literature. We have reviewed all AI-assisted work. We take responsibility for the final
content of this work, including text, claims or artifacts produced with the aid of generative AI.
REPRODUCIBILITYSTATEMENT
All retrieval-side results use public corpora (SciFact, NFCorpus, Natural Questions, BioASQ)
and publicly available encoders. The downstream readers are open-weight models run with
structured decoding, so every number in §4.4 is reproducible without access to an API. The
quantization code and the per-bit-width sweep outputs are available athttps://github.
com/tremblaythibaultl/pdr-artifact, and the protocol implementation athttps:
//github.com/tremblaythibaultl/vpir. Prompts are fixed across bit-widths within a
reader and are given in Appendix E. The cryptographic parameters, the correctness constraint and
the security estimate are stated in closed form in §4.2 and Appendix A.2, so Table 2 is reproducible
without running the protocol. Timings are single-node measurements on a named processor with
a fixed thread count, and the benchmark binary asserts verification and exact decryption at every
shape.
ETHICSSTATEMENT
This work aims to reduce the exposure of sensitive queries and is motivated by a concrete disclosure
risk in clinical decision support. The corpora we use are public research datasets and contain no pa-
tient data. We emphasize that query privacy against the retrieval provider does not guarantee privacy
over the rest of the technological stack. A protocol such as ours could be invoked to argue compli-
ance while leakage persists elsewhere. The guarantee is exactly that of Definition 1 and no broader.
Further, the integrity guarantee is relative to a committed digest. Deploying the protocol without in-
dependent attestation of that digest by peers or auditors offers the appearance of verification without
its substance.
ACKNOWLEDGMENTS
The authors would like to thank Léo Gagnon, Romane Asselin and Charlie Gauthier for insightful
discussions. This work was supported by the Fonds de recherche du Québec - Nature et technologies
(FRQNT) through granthttps://doi.org/10.69777/2006324.
REFERENCES
Marah Abdin, Jyoti Aneja, Harkirat Behl, Sébastien Bubeck, Ronen Eldan, Suriya Gunasekar,
Michael Harrison, Russell J. Hewett, Mojan Javaheripi, Piero Kauffmann, et al. Phi-4 techni-
cal report.arXiv preprint arXiv:2412.08905, 2024.
Agence France-Presse. Hackers steal medical details of 15 million in
france, 2026. URLhttps://www.france24.com/en/live-news/
20260227-hackers-steal-medical-details-of-15-million-in-france.
10

Preprint.
Carlos Aguilar-Melchor, Joris Barrier, Laurent Fousse, and Marc-Olivier Killijian. XPIR: Private in-
formation retrieval for everyone. 2016(2):155–174, April 2016. doi: 10.1515/popets-2016-0010.
Martin R. Albrecht, Rachel Player, and Sam Scott. On the concrete hardness of learning with
errors.J. Math. Cryptol., 9(3):169–203, 2015. URLhttp://www.degruyter.com/view/
j/jmc.2015.9.issue-3/jmc-2015-0016/jmc-2015-0016.xml.
American Medical Association. 2026 physician survey on augmented intelligence. AMA Center
for Digital Health and AI, March 2026. URLhttps://www.ama-assn.org/system/
files/physician-ai-sentiment-report.pdf.
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-RAG: Learning
to retrieve, generate, and critique through self-reflection. InThe Twelfth International Conference
on Learning Representations, 2024.
Yubeen Bae, Minchan Kim, Jaejin Lee, Sangbum Kim, Jaehyung Kim, Yejin Choi, and Niloofar
Mireshghallah. Ppmi: Privacy-preserving llm interaction with socratic chain-of-thought reasoning
and homomorphically encrypted vector databases, 2025. URLhttps://arxiv.org/abs/
2506.17336.
Ron Banner, Yury Nahshan, and Daniel Soudry. Post-training 4-bit quantization of convolutional
networks for rapid-deployment. InAdvances in Neural Information Processing Systems, vol-
ume 32, 2019.
Alexandre Bois, Ignacio Cascudo, Dario Fiore, and Dongwoo Kim. Flexible and efficient verifiable
computation on encrypted data. InPublic-Key Cryptography (PKC), pp. 528–558, 2021. doi:
10.1007/978-3-030-75248-4_19.
Sylvain Chatel, Christian Knabenhans, Apostolos Pyrgelis, Carmela Troncoso, and Jean-Pierre
Hubaux. VERITAS: Plaintext encoders for practical verifiable homomorphic encryption. In
Proceedings of the 2024 ACM SIGSAC Conference on Computer and Communications Security
(CCS), pp. 2520–2534, 2024. doi: 10.1145/3658644.3670282.
Hao Chen, Ilaria Chillotti, Yihe Dong, Oxana Poburinnaya, Ilya Razenshteyn, and M. Sadegh Riazi.
SANNS: Scaling up secure approximatek-nearest neighbors search. In29th USENIX Security
Symposium (USENIX Security), pp. 2111–2128, 2020.
Ilaria Chillotti, Nicolas Gama, and Louis Goubin. Attacking FHE-based applications by software
fault injections. Cryptology ePrint Archive, Paper 2016/1164, 2016. URLhttps://eprint.
iacr.org/2016/1164.
Benny Chor, Oded Goldreich, Eyal Kushilevitz, and Madhu Sudan. Private information retrieval.
Journal of the ACM, 45(6):965–981, 1998. doi: 10.1145/293347.293350.
Florin Cuconasu, Giovanni Trappolini, Federico Siciliano, Simone Filice, Cesare Campagnano,
Yoelle Maarek, Nicola Tonellotto, and Fabrizio Silvestri. The power of noise: Redefining re-
trieval for RAG systems. InProceedings of the 47th International ACM SIGIR Conference
on Research and Development in Information Retrieval (SIGIR), pp. 719–729, 2024. doi:
10.1145/3626772.3657834.
Sankha Das, Rohan Ravi, Nishanth Chandran, and Divya Gupta. BiSON: Billion-scale oblivious
nearest-neighbor search in milliseconds. Cryptology ePrint Archive, Paper 2026/1343, 2026.
URLhttps://eprint.iacr.org/2026/1343.
Chaya Ganesh, Anca Nitulescu, and Eduardo Soria-Vazquez. Rinocchio: SNARKs for ring arith-
metic.Journal of Cryptology, 36(4):41, 2023. doi: 10.1007/s00145-023-09481-3.
Craig Gentry. Fully homomorphic encryption using ideal lattices. InProceedings of the Forty-
First Annual ACM Symposium on Theory of Computing, STOC ’09, pp. 169–178, New York,
NY , USA, 2009. Association for Computing Machinery. ISBN 9781605585062. doi: 10.1145/
1536414.1536440. URLhttps://doi.org/10.1145/1536414.1536440.
11

Preprint.
Kai Greshake, Sahar Abdelnabi, Shailesh Mishra, Christoph Endres, Thorsten Holz, and Mario
Fritz. Not what you’ve signed up for: Compromising real-world llm-integrated applications with
indirect prompt injection.Proceedings of the 16th ACM Workshop on Artificial Intelligence and
Security, 2023. URLhttps://api.semanticscholar.org/CorpusID:258546941.
Alexandra Henzinger, Emma Dauterman, Henry Corrigan-Gibbs, and Nickolai Zeldovich. Private
web search with tiptoe. InProceedings of the 29th Symposium on Operating Systems Princi-
ples, SOSP ’23, pp. 396–416, New York, NY , USA, 2023. Association for Computing Machin-
ery. ISBN 9798400702297. doi: 10.1145/3600006.3613134. URLhttps://doi.org/10.
1145/3600006.3613134.
Kalervo Järvelin and Jaana Kekäläinen. Cumulated gain-based evaluation of IR techniques.ACM
Transactions on Information Systems, 20(4):422–446, 2002. doi: 10.1145/582415.582418.
Yanze Jiang, Xinyang Yang, Xuanming Liu, Yanpei Guo, and Jiaheng Zhang. zkRAG: Efficiently
proving RAG retrieval in zero knowledge. Cryptology ePrint Archive, Paper 2026/709, 2026.
URLhttps://eprint.iacr.org/2026/709.
Vladimir Karpukhin, Barlas O ˘guz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi
Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. In
Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing
(EMNLP), pp. 6769–6781, 2020.
Anastasia Krithara, Anastasios Nentidis, Konstantinos Bougiatiotis, and Georgios Paliouras.
BioASQ-QA: A manually curated corpus for biomedical question answering.Scientific Data,
10(170), 2023. doi: 10.1038/s41597-023-02068-4.
Aditya Kusupati, Gantavya Bhatt, Aniket Rege, Matthew Wallingford, Aditya Sinha, Vivek Ra-
manujan, William Howard-Snyder, Kaifeng Chen, Sham Kakade, Prateek Jain, and Ali Farhadi.
Matryoshka representation learning. InAdvances in Neural Information Processing Systems
(NeurIPS), volume 35, pp. 30233–30249, 2022.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris
Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion
Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav
Petrov. Natural questions: A benchmark for question answering research.Transactions of the
Association for Computational Linguistics, 7:452–466, 2019.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented gener-
ation for knowledge-intensive nlp tasks.Advances in neural information processing systems, 33:
9459–9474, 2020.
Xianming Li and Jing Li. Angle-optimized text embeddings.arXiv preprint arXiv:2309.12871,
2023.
Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, and Meishan Zhang. Towards
general text embeddings with multi-stage contrastive learning.arXiv preprint arXiv:2308.03281,
2023.
Chenqi Lin, Yubo Cui, Zhelei Zhou, Cheng Hong, Yufei Wang, Zhaohui Chen, and Meng Li. Veri-
RAG: Efficient zero-knowledge proofs for verifiable retrieval-augmented generation. Cryptology
ePrint Archive, Paper 2026/637, 2026. URLhttps://eprint.iacr.org/2026/637.
Vadim Lyubashevsky, Chris Peikert, and Oded Regev. On ideal lattices and learning with errors over
rings.J. ACM, 60(6):43:1–43:35, 2013. doi: 10.1145/2535925. URLhttps://doi.org/
10.1145/2535925.
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Hajishirzi.
When not to trust language models: Investigating effectiveness of parametric and non-parametric
memories. InProceedings of the 61st Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pp. 9802–9822, 2023.
12

Preprint.
Christopher D. Manning, Prabhakar Raghavan, and Hinrich Schütze.Introduction to Information
Retrieval. Cambridge University Press, Cambridge, England, 2008. ISBN 978-0-521-86571-5.
Mixedbread AI. mxbai-embed-large-v1 model card.https://huggingface.co/
mixedbread-ai/mxbai-embed-large-v1, 2024. Accessed September 2026.
John Morris, V olodymyr Kuleshov, Vitaly Shmatikov, and Alexander M Rush. Text embeddings
reveal (almost) as much as text. InProceedings of the 2023 Conference on Empirical Methods in
Natural Language Processing, pp. 12448–12460, 2023.
National Library of Medicine. About PubMed, 2026. URLhttps://pubmed.ncbi.nlm.
nih.gov/about/. Accessed September 2026.
NBC News. Most U.S. doctors are quietly using this AI tool. few patients know
about it, 2026. URLhttps://www.nbcnews.com/tech/tech-news/
openevidence-ai-doctor-medical-physician-login-app-what-npi-uptodate-rcna341064.
Jianmo Ni, Chen Qu, Jing Lu, Zhuyun Dai, Gustavo Hernandez Abrego, Ji Ma, Vincent Zhao,
Yi Luan, Keith Hall, Ming-Wei Chang, et al. Large dual encoders are generalizable retrievers. In
Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp.
9844–9855, 2022.
HHS Office for Civil Rights. Standards for privacy of individually identifiable health information.
final rule, 2002.
OLMo Team. 2 OLMo 2 furious.arXiv preprint arXiv:2501.00656, 2025.
Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick Lewis, Majid Yazdani, Nicola De Cao, James
Thorne, Yacine Jernite, Vladimir Karpukhin, Jean Maillard, Vassilis Plachouras, Tim Rock-
täschel, and Sebastian Riedel. KILT: a benchmark for knowledge intensive language tasks. In
Proceedings of the 2021 Conference of the North American Chapter of the Association for Com-
putational Linguistics: Human Language Technologies, pp. 2523–2544, 2021.
Qwen Team. Qwen2.5 technical report.arXiv preprint arXiv:2412.15115, 2025.
Nils Reimers. Cohere int8 & binary embeddings: Scale your vector database to large datasets.
Cohere Blog, 2024. URLhttps://cohere.com/blog/int8-binary-embeddings.
Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bert-
networks. InProceedings of the 2019 conference on empirical methods in natural language
processing and the 9th international joint conference on natural language processing (EMNLP-
IJCNLP), pp. 3982–3992, 2019.
Nils Reimers and Iryna Gurevych. The curse of dense low-dimensional information retrieval for
large index sizes. InProceedings of the 59th Annual Meeting of the Association for Computational
Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume
2: Short Papers), pp. 605–611. Association for Computational Linguistics, 2021. doi: 10.18653/
v1/2021.acl-short.77.
Alireza Salemi and Hamed Zamani. Evaluating retrieval quality in retrieval-augmented generation.
InProceedings of the 47th International ACM SIGIR Conference on Research and Development
in Information Retrieval (SIGIR), pp. 2395–2400, 2024. doi: 10.1145/3626772.3657957.
Sacha Servan-Schreiber, Simon Langowski, and Srinivas Devadas. Private approximate nearest
neighbor search with sublinear communication. InIEEE Symposium on Security and Privacy
(S&P), 2022.
Snowflake. snowflake-arctic-embed-l-v2.0 model card.https://huggingface.co/
Snowflake/snowflake-arctic-embed-l-v2.0, 2024. Accessed September 2026.
Congzheng Song and Ananth Raghunathan. Information leakage in embedding models. InProceed-
ings of the 2020 ACM SIGSAC Conference on Computer and Communications Security (CCS),
pp. 377–390, 2020. doi: 10.1145/3372297.3417270.
13

Preprint.
Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. Beir: A
heterogenous benchmark for zero-shot evaluation of information retrieval models.arXiv preprint
arXiv:2104.08663, 2021.
Louis Tremblay Thibault, Michael Walter, and Jiapeng Zhang. Practical SNARGs for matrix
multiplications over encrypted data. Cryptology ePrint Archive, Paper 2026/027, 2026. URL
https://eprint.iacr.org/2026/027.
David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang, Madeleine van Zuylen, Arman Cohan,
and Hannaneh Hajishirzi. Fact or fiction: Verifying scientific claims. InProceedings of the 2020
Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 7534–7550,
2020.
Baiqiang Wang, Qian Lou, Mengxin Zheng, and Dongfang Zhao. Pir-rag: A system for private
information retrieval in retrieval-augmented generation, 2025. URLhttps://arxiv.org/
abs/2509.21325.
Hongyu Wang, Shuming Ma, Lingxiao Ma, Lei Wang, Wenhui Wang, Li Dong, Shaohan Huang,
Huaijie Wang, Jilong Xue, Ruiping Wang, Yi Wu, and Furu Wei. Bitnet: 1-bit pre-training for
large language models.J. Mach. Learn. Res., 26(1), September 2026. ISSN 1532-4435.
Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Ma-
jumder, and Furu Wei. Text embeddings by weakly-supervised contrastive pre-training.arXiv
preprint arXiv:2212.03533, 2022.
Wolters Kluwer. Health system adoption of Wolters Kluwer UpToDate Ex-
pert AI now reaching approximately 2,500 U.S. hospitals and health systems,
August 2026a. URLhttps://www.wolterskluwer.com/en/news/
health-system-adoption-uptodate-expert-ai-reaching-2500-us-hospitals-health-systems.
Wolters Kluwer. What is UpToDate, 2026b. URLhttps://www.wolterskluwer.com/en/
solutions/uptodate/about. Accessed September 2026.
Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas Muennighoff. C-pack: Packaged resources to
advance general chinese embedding.arXiv preprint arXiv:2309.07597, 2023.
Ikuya Yamada, Akari Asai, and Hannaneh Hajishirzi. Efficient passage retrieval with hashing for
open-domain question answering. InProceedings of the 59th Annual Meeting of the Association
for Computational Linguistics (ACL), pp. 979–986, 2021. doi: 10.18653/v1/2021.acl-short.123.
Puxuan Yu, Luke Merrick, Gaurav Nuti, and Daniel Campos. Arctic-embed 2.0: Multilingual re-
trieval without compromise.arXiv preprint arXiv:2412.04506, 2024.
Jingtao Zhan, Jiaxin Mao, Yiqun Liu, Jiafeng Guo, Min Zhang, and Shaoping Ma. Jointly optimizing
query encoder and product quantization to improve retrieval performance. InProceedings of the
30th ACM International Conference on Information and Knowledge Management (CIKM), pp.
2487–2496, 2021. doi: 10.1145/3459637.3482358.
Jinhao Zhu, Liana Patel, Matei Zaharia, and Raluca Ada Popa. Compass: Encrypted semantic
search with high accuracy. Cryptology ePrint Archive, Paper 2024/1255, 2024. URLhttps:
//eprint.iacr.org/2024/1255.
A PRELIMINARIES
A.1 HOMOMORPHICENCRYPTION
We rely on a linearly homomorphic encryption scheme based on the General Learning With Errors
(GLWE) assumption.
14

Preprint.
A.2 (G)LWEENCRYPTION
Definition 2(GLWE encryption scheme Lyubashevsky et al. (2013)).Letqbe a prime ciphertext
modulus,p < qa plaintext modulus,κthe GLWE dimension parameter andda power of two ring
degree so thatR q:=Z[X]/⟨Xd+ 1⟩is a cyclotomic polynomial ring. We letχ sandχ edenote
the secret key and error distributions overR qrespectively. Finally, we letUdenote the uniform
distribution,R pdenote the message space of the encryption scheme and∆ :=⌊q/p⌉. A GLWE
encryption schemeE := (KGen,Enc,Dec)consists of the following algorithms:
•KGen(1λ): Takes as input a security parameter1λand returns a sampled secret key⃗ s← $
χκ
s.
•Enc(⃗ s, m): Takes as input a secret key⃗ sand a messagem∈ R pand returns a ciphertext
⃗ c∈ Rκ+1
qcomputed as follows:
⃗ a← $U(R q)κ, e← $χe, ⃗ c= (⃗ a,⟨⃗ a,⃗ s⟩+e+ ∆m)∈ Rκ+1
q.
•Dec(⃗ s,⃗ c): Takes as input a secret key⃗ sand a ciphertext⃗ c= (⃗ a, b)∈ Rκ+1
qand returns a
messagem∈ R pcomputed as follows:
m=b− ⟨⃗ a,⃗ s⟩
∆
This encryption scheme is secure under the GLWE assumption for appropriate choices of param-
eters. For a concrete parameter set, its security can be estimated using the widely used lattice
estimator Albrecht et al. (2015).
Table 5 gives the parameter sets we instantiate the protocol with, satisfying the correctness constraint
of Theorem 2 with failure probability at most2−64per coefficient and at leastλ= 128bits of
security for a ternary secret, as estimated with the lattice estimator Albrecht et al. (2015). Round 1
is shown atb= 2, the only bit-width the ringd= 210admits, and atb= 3and4withd= 211.
The remaining rows of Table 2 follow the same pattern (rankκ= 2atd= 210has the same
LWE dimension and the same security, but the implementation we measure supports rank1only).
In each rowσis set at the correctness ceiling of (1), so the security shown is the largest the row
allows. Round 2 encrypts a one-hot vector overmchunks withB=p= 224, so its ceiling is
log2σmax= 64−log2(2r)−1
2log2m−48, i.e.,25.3atm= 213and21.8atm= 220. We use the
minimal widthσ= 3.2, which is correct up tom= 220chunks and gives223bits atd= 212, the
ring at which128bits are reached at that width; beyond220chunks, each bit removed frompbuys
four doublings ofm.
Table 5: GLWE parameters per round: ring dimensiond, rankκ, ciphertext modulusq, plaintext
modulusp, noise standard deviationσand the resulting security parameterλ(ternary secret) and
maximum failure probabilityPr[fail]per coefficient. Round 2 is given form≤220chunks.
Roundd κ q p σ λPr[fail]
Round 1 (b= 2)2101 264−232+ 1 213240.8138 2−64
Round 1 (b= 3)2111 264−232+ 1 215237.8253 2−64
Round 1 (b= 4)2111 264−232+ 1 217234.8227 2−64
Round 22121 264−232+ 1 22421.7223 2−64
The above encryption scheme is additively homomorphic and allows for multiplication by plaintext
elements, provided the noise remains “small”. More precisely, given two ciphertexts⃗ c 1= (⃗ a 1, b1)
and⃗ c 2= (⃗ a 2, b2)encrypting messagesm 1, m2∈ Rprespectively, we have:
• Addition:⃗ c add=⃗ c1+⃗ c2= (⃗ a 1+⃗ a2, b1+b2)is a ciphertext encryptingm add=m 1+m 2.
• Scalar multiplication: For any plaintexty∈ R p,⃗ cscal=y·⃗ c 1= (y·⃗ a 1, y·b 1)is a ciphertext
encryptingm scal=y·m 1.
These properties hold provided the accumulated noise in the resulting ciphertexts remains small
enough to allow for correct decryption, i.e.∥e∥ ∞≤∆/2.
15

Preprint.
A.3 PROOF OFTHEOREM2
Proof.Each scores i=P
jAijxjsatisfies|s i| ≤N(B−1)2< NB2, so the2NB2−1≤pvalues
it can take are distinct modulop. It remains to show that decryption returnss imodp. By linearity,
A·cdecrypts toA·xmodpprovided the noise of every coefficient stays below∆/2in absolute
value. We neglect the rounding of∆ =⌊q/p⌉, which shifts this threshold by at mostp/2, and write
∆/2 =q/2p. The ciphertextccarriesNindependent noise coefficientse 1, . . . , e N∼ N(0, σ2),
and the coefficient ofA·ccarryings ihas noisee′
i=P
j±Aijeπ(j), for a permutationπand signs
fixed by the negacyclic ring structure. As such,e′
iis Gaussian of varianceσ2P
jA2
ij≤NB2σ2,
and the Gaussian tail bound givesPr[|e′
i|> r√
NBσ]≤ε(r). The inequalityq/σ≥2r√
NBp
implies thatr√
NBσ≤q/(2p), so every coordinate is correct except with probabilityε(r).
A.4 SECUREMATRIX-VECTORMULTIPLICATION
We assume the existence of an efficient protocol for verifiable plaintext matrix–encrypted vector
multiplication. Such a primitive can be instantiated with e.g. Tremblay Thibault et al. (2026). The
protocol operates as follows:
1. The client encodes its inputx= (x 1, . . . , x N)∈ZN
pinto ring elements and encrypts each
component, obtaining a ciphertext vectorcwhich is sent to the server.
2. The server holds a plaintext matrixM∈Zm×N
p . It computes the matrix-ciphertext product
row-wise, yielding a result ciphertext vectorc′of dimensionm, which is returned to the
client.
3. The client decryptsc′to obtainM·x.
Definition 3(Passive security).The protocol ispassively secureif the server’s view (the ciphertext
vectorc) reveals no information aboutxbeyond public parameters and dimensions.
Definition 4(Malicious security / Verifiability).The protocol ismaliciously secureif, in addition to
passive security, the client can verify that the server computedM·xcorrectly. That is, a malicious
server cannot convince the client to accept an incorrect resulty̸=M·xexcept with probability
negl(λ).
The malicious security property is achieved via a verifiable computation mechanism which allows
the client to check the server’s computation without re-doing it.
A.5 EMBEDDING-BASEDRETRIEVAL
Modern retrieval systems represent documents and queries as dense vectors (embeddings) inRn
using neural network encoders (Reimers & Gurevych (2019); Ni et al. (2022)). Given a corpus ofm
documents, each documentiis mapped to an embeddinga i∈Rn. These embeddings are collected
as rows of a matrixA∈Rm×n. For a query with embeddingx∈Rn, the relevance of each
document is measured by the inner product⟨a i,x⟩, and the top-kdocuments with highest scores are
retrieved.
A.6 QUANTIZEDEMBEDDINGS
To operate overZ prather thanRwe useclipped symmetric scalar quantization. Letq max= 2b−1−1
and letτbe the99th percentile of|A|over all entries of the embedding matrix. Each coordinatea
is mapped toclip(⌊a q max/τ⌉,−q max, qmax), so the1%of coordinates beyondτsaturate. A query
vector is quantized the same way with its own percentile. Every entry lies in[−q max, qmax].
The usual max-abs rule maps the single largest coordinate toq max. BERT-family encoders carry
one coordinate holding5to6%of the squared norm of every embedding, so under max-abs that
coordinate sets a step4×coarser than the bulk of the coordinates need. Clipping at a percentile
is standard in post-training quantization (Banner et al., 2019). Table 10 compares clipping with
max-abs on every encoder. The99.9th and95th percentiles land within a point of the99th on both
corpora.
16

Preprint.
Compatibility with secure computation.Quantized embeddings are natively integers inZ p, re-
quiring no floating-point emulation, and the inner product of quantized vectors is computed exactly
by the protocol. How closely it preserves the floating-point ranking is the subject of §4.1 and §4.3.
B PROTOCOLMESSAGEFLOW
Figure 2 gives the message flow of the two-round protocol of §2.
Client Server
sk1←$KGen(1λ)
c1←Enc(sk1,x)c1
c′
1←A·c 1
π1←Prove(A,c 1,c′
1)(c′
1, π1)
Verify(π 1)
s←Dec(sk1,c′
1)
i∗←arg max isi
local computation
sk2←$KGen(1λ)
c2←Enc(sk2,ei∗)c2
c′
2←D⊤·c2
π2←Prove(D⊤,c2,c′
2)(c′
2, π2)
Verify(π 2);Dec(sk2,c′
2)Round 1 Round 2
Figure 2: The private dense retrieval protocol. Round 1 retrieves encrypted similarity scores,
Round 2 privately fetches the top document; both use the same verifiable matrix-vector sub-protocol,
and the client verifies each proof before decrypting.
C SECURITY OF THEPROTOCOL
We sketch the proof of Theorem 1.
Client privacy.In each round the server observes only fresh ciphertexts, which by semantic se-
curity of the GLWE scheme are computationally indistinguishable from encryptions of any other
vector of the same dimension. The two rounds use independent keys, so there is no cross-round
linkage: the server’s view in the two rounds is simulatable from public parameters alone. Because
the client verifies each proof before decrypting, its observable behavior (continue or abort) is a func-
tion of the proof’s validity only, which the server can compute itself. As such, the client’s reactions
leak nothing about the decrypted values, which rules out reaction attacks Chillotti et al. (2016).
Integrity.By soundness of the underlying protocol, any server deviation in roundi, be it substitut-
ing the matrix, corrupting the ciphertext or fabricating the proof, causesπ ito fail verification except
with negligible probability:
Pr
Verify(π i) =Accept∧y i̸=M i·xi
≤negl(λ),
where(M 1,x1) = (A,x)and(M 2,x2) = (D⊤,ei∗). Each round is verified against its own
commitment, so a union bound over the two rounds gives the claim for the composed protocol.
D BATCHEDTOP-kRETRIEVAL
After Round 1, the client can identify the top-kindicesi∗
1, . . . , i∗
kand issuekindependent Round-2
queries. Using a singlek-hot vector instead does not work:D⊤·P
jei∗
jyields thesumof thek
17

Preprint.
documents, from which the individual documents cannot be recovered. For moderatek, thekqueries
batch into a single matrix–matrix productD⊤·E, whereE∈ {0,1}m×khas one-hot columns.
Why latency does not grow withk.The verifiable matrix–vector multiplication of Trem-
blay Thibault et al. (2026) proves the relationc′=M·cwith a sum-check protocol compiled
with a polynomial commitment scheme (PCS). Once per database, the server commits to a polyno-
mial encoding ofM. To answer a query, it runs the sum-check, which reduces the claimc′=M·c
to the evaluation of the committed polynomial at a single random point fixed during the protocol,
and then produces a PCS evaluation proof for that point. The evaluation proof is the dominant cost
factor: it accounts for over95%of the server time.
Batching exploits the fact that the sum-check protocol is linear in the claim it proves. More precisely,
givenkclaimsc′
j=M·c jabout the same committed matrix, the verifier samples a random
challengeγand the prover proves the single claimP
jγjc′
j=M·P
jγjcjinstead. If any of the
kclaims is false, the combined claim is false as well except with probability at mostk/|C|over the
choice ofγ, forCthe challenge space. This additive soundness loss is negligible provided|C|is
large enough. The batched claim is a matrix–vector relation of the same shape as a single query.
As such, it is proved with one run of the sum-check and one PCS evaluation proof, regardless ofk.
The server still computes thekproducts and the sum-check messages of the combined claim, whose
cost grows withk, but the evaluation proof, which dominates, is generated once. Thekanswers are
all returned since the client needs each document, which is why communication grows by a factork
whereas the proof is essentially of the same size as for a single query.
E READERS, PROMPTS ANDDOWNSTREAMCORPORA
Protocol description.We evaluate every endpoint, encoder and conditionc∈
{closed,gold,float,8,6,4,3,2}. The evaluation of §4.4 works as follows. (1) The corpus
and the300queries are embedded once in floating point. For a bit-widthc=b, both are quantized
with the clipped quantizer of Appendix A.6. They are then scored in integers, as in Round 1, and
the top-kpassages are kept. Ties are broken toward the lower index. The float condition uses the
floating-point scores. The gold condition uses the labelled relevant passages, at mostkof them.
The closed condition uses no passage. (2) We build one prompt from the template of the endpoint
(below), with the passages in rank order. A prompt that is identical across conditions is answered
once. (3) The reader answers with greedy decoding. Decoding is constrained to the three labels for
claim verification and capped at24tokens for QA. (4) A QA answer is correct if the normalised text
of any reference answer appears in the normalised response. A claim label is correct if it equals the
human label. SciFact accuracy is computed over the188claims with evidence in the corpus. (5) We
compare per-query correctness under each condition with the float condition on the same queries,
using the tests described below. We record retrieval-side statistics alongside, namely top-koverlap
with float and whether a relevant passage is in the topk. The prompts andkwere fixed before any
bit-width was run. Nothing is tuned on the test queries.
Readers.The body reportsphi-4(14B, MIT licence) (Abdin et al., 2024). Table 6 adds
OLMo-2-7B-Instruct(OLMo Team, 2025), whose training data is fully open, and the Qwen2.5
instruction-tuned models at7B and72B (Qwen Team, 2025). All are run locally with vLLM and
greedy decoding. Under the one-word prompt,OLMo-2mostly abstains on claim verification. It
answers NOINFOon87%of SciFact prompts and never CONTRADICT, so its absolute SciFact accu-
racy is low. It still reproduces the shape of the degradation curve, which is what we compare across
readers.
Prompts.The claim-verification prompt states the claim and lists the retrieved abstracts. It asks
for exactly one word from {SUPPORT, CONTRADICT, NOINFO}. Decoding is constrained to those
three strings. As such, there are no parse failures and no exclusions. The QA prompt states the
question and lists the retrieved passages. It asks for the answer in a few words, or for exactly
UNKNOWNif the passages do not contain it. Responses are capped at24tokens. Prompts are
fixed across bit-widths and encoders. Only the retrieved passages change. One SciFactk=5prompt
exceedsOLMo-2’s4,096-token context and is excluded for that reader.
18

Preprint.
BioASQ corpus.We build the corpus from BioASQ-QA (Krithara et al., 2023) (training11b,
4,719questions). For every question of every type, we group the evidence snippets by source doc-
ument. We concatenate them into one passage per (question, document) pair. This gives35,454
passages with a median length of220characters. Queries are a seeded sample of300factoid ques-
tions with a non-emptyexact_answerand at most three gold passages. Of these,190have one
gold passage,54have two and56have three. The cap keeps gold-in-top-kfrom being trivially high.
The relevant set of a question is its own passages. A response is correct if the normalised text of any
reference answer synonym appears in it.
NQ at scale.The300Natural Questions test queries are the100of the original evaluation plus
200more. These200are a seeded draw among the queries whose question matches an NQ-open
answer list. Passages are retrieved from the full BEIR corpus of2,681,468Wikipedia passages. We
use the same embeddings as the scale run of Appendix F.
Statistics.For every (reader, endpoint, encoder) and bit-width, we report the paired difference in
accuracy from floating point, with a normal-approximation95%interval on the per-query differ-
ences. Table 6 gives these differences atb∈ {4,3,2}. We test for a drop with an exact McNemar
test on the discordant pairs. A non-significant McNemar test is not evidence of equivalence. As
such, we also run two one-sided tests (TOST) with equivalence margins of5and3points. A bit-
width is declared equivalent to floating point when both one-sided tests reject atp <0.05, i.e. when
the90%interval lies inside the margin. Atb=4, all32combinations are equivalent within5points
and23are equivalent within3. Atb=3, these counts are30and17. Atb=2, they are9and2.
Chunked SciFact.Table 7 gives the SciFact endpoint forphi-4when the reader receives the top-
kchunks of §4 instead of whole abstracts. Chunks are cut at sentence boundaries and hold at most
1,536bytes including the title (implem/scale/chunk_corpus.py). The5,183abstracts give
7,510chunks of1,070bytes on average. In floating point, a chunk of a gold abstract is in the top
5for79%of claims with BGE and77%with arctic. For whole abstracts, these figures are80and
79%. NFCorpus, chunked the same way, gives5,761chunks, and44%of its abstracts fit in one. In
floating point, the reader does better on five chunks than on five abstracts, by4.3points with BGE
and2.1with arctic. Atk=1it does about one point worse, since a single chunk may leave out the
evidence sentence. Under quantization, the two encoders separate one bit earlier than on abstracts.
The reason is that chunks of one abstract are near-duplicates, and quantization reorders them. Judged
per document, the ranking survives exactly as on abstracts (Table 8). The reader, judged per chunk,
achieves top-1 agreement atb=3is0.82for BGE and0.89for arctic, against0.87and0.93per
document (Appendix F). With BGE atk=5, the loss reaches4.3points atb=3(p= 0.008). On
abstracts, it was2.7points and not significant. With arctic atk=5, every bit-width down tob=2
stays within5points of floating point, as on abstracts. As such, chunking moves the operating point
of the conventional encoder fromb=3tob=4. The quantization-aware encoder stays atb=2.
F THEENCODERSUITE AND THESCALERUN
In this appendix we give the retrieval-side numbers behind §4.1 and §4.3 for every encoder. We also
compare them with the max-abs quantizer and explain why some encoders lose more quality than
others.
Encoders and prefixes.Each encoder is run with the instruction prefix its model card prescribes.
bge-large-en-v1.5andmxbai-embed-large-v1prepend “Represent this sentence for
searching relevant passages: ” to queries.e5-large-v2prepends “query: ” and “passage: ”.
arctic-embed-l-v2.0prepends “query: ”.gte-largeuses no prefix, and Cohere takes
an input type instead. We runarctic-embed-l-v2.0with a512-token window instead of its
native8,192, to match the other encoders. Abstracts rarely exceed this window. The downstream
evaluation of §4.4 uses the clipped quantizer and these prefixes.
Retained quality under the clipped quantizer.Table 9 gives floating-point nDCG@10 for every
encoder and corpus. It also gives the fraction of it retained atb=4andb=3, and top-1 agreement
with the floating-point ranking at the same bit-widths. Table 10 gives the same quantities under
max-abs scaling next to the clipped values.
19

Preprint.
Table 6: Paired accuracy difference from floating point, in points with95%interval, atb=4,3and2.
All readers, endpoints and encoders use the clipped quantizer (n=300queries,188evidence-bearing
claims for SciFact).∗: significant drop, exact McNemarp <0.05.†: equivalent to floating point
within±5points, TOSTp <0.05. “float” is the floating-point accuracy.
Endpoint Encoder floatb=4b=3b=2
phi-4(14B)
SciFactk=5BGE0.734−1.1 [−3.6,+1.5]†−2.7 [−5.4,+0.1]†−5.9 [−10.1,−1.6]∗
SciFactk=5arctic0.713 +0.5 [−1.3,+2.3]†+0.0 [−2.6,+2.6]†−1.6 [−4.7,+1.5]†
SciFactk=1BGE0.766 +0.5 [−0.5,+1.6]†−0.5 [−2.3,+1.3]†−8.0 [−12.1,−3.8]∗
SciFactk=1arctic0.739 +0.0 [+0.0,+0.0]†+1.6 [−0.2,+3.4]†−5.9 [−10.1,−1.6]∗
BioASQ BGE0.633 +0.3 [−1.4,+2.1]†+1.0 [−0.7,+2.7]†−2.0 [−4.6,+0.6]†
BioASQ arctic0.647−1.0 [−2.5,+0.5]†−0.7 [−2.3,+0.9]†−4.0 [−6.6,−1.4]∗
NQ (2.68M) BGE0.643 +0.7 [−2.1,+3.4]†−2.0 [−5.3,+1.3]†−7.0 [−11.5,−2.5]∗
NQ (2.68M) arctic0.643−1.7 [−3.6,+0.3]†−0.3 [−2.9,+2.2]†−4.3 [−8.4,−0.3]
OLMo-2(7B)
SciFactk=5BGE0.213−0.5 [−3.3,+2.2]†−1.6 [−3.9,+0.7]†−6.4 [−10.7,−2.0]∗
SciFactk=5arctic0.198−1.1 [−3.2,+1.0]†−2.1 [−4.7,+0.4]†−2.1 [−5.4,+1.2]†
SciFactk=1BGE0.505 +1.1 [−0.4,+2.5]†−1.1 [−3.1,+1.0]†−4.8 [−8.5,−1.1]∗
SciFactk=1arctic0.479 +0.0 [−1.5,+1.5]†+0.5 [−1.3,+2.3]†−3.7 [−7.1,−0.3]
BioASQ BGE0.503 +1.0 [−1.5,+3.5]†+0.0 [−3.1,+3.1]†−5.0 [−9.3,−0.7]∗
BioASQ arctic0.520−0.3 [−2.9,+2.2]†−1.3 [−3.9,+1.3]†−2.7 [−6.0,+0.7]
NQ (2.68M) BGE0.497 +1.0 [−2.1,+4.1]†−1.3 [−4.9,+2.2]†−5.0 [−9.5,−0.5]∗
NQ (2.68M) arctic0.563−0.3 [−2.5,+1.8]†−3.3 [−6.2,−0.4]∗−8.3 [−12.2,−4.5]∗
Qwen2.5-7B
SciFactk=5BGE0.681−0.5 [−3.7,+2.6]†−1.6 [−4.7,+1.5]†−5.3 [−9.9,−0.7]∗
SciFactk=5arctic0.654−0.5 [−2.3,+1.3]†+1.1 [−1.5,+3.6]†+0.0 [−3.3,+3.3]†
SciFactk=1BGE0.665 +0.5 [−0.5,+1.6]†−1.1 [−3.1,+1.0]†−9.0 [−13.4,−4.7]∗
SciFactk=1arctic0.644−0.5 [−1.6,+0.5]†+0.0 [−2.1,+2.1]†−5.9 [−10.3,−1.4]∗
BioASQ BGE0.477−1.7 [−4.0,+0.7]†−1.3 [−3.9,+1.3]†−4.3 [−7.8,−0.8]∗
BioASQ arctic0.470−0.3 [−2.3,+1.6]†−1.0 [−3.7,+1.7]†−1.7 [−4.8,+1.5]†
NQ (2.68M) BGE0.353 +1.7 [−1.0,+4.4]†+1.7 [−1.6,+4.9]†−1.3 [−5.6,+2.9]†
NQ (2.68M) arctic0.413−1.7 [−3.8,+0.5]†−2.3 [−5.2,+0.5]†−5.3 [−9.4,−1.2]∗
Qwen2.5-72B
SciFactk=5BGE0.830 +0.5 [−1.3,+2.3]†+0.5 [−1.3,+2.3]†−2.7 [−5.4,+0.1]†
SciFactk=5arctic0.819 +0.0 [+0.0,+0.0]†+0.5 [−1.3,+2.3]†−0.5 [−3.3,+2.2]†
SciFactk=1BGE0.830 +0.5 [−0.5,+1.6]†−1.1 [−3.1,+1.0]†−9.0 [−13.7,−4.4]∗
SciFactk=1arctic0.793 +0.0 [+0.0,+0.0]†+0.0 [−2.6,+2.6]†−7.4 [−12.0,−2.9]∗
BioASQ BGE0.580 +0.0 [−0.9,+0.9]†+0.7 [−0.9,+2.3]†−1.7 [−4.5,+1.2]†
BioASQ arctic0.597 +0.0 [−1.3,+1.3]†−0.3 [−1.5,+0.8]†−3.3 [−6.1,−0.6]∗
NQ (2.68M) BGE0.573−1.7 [−4.0,+0.7]†−2.7 [−5.7,+0.4]−6.7 [−10.5,−2.8]∗
NQ (2.68M) arctic0.567 +0.3 [−1.4,+2.1]†+0.0 [−2.3,+2.3]†−5.0 [−8.7,−1.3]∗
Why some encoders lose more than others.Quantization adds a small rounding error to every
score. A query keeps its top-1 document if the gap between its two best scores is larger than this
error. Rounding to a stepsgives an error that is uniform on[−s/2, s/2], whose variance iss2/12.
The error on a score therefore has standard deviationp
(s2
A+s2x)/12, wheres Ais the corpus step
ands xthe query step. For each query, we divide the gap by this standard deviation. We then take the
median over queries. Table 11 gives this ratio for every encoder atb=4. The larger the ratio, the more
often the top-1 document survives. Over the12encoder-corpus pairs, the ratio and the measured
top-1 agreement have Spearman correlation0.94atb=4and0.99atb=3. The two encoders trained
for low-bit readout have the largest ratios. BGE and mxbai come next, and e5 and gte have the
smallest. The table also explains why clipping helps BGE and mxbai. Their largest coordinate holds
over5%of the squared norm, against at most1.9%for the other encoders. Under max-abs scaling,
this one coordinate sets the step. For BGE on SciFact, the step would be0.305/7≈0.044instead
of the clipped0.011, i.e. about4×coarser.
20

Preprint.
Table 7: SciFact with chunks against whole abstracts, readerphi-4. We give floating-point ac-
curacy on the188evidence-bearing claims. We also give the paired difference from floating point
atb=4,3and2, in points with95%interval.∗: significant drop, exact McNemarp <0.05.†:
equivalent within±5points, TOSTp <0.05.
Endpoint Encoder Unit floatb=4b=3b=2
SciFactk=5BGE abstracts0.734−1.1 [−3.6,+1.5]†−2.7 [−5.4,+0.1]†−5.9 [−10.1,−1.6]∗
chunks0.777−2.7 [−5.0,−0.4]†−4.3 [−7.1,−1.4]∗−6.9 [−11.4,−2.5]∗
SciFactk=5arctic abstracts0.713 +0.5 [−1.3,+2.3]†+0.0 [−2.6,+2.6]†−1.6 [−4.7,+1.5]†
chunks0.734 +1.6 [−0.7,+3.9]†+1.6 [−1.9,+5.1]†−1.1 [−5.2,+3.1]†
SciFactk=1BGE abstracts0.766 +0.5 [−0.5,+1.6]†−0.5 [−2.3,+1.3]†−8.0 [−12.1,−3.8]∗
chunks0.755−0.5 [−2.3,+1.3]†−2.1 [−4.7,+0.4]†−12.2 [−17.6,−6.9]∗
SciFactk=1arctic abstracts0.739 +0.0 [+0.0,+0.0]†+1.6 [−0.2,+3.4]†−5.9 [−10.1,−1.6]∗
chunks0.729 +0.5 [−1.8,+2.9]†+0.0 [−3.0,+3.0]†−5.9 [−10.6,−1.1]∗
Table 8: Retrieval quality retained of our two-round protocol under clipped quantization, as the
percentage of floating-point nDCG@10 capped at100, forb∈ {2,3,4,8}on SciFact and NFCorpus,
ranked per chunk and scored per document.
SciFact NFCorpus
low-bit conventional low-bit conventional
barctic Cohere BGE mxbai e5 gte arctic Cohere BGE mxbai e5 gte
2 92.2 89.7 86.4 88.5 82.2 86.6 88.2 88.6 85.9 86.5 78.8 83.6
3 98.7 99.1 97.4 98.7 97.3 97.5 98.2 99.3 99.0 97.4 94.1 97.6
499.9 99.4 99.7 99.4 100.0 99.8 99.6 99.8 99.5 98.9 96.5 100.0
8 99.9 99.5 99.9 100.0 100.0 99.8 99.7 100.0 99.7 100.0 98.7 100.0
Chunk-level retrieval.We first judge the chunked SciFact corpus per document, as in Table 8.
Top-1 agreement atb=4andb=3is then0.94and0.87for BGE and0.96and0.92for arctic. These
are the same values as on whole abstracts (Table 12). We then judge it per chunk, counting every
chunk of a relevant abstract as relevant. Top-1 agreement is then0.92and0.82for BGE and0.93and
0.89for arctic. The difference comes from chunks of one abstract trading places under rounding.
The reader of §4.4 sees this reordering, but a document ranking does not.
Quality at scale.Tables 14 and 13 give the measurements behind the quality-at-scale paragraph
of §4.1. They cover the five encoders of the suite that run locally, namely arctic, BGE, mxbai, e5
and gte. Cohere is hosted and was not run at this scale. The corpus is the BEIR Natural Questions
passage collection (2,681,468passages) with its3,452test queries. We draw subsets of104,105
and106passages at random with a fixed seed. Each subset contains every passage relevant to any
query. As such, nDCG is defined at every size. Embeddings are computed once per encoder on
one H100 and stored in half precision. The clipped percentile is estimated from a random sample
of4,096rows of each subset. Quantized scores are computed in single precision. This is exact for
b≤8, since every partial sum is an integer below224. We also check the scores against 64-bit
integer arithmetic. Ties are broken toward the lower index.
Recall at the reader’sk.The sweep above scores atk=10. The readers of §4.4 consume five
passages. As such, we repeated the sweep atk=5and addedb=2. Table 13 gives Recall@5 retained
at every size. Atb=4, four of the five encoders retain at least98.8%of floating-point Recall@5 at
every size. The fifth, e5, retains97.3%at2.68M. Atb=3, arctic, BGE and mxbai retain at least
98.4%. At2.68M, gte and e5 fall to94.5and92.0%. Atb=2, the loss grows with the corpus. It
goes from1to4%at104to10to23%at2.68M. The lead of arctic over each conventional encoder
also grows, from0.8to2.7points at104to2.1to12.8points at2.68M. This is the retrieval-side
counterpart of the downstream drop of §4.4. On the300NQ questions of that section, the top-5set
of BGE contained a relevant passage for70%of questions in floating point and60%atb=2. This is
86%retained, against86.1%measured here on all3,452queries.
21

Preprint.
Table 9: The encoder suite under the clipped quantizer. Floating-point nDCG@10, the percentage
of it retained atb=4andb=3, and top-1 agreement with the floating-point ranking at the same bit-
widths.✓: trained for low-bit readout.
SciFact NFCorpus
Encoder low-bit float ret.4ret.3top-14top-13float ret.4ret.3top-14top-13
arctic-embed-l-v2.0✓0.706 100.4 99.4 0.96 0.93 0.354 99.6 98.0 0.91 0.82
Cohere embed-v3✓0.718 99.6 98.7 0.96 0.91 0.387 99.5 98.9 0.86 0.78
bge-large-en-v1.5×0.746 99.4 97.1 0.94 0.87 0.381 99.5 98.6 0.83 0.71
mxbai-embed-large-v1×0.739 100.3 100.1 0.93 0.89 0.387 100.1 98.9 0.84 0.73
e5-large-v2×0.722 99.4 96.8 0.88 0.77 0.372 96.3 90.8 0.79 0.64
gte-large×0.743 99.7 97.8 0.94 0.81 0.383 100.0 95.2 0.81 0.62
Table 10: The quantizer lever. Each cell reads max-abs scaling→clipping at the99th percentile.
We give the percentage of floating-point nDCG@10 retained and top-1 agreement atb=4andb=3.
Nothing on the cryptographic side differs between the two columns of each pair.
b=4b=3
Encoder ret. (%) top-1 ret. (%) top-1
SciFact
arctic-embed-l-v2.0100.0→100.4 0.93→0.96 97.6→99.4 0.83→0.93
Cohere embed-v3100.0→99.6 0.93→0.96 95.4→98.7 0.74→0.91
bge-large-en-v1.597.1→99.4 0.79→0.94 77.3→97.1 0.54→0.87
mxbai-embed-large-v199.5→100.3 0.81→0.93 82.3→100.1 0.57→0.89
e5-large-v296.3→99.4 0.79→0.88 82.4→96.8 0.56→0.77
gte-large97.3→99.7 0.81→0.94 88.1→97.8 0.68→0.81
NFCorpus
arctic-embed-l-v2.098.5→99.6 0.83→0.91 96.2→98.0 0.69→0.82
Cohere embed-v398.5→99.5 0.83→0.86 96.4→98.9 0.61→0.78
bge-large-en-v1.596.0→99.5 0.67→0.83 73.4→98.6 0.35→0.71
mxbai-embed-large-v196.2→100.1 0.63→0.84 77.4→98.9 0.37→0.73
e5-large-v293.0→96.3 0.65→0.79 78.4→90.8 0.37→0.64
gte-large96.3→100.0 0.62→0.81 85.0→95.2 0.42→0.62
Table 11: Score gap against rounding error atb=4, clipped quantizer.max|A|: largest corpus
coordinate.s A: clipped step. top dim.: share of the squared norm in the largest coordinate. gap:
median gap between the two best floating-point scores. gap/noise: median ratio of that gap to the
rounding error. top-1: top-1 agreement with floating point.
Encoder Corpusmax|A|s A top dim. (%) gap gap/noise top-1
arctic-embed-l-v2.0SciFact0.200 0.0122 0.5 0.0582 11.72 0.96
Cohere embed-v3SciFact0.221 0.0125 0.8 0.0445 8.83 0.96
bge-large-en-v1.5SciFact0.305 0.0110 5.5 0.0319 7.00 0.94
mxbai-embed-large-v1SciFact0.311 0.0111 5.4 0.0345 7.73 0.93
e5-large-v2SciFact0.170 0.0105 1.9 0.0149 3.48 0.88
gte-largeSciFact0.172 0.0107 1.7 0.0189 4.38 0.94
arctic-embed-l-v2.0NFCorpus0.194 0.0123 0.5 0.0268 5.22 0.91
Cohere embed-v3NFCorpus0.216 0.0126 1.0 0.0211 4.19 0.86
bge-large-en-v1.5NFCorpus0.287 0.0109 5.7 0.0117 2.64 0.83
mxbai-embed-large-v1NFCorpus0.292 0.0110 5.8 0.0135 2.97 0.84
e5-large-v2NFCorpus0.165 0.0105 1.9 0.0063 1.50 0.79
gte-largeNFCorpus0.175 0.0107 1.7 0.0073 1.67 0.81
22

Preprint.
Table 12: Table 8 on whole abstracts instead of chunks. This checks comparability against pub-
lished document-level baselines. We give the percentage of floating-point nDCG@10 retained under
clipped quantization for each bit-widthb.
Trained for low-bit readout Conventional
barctic Cohere BGE mxbai e5 gte
SciFact
2 91.8 92.1 87.0 90.4 84.0 87.6
3 99.4 98.7 97.1 100.1 96.8 97.8
4100.4 99.6 99.4 100.3 99.4 99.7
5 100.7 99.8 99.0 99.9 101.3 99.2
6 100.1 99.7 99.4 100.0 101.2 99.7
7 100.2 99.8 99.2 100.0 100.9 99.7
8 100.0 99.7 99.1 100.2 100.9 99.8
NFCorpus
2 87.2 88.7 83.2 87.4 75.0 82.7
3 98.0 98.9 98.6 98.9 90.8 95.2
499.6 99.5 99.5 100.1 96.3 100.0
5 99.2 99.8 100.6 100.7 97.8 98.7
6 99.9 99.7 100.4 101.0 98.5 99.8
7 99.9 99.8 100.5 100.7 98.6 99.7
8 99.6 99.7 100.4 100.8 98.5 99.7
Table 13: Recall@5 retained under the clipped quantizer versus corpus size on BEIR Natural Ques-
tions (3,452queries), atb∈ {2,3,4}. Floating-point Recall@5 is given for reference. Generated by
implem/analysis/scale_recall5_table.py.
Recall@5 retained (%)
EncodermRecall@5 (float)b=2b=3b=4
arctic-embed-l-v2.01040.988 99.2 99.9 100.0
1050.967 97.0 99.9 99.9
1060.848 93.1 99.5 99.7
2.68·1060.737 89.9 99.0 99.8
bge-large-en-v1.51040.982 97.7 99.9 100.1
1050.944 94.7 99.2 99.5
1060.794 88.3 98.7 98.8
2.68·1060.655 86.1 98.5 99.7
mxbai-embed-large-v11040.983 98.4 99.9 100.1
1050.944 96.3 99.6 99.9
1060.800 90.4 98.8 99.4
2.68·1060.661 87.7 98.4 100.6
e5-large-v21040.988 96.5 99.5 99.8
1050.961 92.1 98.5 99.5
1060.847 82.7 95.8 98.3
2.68·1060.745 77.1 92.0 97.3
gte-large1040.986 97.2 99.7 99.8
1050.945 94.4 99.4 100.0
1060.795 85.9 96.3 99.9
2.68·1060.658 82.6 94.5 99.6
23

Preprint.
Table 14: Quality versus corpus size on BEIR Natural Questions with the clipped quan-
tizer. We give nDCG@10 retained atb=3andb=4. We also give the smallestbthat re-
tains99%of floating-point nDCG@10, or>8when nobup to8does. Generated by
implem/analysis/scale_analysis.py.
nDCG@10 ret. (%) requiredb
Encoderm b=3b=4(top-k)
arctic-embed-l-v2.0104100.0 99.9 3
10599.9 100.1 3
10699.1 99.9 3
2.68·10698.5 99.7 4
bge-large-en-v1.510499.5 99.9 3
10599.0 99.6 3
10698.0 99.1 4
2.68·10698.1 99.3 4
mxbai-embed-large-v110499.8 100.0 3
10599.5 100.0 3
10698.9 100.1 4
2.68·10698.3 99.9 4
e5-large-v210499.0 99.7 3
10597.7 99.3 4
10693.9 98.1 8
2.68·10691.9 97.4>8
gte-large10499.4 99.9 3
10598.4 99.9 4
10695.7 99.9 4
2.68·10694.0 100.6 4
24