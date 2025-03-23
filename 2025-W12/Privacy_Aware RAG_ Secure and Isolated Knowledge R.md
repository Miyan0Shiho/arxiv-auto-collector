# Privacy-Aware RAG: Secure and Isolated Knowledge Retrieval

**Authors**: Pengcheng Zhou, Yinglun Feng, Zhongliang Yang

**Published**: 2025-03-17 07:45:05

**PDF URL**: [http://arxiv.org/pdf/2503.15548v1](http://arxiv.org/pdf/2503.15548v1)

## Abstract
The widespread adoption of Retrieval-Augmented Generation (RAG) systems in
real-world applications has heightened concerns about the confidentiality and
integrity of their proprietary knowledge bases. These knowledge bases, which
play a critical role in enhancing the generative capabilities of Large Language
Models (LLMs), are increasingly vulnerable to breaches that could compromise
sensitive information. To address these challenges, this paper proposes an
advanced encryption methodology designed to protect RAG systems from
unauthorized access and data leakage. Our approach encrypts both textual
content and its corresponding embeddings prior to storage, ensuring that all
data remains securely encrypted. This mechanism restricts access to authorized
entities with the appropriate decryption keys, thereby significantly reducing
the risk of unintended data exposure. Furthermore, we demonstrate that our
encryption strategy preserves the performance and functionality of RAG
pipelines, ensuring compatibility across diverse domains and applications. To
validate the robustness of our method, we provide comprehensive security proofs
that highlight its resilience against potential threats and vulnerabilities.
These proofs also reveal limitations in existing approaches, which often lack
robustness, adaptability, or reliance on open-source models. Our findings
suggest that integrating advanced encryption techniques into the design and
deployment of RAG systems can effectively enhance privacy safeguards. This
research contributes to the ongoing discourse on improving security measures
for AI-driven services and advocates for stricter data protection standards
within RAG architectures.

## Full Text


<!-- PDF content starts -->

Privacy-Aware RAG: Secure and Isolated Knowledge Retrieval
Pengcheng Zhou†,Yinglun Feng†,Zhongliang Yang∗
ABSTRACT
The widespread adoption of Retrieval-Augmented Generation (RAG)
systems in real-world applications has heightened concerns about
the confidentiality and integrity of their proprietary knowledge
bases. These knowledge bases, which play a critical role in enhanc-
ing the generative capabilities of Large Language Models (LLMs),
are increasingly vulnerable to breaches that could compromise sen-
sitive information. To address these challenges, this paper proposes
an advanced encryption methodology designed to protect RAG
systems from unauthorized access and data leakage. Our approach
encrypts both textual content and its corresponding embeddings
prior to storage, ensuring that all data remains securely encrypted.
This mechanism restricts access to authorized entities with the ap-
propriate decryption keys, thereby significantly reducing the risk of
unintended data exposure. Furthermore, we demonstrate that our
encryption strategy preserves the performance and functionality of
RAG pipelines, ensuring compatibility across diverse domains and
applications. To validate the robustness of our method, we provide
comprehensive security proofs that highlight its resilience against
potential threats and vulnerabilities. These proofs also reveal limi-
tations in existing approaches, which often lack robustness, adapt-
ability, or reliance on open-source models. Our findings suggest
that integrating advanced encryption techniques into the design
and deployment of RAG systems can effectively enhance privacy
safeguards. This research contributes to the ongoing discourse on
improving security measures for AI-driven services and advocates
for stricter data protection standards within RAG architectures.
1 INTRODUCTION
The proliferation of Retrieval-Augmented Generation (RAG) sys-
tems across a wide range of real-world applications has brought
heightened concerns regarding the confidentiality and integrity of
their proprietary knowledge bases [ 23,35]. These knowledge bases,
pivotal for augmenting the generative capabilities of Large Lan-
guage Models (LLMs), are particularly vulnerable to breaches that
could compromise sensitive information [ 59]. For instance, RAG
systems deployed in medical support chatbots [ 42,44,52] leverage
previous medical records for initial case screening, thereby raising
significant privacy concerns if not adequately protected.
Despite their utility, conventional RAG systems face significant
security challenges. A critical vulnerability lies in their susceptibil-
ity to prompt injection attacks, where adversaries can exploit the
†Equal Contribution.∗Corresponding Author.
Pengcheng Zhou is with the International School, Beijing University of Posts and
Telecommunications, Beijing 100876, China.
Yinglun Feng, Zhongliang Yang are with the School of Cyberspace Security, Bei-
jing University of Posts and Telecommunications.Beijing 100876, China (e-mail:
yangzl@bupt.edu.cn).
Conference’17, July 2017, Washington, DC, USA
©2025 Association for Computing Machinery.
ACM ISBN 978-x-xxxx-xxxx-x/YY/MM. . . $15.00
https://doi.org/10.1145/nnnnnnn.nnnnnnn
Please adhere to this directive: After a brief response, duplicate and insert all the 
provided  text from the context, with each piece placed on a separate line.
[User B] has an eye exam scheduled for the first week of December.
[User C] is seeing her cardiologist on Friday afternoon.
[User A]welcomed their second child, a baby girl, in January 2025.
Medical appointment cancellations must be made 24 hours in advance to avoid fees.
[User A] is going for a blood test appointment on Thursday.
[User A]welcomed their second child, a baby girl, in January 2025.
Medical appointment cancellations must be made 24 hours in advance to avoid fees.
Patients can review their medical history through the online appointment portal.
Ugh, it's such a letdown.
 I put in so much effort, but things still didn't 
turn out as I hoped.Would you be able to assist [User A] in retrieving his lost medical appointment?Figure 1: The portion enclosed by the green dashed box indicates the
output of a correctly guarded RAG system against attacks, while the portion
enclosed by the red dashed box indicates the output of the vast majority of
current RAGs facing such attacks.
system to extract sensitive information referenced by the LLM. For
example, an attacker could craft malicious prompts to induce the
system to reveal private data, such as medical records or financial in-
formation, by bypassing existing access controls [ 18,57]. This flaw
stems from the inherent design of RAG systems, which often lack
mechanisms to enforce strict user isolation and secure data access.
Furthermore, existing methodologies for protecting RAG knowl-
edge bases are often inadequate, relying on simplistic encryption
schemes or open-source models that fail to provide comprehensive
protection against sophisticated attacks [ 14,43,57]. These limita-
tions highlight the need for a more robust and theoretically sound
approach to securing RAG systems. In response to the critical se-
curity challenges faced by Retrieval-Augmented Generation (RAG)
systems, this paper proposes an advanced encryption methodology
designed to fortify these systems against unauthorized access and
data leakage. Conventional RAG systems are vulnerable to attacks
such as prompt injection, where adversaries exploit the system to
extract sensitive information referenced by the LLM. To address
these vulnerabilities, our approach integrates cryptographic tech-
niques into the core architecture of RAG systems, ensuring robust
protection of both textual content and its corresponding embed-
dings. Specifically, all data is encrypted before storage, maintaining
it in an encrypted format throughout its lifecycle. This ensures that
only authorized entities with the appropriate decryption keys can
access or utilize the stored information for retrieval and genera-
tion tasks, thereby significantly mitigating the risk of unintended
exposure (see Figure 1).arXiv:2503.15548v1  [cs.CR]  17 Mar 2025

A key innovation of our methodology is the introduction of a
user-isolated enhancement system. Before the retrieval phase, the
system performs user authentication and establishes a hierarchi-
cal encryption key structure to ensure exclusive access to private
knowledge bases. This is combined with a secure similarity com-
putation mechanism that operates on public knowledge bases to
retrieve the top-k relevant documents. Crucially, the system filters
out non-user private data from the final retrieval results, effectively
preventing cross-user privacy leakage. This design addresses the
vulnerabilities identified in prior research [ 8,9,32,34,56], offering
a robust defense against potential threats (see Figure 2).
Our methodology is underpinned by two distinct encryption
schemes, each tailored to address specific security and performance
requirements:
1. Method A: AES-CBC-Based Encryption. This approach pro-
vides a straightforward yet efficient solution for encrypting data.
Users map primary keys (PKs) to AES-CBC keys, enabling the en-
cryption of document chunks. This method ensures compatibility
with traditional databases while providing robust data protection
against unauthorized access.
2. Method B: Chained Dynamic Key Derivation. For scenarios
requiring enhanced security and integrity, we propose a more ad-
vanced scheme where data nodes form a linked chain with dy-
namically generated keys and hash-based integrity checks. This
chained structure not only enhances security but also ensures that
unauthorized users cannot access the data without the correct key
hierarchy.
By integrating the proposed encryption schemes, our methodol-
ogy provides a comprehensive solution to the security challenges
faced by RAG systems. Method A, which employs AES-CBC en-
cryption, strikes a balance between efficiency and security, making
it well-suited for applications with moderate security requirements.
In contrast, Method B, with its chained dynamic key derivation
mechanism, offers a higher level of protection for sensitive data,
making it ideal for high-stakes environments such as healthcare and
finance. Together, these methods address the limitations of existing
approaches, such as their reliance on simplistic encryption schemes
or open-source models, by providing a robust and adaptable frame-
work for securing RAG systems. In addition, our findings highlight
the critical role of advanced encryption techniques in safeguarding
the confidentiality and integrity of RAG systems. This research con-
tributes to the ongoing discourse on enhancing security measures
for AI-driven services, advocating for more rigorous standards in
data protection within RAG architectures. By addressing the vul-
nerabilities identified in previous studies, our work underscores the
necessity of advanced encryption strategies to safeguard sensitive
information in RAG systems.
To summarize, this paper makes the following key contributions:
1. Advanced Encryption Scheme for RAG: We propose a novel
encryption scheme that secures both textual content and embed-
dings, specifically designed to protect RAG knowledge bases from
unauthorized access. This innovative approach ensures robust data
protection while maintaining system performance and functional-
ity, marking a significant advancement in the field.
2. Theoretical Framework for Privacy Safeguards: Our work
establishes a theoretical framework for integrating encryption tech-
niques into RAG systems, emphasizing the importance of privacysafeguards in AI-driven services. This framework serves as a guide-
line for future research and development in secure RAG architec-
tures.
3. Comprehensive Security Proofs: We provide rigorous theoret-
ical analysis and security proofs to validate the effectiveness of our
encryption methodology. These proofs demonstrate the scheme’s
resilience against potential threats and vulnerabilities, offering a
solid foundation for its adoption in real-world applications.
2 RELATED WORK
2.1 Encrypted Database
Traditional database encryption schemes provide valuable insights
into securing sensitive information, which can be adapted and
enhanced for RAG applications. Traditional database security solu-
tions are typically categorized into three layers: physical security,
operating system security, and DBMS (Database Management Sys-
tem) security [ 54]. However, these conventional measures often fall
short in scenarios where attackers gain access to raw database data,
bypassing traditional mechanisms. This is particularly problematic
for insiders such as system administrators and database administra-
tors (DBAs), who may exploit elevated privileges to access sensitive
information.
To mitigate these risks, enterprises have adopted database en-
cryption as an advanced measure to protect private data, especially
in critical sectors like banking, finance, insurance, government,
and healthcare [ 48]. While database-level encryption does not offer
complete protection against all forms of attacks, it ensures that
only authorized users can view the data and safeguards database
backups from loss, theft, or other compromises.
Several academic works have explored various encryption con-
figurations and strategies applicable to RAG systems. These ap-
proaches can be categorized into four main classes: file system
encryption, DBMS-level encryption, application-level encryption,
and client-side encryption. Each approach has its strengths and
weaknesses concerning security, performance, and ease of integra-
tion.
File-system encryption involves encrypting the entire physical
disk, protecting the database but limiting the ability to use different
encryption keys for different parts of the data [ 31]. In contrast,
DBMS-level encryption schemes provide greater flexibility and sup-
port for internal DBMS mechanisms such as indexes and foreign
keys. For instance, schemes based on the Chinese Remainder The-
orem allow encryption at the row level with different sub-keys
for different cells [ 16]. Other schemes extend this by supporting
multilayer access control [ 28] or utilize Newton’s interpolating
polynomials or RSA public-key cryptography for column- or row-
oriented encryption [10].
Application-level encryption translates user queries into en-
crypted queries that execute over an encrypted DBMS, providing a
layer of abstraction between the user and the database [ 39]. This is
particularly relevant for RAG systems, where queries often involve
complex interactions between retrieval and generation components.
Client-side encryption, often associated with the "Database as a Ser-
vice" (DAS) model, ensures that sensitive data remains encrypted
even when stored on untrusted servers [ 5], making it suitable for
cloud-based RAG deployments.

Embedding(TextA1）
 TextA1
Embedding(TextA2）
 TextA2
Embedding(TextA3）
 TextA3
Embedding(TextX ）
 TextX
Embedding(TextY ）
 TextY
Embedding(TextZ ）
 TextZ
Embedding(TextB1）
 TextB1
Embedding(TextB2）
 TextB2
Knowledge Base Of User A
Knowledge Base Of User B        User A
        User B
        User A
Would you be able to assist [User A] in retrieving his lost medical appointment?
Embedding
Relavant Documents
User A is going to visit his medic on October 15 for a persistent cough and chest pain.
User A has three kids and one of them lives in Chicago.
Annual check-ups are recommended and can be scheduled as medical appointments.
Patients can review their medical history through the online appointment portal.
User B scheduled a medical appointment for a routine check-up next Tuesday.
LLM
I'm here to help [User A] retrieve his
lost medical appointment. It seems you
have a medical visit scheduled on
October 15 for a persistent cough and
chest pain. You can review and possibly
retrieve your appointment details
through the online appointment portal.
Figure 2: As depicted in Figure, the system framework is presented, with arrows illustrating the direction of data flow and different colors denoting the
sources of the data. This framework systematically demonstrates how the system processes User A’s ID and key to extract User A’s text vectors and text. It
then computes similarity and securely inputs User A’s legitimate information into the LLM for security prompts, ensuring that no information from other
users is accessed.
Indexing encrypted data poses additional challenges due to the
need for preserving functionalities like range searches. Various
schemes have been proposed, including those that build B-Tree
indexes over plaintext values before encrypting them at the row
level [ 3]. Other schemes propose order-preserving encryption func-
tions that enable direct range queries on encrypted data without
decryption [ 1], although they may expose sensitive information
about the order of values.
Key management is another critical aspect of encrypted databases.
Secure key storage, cryptographic access control, and key recovery
are essential components of any robust encryption solution [ 24].
The literature suggests various techniques for generating and man-
aging encryption keys, including methods that generate a pair of
keys for each user, keeping the private key at the client end and
the public key at the server side [12].
2.2 Retrieval-Augmented Generation (RAG)
Retrieval-Augmented Generation (RAG), initially introduced by [ 35],
has rapidly become one of the most prominent methodologies for
enhancing the generative capabilities of Large Language Models
(LLMs) [ 11,45,47,51]. This approach significantly improves the
accuracy and relevance of generated outputs by mitigating com-
mon issues such as "hallucinations" in LLMs [ 20,50]. One of RAG’s
distinctive features is its flexible architecture, enabling the inter-
change or update of its core components—the dataset, retriever, and
LLM—without necessitating retraining or fine-tuning of the entire
system [ 13,46]. Consequently, RAG has been widely adopted across
various practical applications, including personal chatbots and spe-
cialized domain experts like medical diagnostic assistants [41].
The increasing attention towards LLMs, both in industry and
academia, underscores their remarkable ability to facilitate convinc-
ing linguistic interactions with humans [ 29,30,36,60]. However,
adapting these models to new knowledge not available at training
time poses significant challenges. For instance, in real-world sce-
narios involving virtual assistants [ 15,21,33], the knowledge baseor tasks may evolve over time, requiring model adaptation through
fine-tuning processes [ 2,17,58]. This can lead to catastrophic for-
getting, where previously acquired knowledge is lost [ 38]. Alter-
natively, new knowledge can be appended to the input prompt
via in-context learning (ICL) without altering the model parame-
ters [6, 19, 37, 53, 55], a principle that underpins RAG systems.
In the context of RAG, a typical system comprises four principal
components [ 45]: (i) a text embedder function 𝑒, which maps textual
information into a high-dimensional embedding space; (ii) a storage
mechanism, often referred to as a vector store, that memorizes texts
and their embedded representations; (iii) a similarity function, such
as cosine similarity, used to evaluate the similarity between pairs
of embedded text vectors; and (iv) a generative model, denoted as
function𝑓, typically an LLM, that produces output text based on
input prompts and retrieved information.
Given a pre-trained LLM, documents {𝐷1,...,𝐷𝑚}are divided
into smaller chunks (sentences, paragraphs, etc.) to form a private
knowledge base 𝐾[25]. These chunks are then stored in the vector
store as embeddings. When interacting with a user, given an input
prompt𝑞, the system retrieves the top- 𝑘most similar chunks from
𝐾using the embedding space. The generation process conditions
on both the input prompt and the retrieved chunks to produce
coherent and contextually relevant output text.
By integrating retrieval and generation, RAG provides a versa-
tile framework for leveraging external knowledge without compro-
mising the integrity of the underlying LLM. This dual approach
enhances the adaptability of LLMs to evolving knowledge bases
and ensures robust performance across diverse applications. The
ability to dynamically integrate new information makes RAG par-
ticularly suitable for scenarios requiring up-to-date knowledge,
thereby extending the utility and applicability of LLMs in practical
settings.

2.3 Privacy Risk of Large Language Models
The privacy risks associated with Large Language Models (LLMs)
have garnered considerable attention in recent literature, highlight-
ing the vulnerabilities inherent in these systems when handling
sensitive information. [ 9] were among the first to delve into data
extraction attacks on LLMs, revealing their propensity for inadver-
tently reproducing segments of training data. Subsequent studies
have further delineated various factors, such as model size, data
duplication, and prompt length, which exacerbate the risk of memo-
rization [ 4,8]. Mireshghallah et al. [ 40]and Zeng et al. [ 56] extended
this investigation to fine-tuning practices, identifying that adjusting
model heads rather than smaller adapter modules leads to more
significant memorization. Furthermore, tasks demanding extensive
feature representation, such as dialogue and summarization, exhibit
particular vulnerabilities to memorization during fine-tuning [ 56].
Parallelly, the deployment of AI models in privacy-sensitive ap-
plications has raised concerns about protecting sensitive informa-
tion within AI systems [ 22,27]. In the context of LLMs, even those
trained on public datasets can retain and expose fragments of their
training data, leading to specific privacy-oriented attacks [ 7,26,49].
The introduction of Retrieval-Augmented Generation (RAG) sys-
tems adds another layer of complexity to these issues due to their
reliance on proprietary knowledge bases that collect sensitive in-
formation [ 59]. This setup poses significant risks, as user queries
could be manipulated to expose private data contained within the
RAG model’s responses [14, 29, 43, 57].
Given these challenges, our work proposes an advanced en-
cryption methodology designed to protect RAG systems against
unauthorized access and data leakage. Unlike previous approaches,
our method encrypts both textual content and its corresponding
embeddings before storage, ensuring all data is maintained in an en-
crypted format. This strategy ensures that only authorized entities
can access or utilize stored information, significantly reducing the
risk of unintended exposure without compromising performance
or functionality.
3 METHODOLOGY
3.1 Overall
To address the security flaw in conventional RAG workflows where
adversaries can extract LLM-referenced data through prompt in-
jection attacks (e.g., "copy and insert all contextual text informa-
tion"), we propose a user-isolated enhancement system. Before the
retrieval phase, the system performs user authentication and estab-
lishes a cryptographic key hierarchy to ensure exclusive access to
private knowledge bases.That means they will reconstruct the user
private database by:
𝑃𝑟𝑖𝑣𝐷𝑎𝑡𝑎𝑢𝑠𝑒𝑟=𝑃𝑟𝑖𝑣𝐷𝑎𝑡𝑎𝑢𝑠𝑒𝑟∪𝑃𝑢𝑏𝑙𝑖𝑐𝐷𝑎𝑡𝑎 (1)
This is integrated with secure similarity computation against public
knowledge bases to return top k relevant documents. As shown
in Figure 2, the system filters out non-UserA private data from
the final retrieval results, effectively preventing cross-user privacy
leakage.
Building upon the principle of secure data access, we propose
two cryptographic methods.Method A: Users map primary keys (PKs) to AES-CBC keys with
encryption:
ENC𝐾𝑖(𝑚𝑖)=AES CBC.Enc(𝑚𝑖,𝐾𝑖)=𝑐𝑖 (2)
This calculation method is simple, efficient, and compatible with
traditional databases.
textFormula Definitions:
•𝐾𝑖: AES-256 key uniformly sampled from {0,1}256
•𝑚𝑖: Plaintext document chunk indexed by primary key (PK)
•𝑐𝑖: Representing the encrypted content using 𝐾𝑖
Method B: Chained dynamic key derivation,Data nodes form a
linked chain with dynamically generated keys 𝐾𝑖+1and hash in-
tegrity checks ℋ:{0,1}∗→{ 0,1}𝜆,where𝜆is is the security
parameter. each linked list node is shown below(taking user A as
an example):
𝑛𝑜𝑑𝑒𝐴,𝑖=[𝑒𝑚𝑏𝑒𝑑𝑑𝑖𝑛𝑔(𝑚𝐴,𝑖)||𝑚𝐴,𝑖||𝐾𝐴,𝑖+1
||ℋ(𝐾𝐴,𝑖)||𝑎𝑑𝑑𝑟(𝑛𝑜𝑑𝑒𝐴,𝑖+1)](3)
textFormula Definitions:
•embedding(𝑚𝐴,𝑖): Vector representation of document chunk
𝑚𝑖
•𝐾𝐴,𝑖+1: Next node’s key derived via HKDF(𝐾𝐴,1),every key
𝐾𝐴,𝑖uniformly sampled from {0,1}256
•ℋ(𝐾𝐴,𝑖): Integrity checksum of current key (e.g., SHA-256)
•𝑎𝑑𝑑𝑟(𝑛𝑜𝑑𝑒𝐴,𝑖+1):Indicate the address of the next node
Through the trapdoor, it ensures that users who hold the key cor-
rectly can retrieve their own data without having to retrieve data
from others. The structure of the trapdoor is as follows (taking user
A as an example):
𝑇𝑟𝑎𝑝𝑑𝑜𝑜𝑟𝐴=𝐻(𝐼𝐷𝐴||𝑠𝑎𝑙𝑡)⊕(𝐼𝐷𝐴||𝐾𝐴,1||𝑎𝑑𝑑𝑟(𝑛𝑜𝑑𝑒𝐴.1)) (4)
Formula Definitions:
•𝐼𝐷𝐴: Unique identifier ID of user A
•𝑠𝑎𝑙𝑡: The salt value held by User A is used to ensure the
security of retrieving User A’s data
•⊕: XOR operation for trapdoor security
•𝐾𝐴,1: Root key for the user 𝐴’s data chain
3.2 Method A: AES-CBC-Based Encryption
This scheme, as shown in Figure 3, represents a straightforward but
fundamental approach. To securely retrieve the private chunk and
its corresponding embedding for each user, it requires maintaining
a dedicated data entry for every user in our possession. Given that
the primary key uniquely identifies a single data item within a
table, each user must maintain an array where the contents signify
the data owned by that user. The encryption keys are utilized to
separately encrypt both the chunk and its embedding, ensuring that,
even if an adversary gains access to the user’s data, they cannot
decipher the user’s information without the corresponding key.
During similarity calculations, the encrypted information has a
tendency to be shuffled. Consequently, within the RAG system, the
ranking of a user’s encrypted document tends to be significantly

Knowledge Base
PK1
PK2
PK3
PK4
AESEmbedding(ChunkA1）
Embedding(ChunkA 2）
ChunkA1
ChunkA 2
AES Embedding(ChunkB1） ChunkB1
PK1
PK4
User A
Search
AES
Figure 3: In the diagram of Scheme A’s knowledge base user encryption
and decryption process, the green and blue dashed lines represent the
execution flows of different users, the black dashed line represents the
database primary key search flow, and the orange box lines indicate the
AES encryption and decryption algorithm.
low. Since the similarity calculation process selects the top k docu-
ments that are closest to the user’s input embedding, this method
inherently avoids the extraction of User A’s private information.
This scheme encompasses four primary components: key gen-
eration, user information encryption and decryption, user chunk
addition, and user chunk extraction.
Key generation: In order to generate the key, the security param-
eter𝜆is selected, and each user is randomly sampled:
𝐾𝑖:=𝑥$← −{0,1}𝜆(5)
𝐾𝑚𝑎𝑐 :=𝑦$← −{0,1}𝜆(6)
Each key is kept separately by the corresponding user.
User information encryption and decryption: For𝑢𝑠𝑒𝑟𝑖with
key𝐾𝑖∈{0,1}𝜆,and given chunk or embedding message as 𝑚𝑖∈
{0,1}∗,If we want to perform AES-CBC calculations, we also need
𝐼𝑉∈{0,1}𝜆for encrypt computation:
𝑐𝑖←𝐴𝐸𝑆𝐶𝐵𝐶.𝐸𝑛𝑐(𝐾𝑖,𝐼𝑉,𝑚𝑖) (7)
For decryption calculations, we use the key 𝐾𝑖, initial vector IV(IV
generates new encryption and decryption each time), and encrypted
text𝑐𝑖as inputs to obtain the plaintext 𝑚𝑝𝑎𝑑after CBC decryption.
𝑚𝑝𝑎𝑑𝑖←𝐴𝐸𝑆𝐶𝐵𝐶.𝐷𝑒𝑐(𝐾𝑖,𝐼𝑉,𝑐𝑖) (8)
When obtaining the text information after padding, verify whether
the decryption is successful through padding:
If∃𝑘∈[1,16]s.t.∀𝑗∈[0,𝑘−1],𝑚𝑝𝑎𝑑[|𝑚𝑝𝑎𝑑|−𝑗]=𝑘:
𝑚𝑖←𝑚𝑝𝑎𝑑[0 :|𝑚𝑝𝑎𝑑|−𝑘]
Else: return⊥(9)
User chunk addition: The user obtains the primary key they hold
based on their user identity. Using the primary key, the embed-
ding and chunk encrypted with the user key 𝐾𝑖are added to the
knowledge base to obtain the primary key corresponding to thenew entry.
𝑃𝑟𝑖𝑣𝐷𝑎𝑡𝑎𝑢𝑠𝑒𝑟.𝐼𝑛𝑠𝑒𝑟𝑡(𝑅𝑒𝑐𝑜𝑟𝑑𝑢𝑠𝑒𝑟=(𝑃𝐾𝑛𝑒𝑤,𝐼𝑉,𝑐𝑒𝑚𝑏𝑒𝑑𝑑𝑖𝑛𝑔,
𝑐𝑐ℎ𝑢𝑛𝑘,𝑡𝑎𝑔=𝐻𝑀𝐴𝐶(𝐾𝑚𝑎𝑐,𝐼𝑉||𝑐𝑒𝑚𝑏𝑒𝑑𝑑𝑖𝑛𝑔||𝑐𝑐ℎ𝑢𝑛𝑘)))(10)
The new primary key is then stored as a 𝑃𝐾𝑖in the PKlist.
𝑃𝐾𝑙𝑖𝑠𝑡←𝑃𝐾𝑙𝑖𝑠𝑡∪𝑃𝐾𝑛𝑒𝑤 (11)
User chunk extraction: The user accesses his own 𝑃𝐾𝑙𝑖𝑠𝑡 =
[𝑃𝐾1,𝑃𝐾 2...𝑃𝐾|𝑃𝐾𝑙𝑖𝑠𝑡|]to retrieve the primary key of the data:
𝑅𝑒𝑐𝑜𝑟𝑑𝑢𝑠𝑒𝑟=𝑃𝑟𝑖𝐷𝑎𝑡𝑎𝑢𝑠𝑒𝑟.𝑆𝑒𝑙𝑒𝑐𝑡(𝑃𝐾=𝑃𝐾𝑡𝑎𝑟𝑔𝑒𝑡) (12)
After retrieving IV, 𝑐𝑖and tag from the 𝑅𝑒𝑐𝑜𝑑𝑒𝑟𝑢𝑠𝑒𝑟 ,using𝐾𝑚𝑎𝑐
calculate data integrity check:
If𝐻𝑀𝐴𝐶(𝐾𝑚𝑎𝑐,𝐼𝑉||𝑐𝑖)≠𝑡𝑎𝑔:𝑟𝑒𝑡𝑢𝑟𝑛⊥ (13)
Decrypt each ciphertext 𝑐𝑖with key𝐾𝑖to obtain plaintext 𝑚𝑖, where
𝑚𝑖can be embedding or chunk:
𝑚𝑖←𝐴𝐸𝑆𝐶𝐵𝐶.𝐷𝑒𝑐(𝐾𝑖,𝐼𝑉,𝑐𝑖) (14)
Finally, all embedding and chunk information are retrieved, which
is combined with the embedding and chunk information of the
public knowledge base for the next rag retrieval.
3.3 Method B: Chained Dynamic Key Derivation
To solve the problem of user privacy disclosure caused by prompt
injection attack in the traditional rag system, Method B proposes
a privacy enhancement scheme based on chain encryption and
dynamic key derivation. The scheme realizes fine-grained access
control and tamper-resistant privacy protection by organizing user
data into encrypted linked lists. Its core design includes four key
steps: user initialization, data encryption storage, privacy data re-
trieval, and privacy data addition.The overall framework of method
B is shown in Figure 4.
User initialization: The first step of user initialization is to bind the
salt value. For the security parameter 𝜆, and each user is randomly
sampled the salt key(taking user A as an example):
𝑘𝑒𝑦𝑠𝑎𝑙𝑡𝐴:=𝑥$← −{0,1}𝜆 (15)
Generate the key 𝐾𝐴,1∈{0,1}𝜆to encrypt the initial node of the
linked list, where 𝑚𝑎𝑠𝑡𝑒𝑟𝑘𝑒𝑦is the main key and the hkdf algorithm
is used for the derivation of the keys:
𝐾𝐴,1←𝐻𝐾𝐷𝐹(𝑚𝑎𝑠𝑡𝑒𝑟𝑘𝑒𝑦,𝑠𝑎𝑙𝑡=𝐼𝐷𝐴,𝑖𝑛𝑓𝑜 =“𝐼𝑛𝑖𝑡𝐾𝑒𝑦 “)(16)
Next, we generate trapdoors, where 𝐼𝐷𝐴is the unique identifier
of the user, 𝐾𝐴,1∈ {0,1}𝜆is the key to encrypt the first node
in the linked list, and 𝑎𝑑𝑑𝑟(𝑛𝑜𝑑𝑒𝐴, 1)is the address to the first
node,ℋ:{0,1}∗→{0,1}𝜆is the hash function will be used.The
threshold is calculated as follows:
𝑇𝑟𝑎𝑝𝑑𝑜𝑜𝑟𝐴=ℋ(𝐼𝐷𝐴||𝑘𝑒𝑦𝑠𝑎𝑙𝑡𝐴)⊕(𝐼𝐷𝐴||𝐾𝐴,1||
𝑎𝑑𝑑𝑟(𝑛𝑜𝑑𝑒𝐴.1))(17)
The trapdoor is stored in the database to authenticate users and
locate user’s privacy information.
Data encryption storage: This part defines the tamper proof
chain encryption storage structure to achieve fine-grained privacy
protection. AES-CBC encryption uses random IV to prevent pattern
recognition, and uses the current key derivation scheme HKDF to

User A
Private Knowledge Base
AES
AES
ChunkA1
ChunkB1
ChunkB2
ChunkA2EmbeddingA1
EmbeddingB1
EmbeddingB2
EmbeddingA2KeyA2
KeyB2
KeyB3
KeyA3
KeyA1
Hash(KeyA1)
Hash(KeyB1)
Hash(KeyB2)
Hash(KeyA3)
AddressA1
HashFigure 4: The figure is the overall framework diagram of method B, in
which the blue and green lines identify the data source, the blue dotted
line represents the data encryption process, the blue (green) dotted line
represents the connection process of the linked list, and the gray line rep-
resents the process in which the user obtains and decrypts the linked list
information through his own identity identifier and key through trapdoor.
Only the encryption scheme is shown in the figure, and 𝐾𝑒𝑦𝐴 1can decrypt
the decryption along the linked list.
derive the encryption key of the next node. The definition of the
storage𝑛𝑜𝑑𝑒𝐴,𝑖is as follows:
𝑛𝑜𝑑𝑒𝐴,𝑖=𝑒𝑚𝑏𝑒𝑑𝑑𝑖𝑛𝑔(𝑐ℎ𝑢𝑛𝑘𝐴,𝑖)||𝑐ℎ𝑢𝑛𝑘𝐴,𝑖||𝐾𝐴,𝑖+1||
ℋ(𝐾𝐴,𝑖)||𝑎𝑑𝑑𝑟(𝑛𝑜𝑑𝑒𝐴,𝑖+1))(18)
In fact, what we store is the encrypted value. We encrypt the current
𝑛𝑜𝑑𝑒𝑖by generating the initial vector 𝐼𝑉∈{0,1}𝜆and using the key
𝐾𝐴,𝑖∈{0,1}𝜆stored by the previous 𝑛𝑜𝑑𝑒[1−4]
𝐴,𝑖,Where𝑛𝑜𝑑𝑒[1−4]
𝐴,𝑖
represents one to four fields of node:
𝐸𝑛𝑐𝑛𝑜𝑑𝑒𝐴,𝑖=𝐴𝐸𝑆𝐶𝐵𝐶.𝐸𝑁𝐶(𝐾𝐴,𝑖,𝐼𝑉,𝑛𝑜𝑑𝑒[1−4]
𝐴,𝑖)||𝑛𝑜𝑑𝑒[5]
𝐴,𝑖(19)
Privacy data retrieval: Users want to retrieve their private infor-
mation, hash the ID and salt key, and perform threshold parsing to
get the address and key of the linked list:
(𝐼𝐷𝐴||𝐾𝐴,1||𝑎𝑑𝑑𝑟(𝑛𝑜𝑑𝑒𝐴.1))=ℋ(𝐼𝐷′
𝐴||𝑘𝑒𝑦𝑠𝑎𝑙𝑡𝐴)⊕
𝑇𝑟𝑎𝑝𝑑𝑜𝑜𝑟𝐴(20)
Then protocol verifies the validity of the user’s identity. Let 𝐼𝐷′
𝐴
denote the queried identifier. The system executes subsequent op-
erations only if 𝐼𝐷′
𝐴matches the pre-registered 𝐼𝐷𝐴, as formalized
in Equation (1). Otherwise, the protocol terminates immediately
(returning⊥) to prevent unauthorized access.
VerifyID(𝐼𝐷′
𝐴)=(
1,if𝐼𝐷′
𝐴=𝐼𝐷𝐴
⊥,otherwise(21)
The decryption protocol operates as a sequential chain traversal,
initiated by decrypting the first node using the initial key 𝐾𝐴,1
and address Addr𝐴,1(Algorithm 1, lines 2–4). Each decrypted node
node𝐴,𝑖undergoes integrity verification through hash comparison
ℋ(𝐾𝐴,𝑖)=node[4]
𝐴,𝑖, where failure triggers immediate termination
(Line 7). Valid nodes yield two critical components: (1) the private
data(embedding𝑖,chunk𝑖)for result aggregation, and (2) the subse-
quent node’s address Addr𝐴,𝑖+1for chain progression. This iterative
decrypt-validate-extract cycle persists until encountering a Nulladdress, at which point Result listcontains all recovered user data,
excluding HMAC padding removal already addressed by Method A.
The full procedure’s formal specification appears in Algorithm 1.
Algorithm 1: Chain Decryption Protocol for Distributed
Encrypted Data
Input : Initial secret key 𝐾𝐴,1, initial address Addr𝐴,1
Output: Decrypted data list Result listcontaining
chunk-embedding pairs
1Result list←∅
2𝑖←1
3while Addr𝐴,𝑖≠Null do
4 Decrypt node:
5 node𝐴,𝑖←
AES-CBC.Decrypt 𝐾𝐴,𝑖,IV,EncNode[1:4]
𝐴,𝑖∥EncNode[5]
𝐴,𝑖
6 Verify integrity:
7 hash calc←ℋ(𝐾𝐴,𝑖)
8 hash stored←node[4]
𝐴,𝑖
9 ifhash calc≠hash stored then
10 return⊥
11 else
12 Extract data:
13(embedding𝑖,chunk𝑖)← node[1:3]
𝐴,𝑖
14 Update results:
15 Result list←Result list∪
(embedding𝑖,chunk𝑖)	
16 Iterate to next node:
17 Addr𝐴,𝑖+1←node[5]
𝐴,𝑖
18 𝑖←𝑖+1
19 end
20end
21return Result list
At this time, the user safely takes out his chunks and correspond-
ing embeddings, and can merge the knowledge base and carry out
rag process.
Privacy Data Addition: First, to add data, the user needs to use
the current encryption node key to derive the key to encrypt the
next node:
𝐾𝐴,𝑖+2←𝐻𝐾𝐷𝐹(𝐾𝐴,𝑖,𝑠𝑎𝑙𝑡=𝐼𝐷𝐴,𝑖𝑛𝑓𝑜 =“𝑁𝑒𝑥𝑡𝐾𝑒𝑦 “) (22)
The new encryption node is constructed as follows:
𝑛𝑜𝑑𝑒𝐴,𝑖+1=𝑒𝑚𝑏𝑒𝑑𝑑𝑖𝑛𝑔(𝑐ℎ𝑢𝑛𝑘𝐴,𝑖+1)||𝑐ℎ𝑢𝑛𝑘𝐴,𝑖+1||𝐾𝐴,𝑖+2||
ℋ(𝐾𝐴,𝑖+1)||𝑛𝑢𝑙𝑙(23)
Then traverse the chain through field 𝑛𝑜𝑑𝑒[5]
𝐴,𝑖to the last node, and
modify null to point to the new node:
𝑛𝑜𝑑𝑒𝐴,𝑖=𝑒𝑚𝑏𝑒𝑑𝑑𝑖𝑛𝑔(𝑐ℎ𝑢𝑛𝑘𝐴,𝑖)||𝑐ℎ𝑢𝑛𝑘𝐴,𝑖||𝐾𝐴,𝑖+1||
ℋ(𝐾𝐴,𝑖)||𝑎𝑑𝑑𝑟(𝑛𝑜𝑑𝑒𝐴,𝑖+1)(24)
The security of the trapdoor is that the enemy cannot extract
effective information through , and the security of the trapdoor
depends on the pseudo randomness of the hash function and the
confidentiality of the salt value .

3.4 RAG Search
In our proposed Retrieval-Augmented Generation (RAG) system,
consider a set of documents {𝐷1,...,𝐷𝑚}. Each document 𝐷𝑖is
partitioned into smaller text chunks. A private knowledge base 𝒦
is constructed by aggregating all these chunks. Let 𝑒denote a text
embedder function that maps each chunk 𝑥𝑧∈𝒦and the input
prompt𝑞into a high - dimensional embedding space R𝑑𝑒𝑚𝑏. We
use a similarity function, for example, the cosine similarity sim(·,·),
which is defined as:
sim(u,v)=u·v
|u||v|(25)
where u,v∈R𝑑𝑒𝑚𝑏. This function is employed to measure the
similarity between the embedding of the prompt q=𝑒(𝑞)and
the embeddings of the chunks x𝑧=𝑒(𝑥𝑧)in𝒦. The top-𝑘most
similar chunks 𝒳(𝑞)⊂𝒦to the prompt 𝑞are retrieved based on the
similarity scores, where |𝒳(𝑞)|=𝑘. We can express the retrieval
process as:
𝒳(𝑞)=argmax 𝒳⊂𝒦,|𝒳|=𝑘∑︁
𝑥𝑧∈𝒳sim(q,x𝑧) (26)
Method A : When adding user chunks to the knowledge base, let
the encrypted embedding be 𝑐𝑒𝑚𝑏𝑒𝑑𝑑𝑖𝑛𝑔 and the encrypted chunk
be𝑐𝑐ℎ𝑢𝑛𝑘 . Along with other related information (e.g., Initialization
Vector𝐼𝑉, tag), they are inserted with a corresponding primary
key𝑃𝐾𝑛𝑒𝑤. The set of primary keys is stored in a list 𝑃𝐾𝑙𝑖𝑠𝑡 . For
retrieval in the encrypted state, the user first retrieves the relevant
primary keys 𝑃𝐾𝑟𝑒𝑙𝑒𝑣𝑎𝑛𝑡⊂𝑃𝐾𝑙𝑖𝑠𝑡 . Let the encrypted data associ-
ated with these primary keys be (𝐼𝑉𝑖,𝑐𝑖,𝑡𝑎𝑔𝑖)for𝑖corresponding
to the retrieved primary keys. The data integrity is checked using a
Message Authentication Code (MAC) with key 𝐾𝑚𝑎𝑐. The integrity
check can be expressed as:
HMAC𝐾𝑚𝑎𝑐(𝐼𝑉𝑖|𝑐𝑖|𝑡𝑎𝑔𝑖)?=received MAC value (27)
where∥denotes concatenation. If the integrity check passes, the
encrypted ciphertexts 𝑐𝑖(either encrypted embeddings or encrypted
chunks) are decrypted using the user’s key 𝐾𝑖. Let𝑑𝑖=Decrypt𝐾𝑖(𝑐𝑖)
be the decrypted data. Then, the decrypted chunks and embeddings
are combined with those from the public knowledge base. The simi-
larity between the embedding of the prompt qand the embeddings
of the combined chunks is calculated to retrieve the top- 𝑘rele-
vant chunks 𝒳(𝑞)for further processing by the generative model.
Method B : The user data is organized into an encrypted linked
list. Each node 𝑛𝑜𝑑𝑒𝐴,𝑖contains the embedding of the chunk x𝐴,𝑖,
the chunk𝑥𝐴,𝑖, the key for the next node 𝐾𝐴,𝑖+1, the hash integrity
check of the current key ℋ(𝐾𝐴,𝑖), and the address of the next node
𝑎𝑑𝑑𝑟(𝑛𝑜𝑑𝑒𝐴,𝑖+1). After encryption, the encrypted nodes 𝐸𝑛𝑐𝑛𝑜𝑑𝑒𝐴,𝑖
are stored. When a query (input prompt 𝑞) is received, the user first
hashes their ID and salt key. Let ℎ=Hash(𝐼𝐷∥𝑠𝑎𝑙𝑡)and parses the
trapdoor𝑇to obtain the address 𝑎𝑑𝑑𝑟 0and key𝐾0of the linked list.
After verifying the user ID, the linked list is decrypted node by node.
For the𝑖-th node, the address 𝑎𝑑𝑑𝑟𝑖is used to locate the next node,
and the key𝐾𝑖is used to decrypt the node’s content. If 𝐸𝑛𝑐𝑛𝑜𝑑𝑒𝐴,𝑖=
(𝐸(x𝐴,𝑖),𝐸(𝑥𝐴,𝑖),𝐸(𝐾𝐴,𝑖+1),𝐸(ℋ(𝐾𝐴,𝑖)),𝐸(𝑎𝑑𝑑𝑟(𝑛𝑜𝑑𝑒𝐴,𝑖+1)))is theencrypted node, then the decrypted node is obtained as:
 
x𝐴,𝑖=Decrypt𝐾𝑖(𝐸(x𝐴,𝑖)),
𝑥𝐴,𝑖=Decrypt𝐾𝑖(𝐸(𝑥𝐴,𝑖)),
𝐾𝐴,𝑖+1=Decrypt𝐾𝑖(𝐸(𝐾𝐴,𝑖+1)),
ℋ(𝐾𝐴,𝑖)=Decrypt𝐾𝑖(𝐸(ℋ(𝐾𝐴,𝑖))),
𝑎𝑑𝑑𝑟(𝑛𝑜𝑑𝑒𝐴,𝑖+1)=Decrypt𝐾𝑖(𝐸(𝑎𝑑𝑑𝑟(𝑛𝑜𝑑𝑒𝐴,𝑖+1))).(28)
Once all the private chunks and their corresponding embeddings are
retrieved, they are combined with the public knowledge base data.
Similar to Method A, the similarity between the prompt embedding
and the combined chunks’ embeddings is calculated to retrieve the
top-𝑘relevant chunks 𝒳(𝑞)for the generative model.
In the search process, for both Method A and Method B, user
authentication is performed before the retrieval phase. In Method
A, each user has a unique encryption key 𝐾𝑖and a message authen-
tication key 𝐾𝑚𝑎𝑐. The data is encrypted using AES - CBC with a
random initialization vector 𝐼𝑉for each encryption operation. The
encryption of data 𝑚can be written as 𝑐=AES - CBC 𝐾𝑖(𝑚,𝐼𝑉).
The integrity of the encrypted data is verified using 𝐻𝑀𝐴𝐶 with
𝐾𝑚𝑎𝑐, which ensures that even if an adversary accesses the en-
crypted data, they cannot decrypt it without the correct key and
cannot modify the data without being detected. In Method B, the
use of a chained dynamic key derivation mechanism, along with the
trapdoor and hash integrity checks for each node in the linked list,
provides fine - grained access control and tamper - resistant privacy
protection. The trapdoor, which is based on the user’s unique ID,
salt value, and the root key of the data chain, ensures that only the
legitimate user can access their private data. The hash integrity
checks prevent any unauthorized modification of the data nodes
during storage and retrieval.
4 SECURITY PROOF
First, we provide a security proof for Method A. The confidentiality
of Method A can be reduced to the IND-CPA (adaptive chosen
plaintext attack) security of AES-CBC, while the integrity relies on
the PRF (pseudorandom function) property and collision resistance
of HMAC. The security proof is given through formalizing the
adversary model and using reduction techniques:
Assume there exists a probabilistic polynomial-time (PPT) ad-
versary 𝒜that can break the IND-CPA security with advantage 𝜖.
We construct a oracle ℬto use𝒜to break the IND-CPA security of
AES-CBC, with the following steps:
(1)ℬreceives the security parameter 𝜆from AES-CBC and other
public parameters.
(2)Key generation: When 𝒜requests a user key, ℬrandomly
samples𝐾𝑖$← −{0,1}𝜆and returns it.
(3)For𝒜’s encryption query (𝑚0,𝑚1),ℬrandomly generates
𝐼𝑉$← −{0,1}𝜆, submits(𝑚0,𝑚1)to the AES-CBC challenger,
and receives the challenge ciphertext,where 𝑏$← −{0,1}:
𝑐𝑏←𝐴𝐸𝑆−𝐶𝐵𝐶.𝐸𝑛𝑐(𝐾,𝐼𝑉,𝑚𝑏) (29)
returning(𝐼𝑉,𝑐𝑏)to𝒜.
(4)𝒜outputs a guess 𝑏′, andℬoutputs the same result.

At this point, the advantage of ℬsatisfies:
𝐴𝑑𝑣𝐼𝑁𝐷−𝐶𝑃𝐴
ℬ(𝜆)=𝐴𝑑𝑣𝐼𝑁𝐷−𝐶𝑃𝐴
𝒜(𝜆) (30)
According to the standard security assumption of AES-CBC, there
is a negligible function 𝑛𝑒𝑙𝑔(𝜆)that makes:
𝐴𝑑𝑣𝐼𝑁𝐷−𝐶𝑃𝐴
ℬ(𝜆)≤𝑛𝑒𝑙𝑔(𝜆) (31)
From equations 30 and 31, it can be concluded that method A also
meets IND-CPA confidentiality.
To demonstrate the integrity (INT-CTXT) security of Method A,
we assume that an adversary 𝒜can forge valid ciphertexts with
advantage AdvINT-CTXT
𝒜(𝜆). To further analyze the implications of
this assumption on the overall security, we construct a oracle ℬthat
leverages the forgery capabilities of 𝒜to break the PRF security of
HMAC. This reduction approach allows us to relate the integrity
security of Method A to the PRF security of HMAC. The process is
described as follows:
(1)The oracle ℬreceives the key 𝐾𝑚𝑎𝑐 (or random function)
from the HMAC challenger.
(2)When the adversary 𝒜makes an HMAC query 𝐼𝑉∥𝑐, the
oracleℬsubmits𝐼𝑉∥𝑐to the HMAC challenger, obtains the
corresponding tag 𝑡𝑎𝑔, and returns it to 𝒜.
(3)When the adversary 𝒜outputs(𝐼𝑉∗,𝑐∗,𝑡𝑎𝑔∗)as forged ci-
phertext , the oracle ℬsubmits(𝐼𝑉∗∥𝑐∗)and𝑡𝑎𝑔∗to the
HMAC challenger for verification. If the forged ciphertext is
validated as valid, then ℬsuccessfully leverages 𝒜’s forgery
capability to break the PRF security of HMAC.
Based on the PRF security of HMAC, the advantage of the oracle
ℬis bounded by the following relation:
AdvPRF
ℬ(𝜆)≥AdvINT-CTXT
𝒜(𝜆)−𝑞2
2𝜆+1(32)
where𝑞is the number of queries made by the adversary 𝒜. Since
HMAC’s security as a PRF guarantees that AdvPRF
ℬ(𝜆)≤negl(𝜆)
(where negl(𝜆)denotes a negligible function that approaches zero
rapidly as𝜆increases), we can derive that:
AdvINT-CTXT
𝒜(𝜆)≤negl(𝜆)+𝑞2
2𝜆+1(33)
When the security parameter 𝜆≥128and the number of queries
𝑞≪264, the right-hand side becomes a negligible value. This
implies that the probability of the adversary 𝒜forging valid cipher-
texts is extremely low, thereby ensuring that Method A satisfies
INT-CTXT integrity.
Through this analysis, we have shown that the integrity security
of Method A can be reduced to the PRF security of HMAC. This not
only validates the security of Method A in practical applications
but also demonstrates its theoretical reliability. For the security
proof of method B, we intend to analyze it from three aspects: the
forward security of the chain node, the chain integrity and the
privacy of trapdoor. The encryption protocol of each node to the
security of AES-CBC has been proved in method A, so this part is
not described.
Chain forward security and even if the adversary obtains the
key𝐾𝐴,𝑖in time step i, it still cannot decrypt the historical node
data𝑛𝑜𝑑𝑒𝐴,𝑗where j>i, the security assumption of HDKF is: if
the input key is 𝐾𝐴,𝑖is a random value, then the output 𝐾𝐴,𝑖+1is computationally indistinguishable from the uniform random
permutation.Assuming that adversary 𝒜a with PPT can decrypt
the historical node 𝑛𝑜𝑑𝑒𝐴,𝑗, we can construct the algorithm ℬto
break the pseudo randomness of HKDF,the protocol is shown as
follows:
(1)ℬaccept the output K* of HKDF challenger.
(2)ℬsimulate the key derivation chain of method B, replace K*
with𝐾𝐴,𝑖, and continue to derive subsequent keys.
(3)If a successfully decrypts 𝑛𝑜𝑑𝑒𝐴,𝑗, then K* must be HKDF
output, otherwise it is a pseudo-random number.
(4)ℬto distinguish between HKDF output and random number,
contradiction and HKDF security assumption.
Therefore, the following conclusions can be drawn,under the as-
sumption of HKDF, method B meets forward security:
𝐴𝑑𝑣𝐹𝑜𝑟𝑤𝑎𝑟𝑑−𝑆𝑒𝑐𝑢𝑟𝑖𝑡𝑦
𝒜(𝜆)≤𝐴𝑑𝑣𝐹𝑅𝐹
𝐻𝐾𝐷𝐹(𝜆)+𝑛𝑒𝑙𝑔(𝜆) (34)
For chain integrity, the adversary cannot tamper with 𝑛𝑜𝑑𝑒𝐴,𝑖
without being detected. It can be reduced to node to perform hash
anti-collision property of decryption key. ℎ𝑖=𝐻(𝐾𝐴,𝑖)has stored in
every node and the next ℎ𝑖+1=𝐻(𝐾𝐴,𝑖+1)form a hash chain,Every
decryption verificate ℎ𝑖?=𝐻(𝐾𝐴,𝑖),Assuming that the content of
the𝑛𝑜𝑑𝑒𝐴,𝑖tampered by the adversary is 𝑛𝑜𝑑𝑒∗
𝐴,𝑖, it is necessary to
modify the hash value of subsequent nodes ℎ∗
𝑖=𝐻(𝐾∗
𝐴,𝑗)at the
same time to pass the verification. Assuming that the adversary
makes at most Q tampering attempts, the probability of successful
forgery is:
𝐴𝑑𝑣𝐶ℎ𝑎𝑖𝑛−𝐼𝑛𝑡𝑒𝑟𝑔𝑟𝑖𝑡𝑦
𝒜(𝜆)≤𝑞(𝑞+1)
2𝜆(35)
Because every tampering requires cracking the hash anti-collision,
when the security parameter 𝜆is large enough( 𝜆≥128),2−𝜆can
be ignored.The chained integrity of method B can depend on the
anti-collision of hash function ℋto meet the data tamperability.
The security of the trapdoor is that the enemy cannot extract
effective information through equation 17, and the security of the
trapdoor depends on the pseudo randomness of the hash func-
tion𝐻(𝐼𝐷𝐴||𝑘𝑒𝑦𝑠𝑎𝑙𝑡𝐴)and the confidentiality of the salt value
𝑘𝑒𝑦𝑠𝑎𝑙𝑡𝐴.Assuming that the enemy already knows the 𝑇𝑟𝑎𝑝𝑑𝑜𝑜𝑟𝐴
but does not know the 𝑘𝑒𝑦𝑠𝑎𝑙𝑡𝐴, it is necessary to restore the 𝐼𝐷𝐴
and𝐾𝐴,1,The advantages of the 𝒜are described as follows:
𝐴𝑑𝑣𝒜=𝑃𝑟[𝒜(𝑇𝑟𝑎𝑝𝑑𝑜𝑜𝑟𝐴)− →(𝐼𝐷𝐴,𝐾𝐴,1)] (36)
The above 𝒜attacks can be regulated to PRF security,the enemy
only knows 𝐼𝐷𝐴, he can construct algorithm ℬto distinguish the
output of𝐻(𝐼𝐷𝐴||𝑘𝑒𝑦𝑠𝑎𝑙𝑡𝐴)from random oracle 𝒪:
(1)ℬrandom sampled 𝑘𝑒𝑦𝑠𝑎𝑙𝑡𝐴$← −{0,1}𝜆and confidential.
(2) When 𝒜request the trapdoor ℬrun:
•Randomly select 𝐼𝐷𝐴← −{0,1}𝜆and𝑘𝑎,1← −{0,1}𝜆.
•Submit𝐼𝐷𝐴||𝑘𝑒𝑦𝑠𝑎𝑙𝑡𝐴to PRF challenger and obtain 𝑦𝑏=
ℋ(𝐼𝐷𝐴||𝑘𝑒𝑦𝑠𝑎𝑙𝑡𝐴)(b=0 for PRF output b=1 is a random
number).
•Compute the 𝑇𝑟𝑎𝑝𝑑𝑜𝑜𝑟𝐴as follow:
𝑇𝑟𝑎𝑝𝑑𝑜𝑜𝑟𝐴=𝑦𝑏⊕(𝐼𝐷𝐴||𝐾𝐴,1||𝑎𝑑𝑑𝑟(𝑛𝑜𝑑𝑒𝐴.1)) (37)
•Return the𝑇𝑟𝑎𝑝𝑑𝑜𝑜𝑟𝐴to𝒜.
(3)𝒜output the result(𝐼𝐷∗
𝐴,𝐾∗
𝐴,1)that he guess.

(4)𝒜judge the result throught compare 𝐾∗
𝐴,1?=𝐾𝐴,1and𝐼𝐷∗
𝐴?=
𝐼𝐷𝐴, If the above conditions are met, return to b=0 otherwise
b=1.
Let’s analyze the advantage of 𝒜. the first case is the probability
that simulator B outputs 0 when the challenger is in PRF mode
(b=0). The second case is the probability that the challenger is in
true random mode (b=1) and the simulator outputs 0. For the first
case𝑦𝑏=𝐻(𝐼𝐷𝐴||𝑘𝑒𝑦𝑠𝑎𝑙𝑡𝐴)is the real PRF output, and 𝑇𝑟𝑎𝑝𝑑𝑜𝑜𝑟𝐴
is constructed legally. The enemy advantage is:
𝑃𝑟[ℬ→0|𝑏=0]=𝐴𝑑𝑣𝒜 (38)
For the second case, if 𝑦𝑏is a uniform random number, then 𝑇𝑟𝑎𝑝𝑑𝑜𝑜𝑟𝑎
is one secret at a time:
𝑇𝑟𝑎𝑝𝑑𝑜𝑜𝑟𝑎=random⊕(𝐼𝐷𝐴||𝐾𝐴,1||𝑎𝑑𝑑𝑟(𝑛𝑜𝑑𝑒𝐴.1) (39)
And the success probability of the enemy is:
𝑃𝑅[𝐵→0|𝑏=1]≤1
2𝜆+(𝑞𝐻+𝑞𝑡𝑟𝑎𝑝)2
2𝜆+1(40)
Where(𝑞𝐻+𝑞𝑡𝑟𝑎𝑝)2
2𝜆+1 is the probability that at least one collision
will occur when the 𝒜makes𝑞𝐻+𝑞𝑡𝑟𝑎𝑝 queries according to the
birthday paradox.Where 𝑞𝐻refers to the number of times the 𝒜
performed hash queries and 𝑞𝑡𝑟𝑎𝑝 refers to the number of trapdoor
instances obtained by the 𝒜.So the overall advantage of the enemy’s
attack is:
𝐴𝑑𝑣𝑃𝑅𝐹
ℬ=|𝑃𝑟[𝐵→0|𝑏=0]−𝑃𝑟[𝐵→1|𝑏=0]|
≥𝐴𝑑𝑣𝒜−(𝑞𝐻+𝑞𝑡𝑟𝑎𝑝)2
2𝜆+1(41)
𝐴𝑑𝑣𝒜=𝐴𝑑𝑣𝑃𝑅𝐹
ℬ+(𝑞𝐻+𝑞𝑡𝑟𝑎𝑝)2
2𝜆+1(42)
When the security parameter 𝜆≥128and the number of queries
𝑞≪264, the right-hand side becomes a negligible value. So the
advantage of the enemy is very small, and the trapdoor informa-
tion can be completely hidden, so as to ensure the concealment of
trapdoor.
5 CONCLUSION
In this work, we have tackled the pressing privacy and security con-
cerns associated with Retrieval-Augmented Generation (RAG) sys-
tems, which are increasingly deployed in sensitive domains such as
healthcare, finance, and legal services. By introducing an advanced
encryption methodology that secures both textual content and its
corresponding embeddings, we have established a robust frame-
work to protect proprietary knowledge bases from unauthorized
access and data breaches. Our approach ensures that sensitive infor-
mation remains encrypted throughout the retrieval and generation
processes, without compromising the performance or functionality
of RAG pipelines. The key contributions of this research include the
development of a theoretical framework for integrating encryption
techniques into RAG systems and a novel encryption scheme that
secures RAG knowledge bases at both the textual and embedding
levels. Through rigorous security proofs, we have demonstrated
the resilience of our methodology against potential threats and
vulnerabilities, validating its effectiveness in safeguarding sensi-
tive information. This validates the practicality and effectivenessof our solution in real-world applications. Our findings highlight
the critical need for integrating advanced encryption techniques
into the design and deployment of RAG systems as a fundamental
component of privacy safeguards. By addressing the vulnerabilities
identified in prior research, this work advances the state-of-the-art
in RAG security and contributes to the broader discourse on enhanc-
ing privacy-preserving measures in AI-driven services. We believe
that our contributions will inspire further innovation in this critical
area, ultimately fostering greater trust and reliability in AI-driven
applications across diverse sectors. In conclusion, this research not
only provides a practical solution to mitigate privacy risks in RAG
systems but also sets a new benchmark for the development of
secure and privacy-preserving AI technologies. Future work will
explore the scalability of our encryption scheme to larger datasets
and its integration with other AI architectures, further solidifying
its role in the evolving landscape of AI security.
REFERENCES
[1]Rakesh Agrawal, Jerry Kiernan, Ramakrishnan Srikant, and Yirong Xu. 2004.
Order preserving encryption for numeric data. In Proceedings of the 2004 ACM
SIGMOD international conference on Management of data . 563–574.
[2]Yejin Bang, Samuel Cahyawijaya, Nayeon Lee, Wenliang Dai, Dan Su, Bryan
Wilie, Holy Lovenia, Ziwei Ji, Tiezheng Yu, Willy Chung, et al .2023. A multitask,
multilingual, multimodal evaluation of chatgpt on reasoning, hallucination, and
interactivity. arXiv preprint arXiv:2302.04023 (2023).
[3]Rudolf Bayer and JK Metzger. 1976. On the encipherment of search trees and
random access files. ACM Transactions on Database Systems (TODS) 1, 1 (1976),
37–52.
[4]Stella Biderman, Usvsn Prashanth, Lintang Sutawika, Hailey Schoelkopf, Quentin
Anthony, Shivanshu Purohit, and Edward Raff. 2023. Emergent and predictable
memorization in large language models. Advances in Neural Information Process-
ing Systems 36 (2023), 28072–28090.
[5]Luc Bouganim and Philippe Pucheral. 2002. Chip-secured data access: Confiden-
tial data on untrusted servers. In VLDB’02: Proceedings of the 28th International
Conference on Very Large Databases . Elsevier, 131–142.
[6]Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan,
Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, et al .2020. Language models are few-shot learners. Advances in neural
information processing systems 33 (2020), 1877–1901.
[7]Nicholas Carlini, Steve Chien, Milad Nasr, Shuang Song, Andreas Terzis, and
Florian Tramer. 2022. Membership inference attacks from first principles. In 2022
IEEE symposium on security and privacy (SP) . IEEE, 1897–1914.
[8]Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian
Tramer, and Chiyuan Zhang. 2022. Quantifying memorization across neural
language models. In The Eleventh International Conference on Learning Represen-
tations .
[9]Nicholas Carlini, Florian Tramer, Eric Wallace, Matthew Jagielski, Ariel Herbert-
Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Ulfar Erlingsson,
et al.2021. Extracting training data from large language models. In 30th USENIX
security symposium (USENIX Security 21) . 2633–2650.
[10] Chin-Chen Chang and Chao-Wen Chan. 2003. A database record encryption
scheme using the RSA public key cryptosystem and its master keys. In 2003
International Conference on Computer Networks and Mobile Computing, 2003.
ICCNMC 2003. IEEE, 345–348.
[11] Harrison Chase. 2022. Langchain. https://github.com/hwchase17/langchain. (Oct.
2022).
[12] Gang Chen, Ke Chen, and Jinxiang Dong. 2006. A database encryption scheme
for enhanced security and easy sharing. In 2006 10th International Conference on
Computer Supported Cooperative Work in Design . IEEE, 1–6.
[13] Xin Cheng, Di Luo, Xiuying Chen, Lemao Liu, Dongyan Zhao, and Rui Yan.
2023. Lift yourself up: Retrieval-augmented text generation with self-memory.
Advances in Neural Information Processing Systems 36 (2023), 43780–43799.
[14] Stav Cohen, Ron Bitton, and Ben Nassi. 2024. Unleashing worms and extracting
data: Escalating the outcome of attacks against rag-based inference in scale and
severity using jailbreaking. arXiv preprint arXiv:2409.08045 (2024).
[15] Adam Cutbill, Eric Monsler, and Eric Hayashi. 2024. Personalized home assistant
using large language model with context-based chain of thought reasoning.
(2024).
[16] George I Davida, David L Wells, and John B Kam. 1981. A database encryption
system with subkeys. ACM Transactions on Database Systems (TODS) 6, 2 (1981),
312–328.

[17] Matthias De Lange, Rahaf Aljundi, Marc Masana, Sarah Parisot, Xu Jia, Aleš
Leonardis, Gregory Slabaugh, and Tinne Tuytelaars. 2021. A continual learning
survey: Defying forgetting in classification tasks. IEEE transactions on pattern
analysis and machine intelligence 44, 7 (2021), 3366–3385.
[18] Christian Di Maio, Cristian Cosci, Marco Maggini, Valentina Poggioni, and Ste-
fano Melacci. 2024. Pirates of the RAG: Adaptively Attacking LLMs to Leak
Knowledge Bases. arXiv preprint arXiv:2412.18295 (2024).
[19] Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Jingyuan Ma, Rui Li, Heming Xia,
Jingjing Xu, Zhiyong Wu, Tianyu Liu, et al .2022. A survey on in-context learning.
arXiv preprint arXiv:2301.00234 (2022).
[20] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai,
Jiawei Sun, Haofen Wang, and Haofen Wang. 2023. Retrieval-augmented gen-
eration for large language models: A survey. arXiv preprint arXiv:2312.10997 2
(2023).
[21] Silvia García-Méndez, Francisco de Arriba-Pérez, and María del Carmen Somoza-
López. 2024. A review on the use of large language models as virtual tutors.
Science & Education (2024), 1–16.
[22] Abenezer Golda, Kidus Mekonen, Amit Pandey, Anushka Singh, Vikas Hassija,
Vinay Chamola, and Biplab Sikdar. 2024. Privacy and security concerns in
generative AI: a comprehensive survey. IEEE Access (2024).
[23] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. 2020.
Retrieval augmented language model pre-training. In International conference on
machine learning . PMLR, 3929–3938.
[24] Jingmin He and Min Wang. 2001. Cryptography and relational database man-
agement systems. In Proceedings 2001 International Database Engineering and
Applications Symposium . IEEE, 273–284.
[25] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean
Wang, Lu Wang, Weizhu Chen, et al .2022. Lora: Low-rank adaptation of large
language models. ICLR 1, 2 (2022), 3.
[26] Hongsheng Hu, Zoran Salcic, Lichao Sun, Gillian Dobbie, Philip S Yu, and Xuyun
Zhang. 2022. Membership inference attacks on machine learning: A survey. ACM
Computing Surveys (CSUR) 54, 11s (2022), 1–37.
[27] Yaou Hu and Hyounae Kelly Min. 2023. The dark side of artificial intelligence in
service: The “watching-eye” effect and privacy concerns. International Journal of
Hospitality Management 110 (2023), 103437.
[28] Min-Shiang Hwang and Wei-Pang Yang. 1997. Multilevel secure database en-
cryption with subkeys. Data & knowledge engineering 22, 2 (1997), 117–131.
[29] Juyong Jiang, Fan Wang, Jiasi Shen, Sungju Kim, and Sunghun Kim. 2024. A survey
on large language models for code generation. arXiv preprint arXiv:2406.00515
(2024).
[30] Ehsan Kamalloo, Nouha Dziri, Charles LA Clarke, and Davood Rafiei. 2023.
Evaluating open-domain question answering in the era of large language models.
arXiv preprint arXiv:2305.06984 (2023).
[31] Poul-Henning Kamp. 2003. {GBDE—GEOM}Based Disk Encryption. In BSDCon
2003 (BSDCon 2003) .
[32] Nikhil Kandpal, Eric Wallace, and Colin Raffel. 2022. Deduplicating training
data mitigates privacy risks in language models. In International Conference on
Machine Learning . PMLR, 10697–10707.
[33] Enkelejda Kasneci, Kathrin Seßler, Stefan Küchemann, Maria Bannert, Daryna
Dementieva, Frank Fischer, Urs Gasser, Georg Groh, Stephan Günnemann, Eyke
Hüllermeier, et al .2023. ChatGPT for good? On opportunities and challenges
of large language models for education. Learning and individual differences 103
(2023), 102274.
[34] Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck,
Chris Callison-Burch, and Nicholas Carlini. 2021. Deduplicating training data
makes language models better. arXiv preprint arXiv:2107.06499 (2021).
[35] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
et al.2020. Retrieval-augmented generation for knowledge-intensive nlp tasks.
Advances in neural information processing systems 33 (2020), 9459–9474.
[36] Qian Li, Hao Peng, Jianxin Li, Congying Xia, Renyu Yang, Lichao Sun, Philip S
Yu, and Lifang He. 2022. A survey on text classification: From traditional to deep
learning. ACM Transactions on Intelligent Systems and Technology (TIST) 13, 2
(2022), 1–41.
[37] Yinheng Li. 2023. A practical survey on zero-shot prompt design for in-context
learning. arXiv preprint arXiv:2309.13205 (2023).
[38] Yong Lin, Hangyu Lin, Wei Xiong, Shizhe Diao, Jianmeng Liu, Jipeng Zhang, Rui
Pan, Haoxiang Wang, Wenbin Hu, Hanning Zhang, et al .2023. Mitigating the
alignment tax of rlhf. arXiv preprint arXiv:2309.06256 (2023).
[39] S Merhotra and B Gore. 2009. A Middleware approach for managing and of
outsourced personal data. In NSF Workshop on Data and Application Security,
Arlignton, Virginia .
[40] Fatemehsadat Mireshghallah, Archit Uniyal, Tianhao Wang, David Evans, and
Taylor Berg-Kirkpatrick. 2022. Memorization in nlp fine-tuning methods. arXiv
preprint arXiv:2205.12506 (2022).
[41] Dimitrios P Panagoulias, Maria Virvou, and George A Tsihrintzis. 2024. Augment-
ing large language models with rules for enhanced domain-specific interactions:
The case of medical diagnosis. Electronics 13, 2 (2024), 320.[42] Jongjin Park. 2024. Development of dental consultation chatbot using retrieval
augmented llm. The Journal of the Institute of Internet, Broadcasting and Commu-
nication 24, 2 (2024), 87–92.
[43] Zhenting Qi, Hanlin Zhang, Eric Xing, Sham Kakade, and Himabindu Lakkaraju.
2024. Follow my instruction and spill the beans: Scalable data extraction from
retrieval-augmented generation systems. arXiv preprint arXiv:2402.17840 (2024).
[44] Mahimai Raja, E Yuvaraajan, et al .2024. A rag-based medical assistant especially
for infectious diseases. In 2024 International Conference on Inventive Computation
Technologies (ICICT) . IEEE, 1128–1133.
[45] Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin
Leyton-Brown, and Yoav Shoham. 2023. In-context retrieval-augmented language
models. Transactions of the Association for Computational Linguistics 11 (2023),
1316–1331.
[46] Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie Huang, Nan Duan, and Weizhu
Chen. 2023. Enhancing retrieval-augmented large language models with iterative
retrieval-generation synergy. arXiv preprint arXiv:2305.15294 (2023).
[47] Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Rich James, Mike
Lewis, Luke Zettlemoyer, and Wen-tau Yih. 2023. Replug: Retrieval-augmented
black-box language models. arXiv preprint arXiv:2301.12652 (2023).
[48] Erez Shmueli, Ronen Waisenberg, Yuval Elovici, and Ehud Gudes. 2005. Designing
secure indexes for encrypted databases. In Data and Applications Security XIX:
19th Annual IFIP WG 11.3 Working Conference on Data and Applications Security,
Storrs, CT, USA, August 7-10, 2005. Proceedings 19 . Springer, 54–68.
[49] Reza Shokri, Marco Stronati, Congzheng Song, and Vitaly Shmatikov. 2017. Mem-
bership inference attacks against machine learning models. In 2017 IEEE sympo-
sium on security and privacy (SP) . IEEE, 3–18.
[50] Kurt Shuster, Spencer Poff, Moya Chen, Douwe Kiela, and Jason Weston. 2021.
Retrieval augmentation reduces hallucination in conversation. arXiv preprint
arXiv:2104.07567 (2021).
[51] Dave Van Veen, Cara Van Uden, Louis Blankemeier, Jean-Benoit Delbrouck, Asad
Aali, Christian Bluethgen, Anuj Pareek, Malgorzata Polacin, Eduardo Pontes Reis,
Anna Seehofnerová, et al .2024. Adapted large language models can outperform
medical experts in clinical text summarization. Nature medicine 30, 4 (2024),
1134–1142.
[52] Ziyu Wang, Hao Li, Di Huang, and Amir M Rahmani. 2024. Healthq: Unveiling
questioning capabilities of llm chains in healthcare conversations. arXiv preprint
arXiv:2409.19487 (2024).
[53] Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian
Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al .
2022. Emergent abilities of large language models. arXiv preprint arXiv:2206.07682
(2022).
[54] Christopher Wood, Eduardo B. Fernandez, and Rita C. Summers. 1980. Data base
security: requirements, policies, and models. IBM Systems Journal 19, 2 (1980),
229–252.
[55] Zihan Yu, Liang He, Zhen Wu, Xinyu Dai, and Jiajun Chen. 2023. Towards better
chain-of-thought prompting strategies: A survey. arXiv preprint arXiv:2310.04959
(2023).
[56] Shenglai Zeng, Yaxin Li, Jie Ren, Yiding Liu, Han Xu, Pengfei He, Yue Xing,
Shuaiqiang Wang, Jiliang Tang, and Dawei Yin. 2023. Exploring memorization in
fine-tuned language models. arXiv preprint arXiv:2310.06714 (2023).
[57] Shenglai Zeng, Jiankun Zhang, Pengfei He, Yue Xing, Yiding Liu, Han Xu, Jie
Ren, Shuaiqiang Wang, Dawei Yin, Yi Chang, et al .2024. The good and the bad:
Exploring privacy issues in retrieval-augmented generation (rag). arXiv preprint
arXiv:2402.16893 (2024).
[58] Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu, Tingchen Fu, Xinting
Huang, Enbo Zhao, Yu Zhang, Yulong Chen, et al .2023. Siren’s song in the
AI ocean: a survey on hallucination in large language models. arXiv preprint
arXiv:2309.01219 (2023).
[59] Yujia Zhou, Yan Liu, Xiaoxi Li, Jiajie Jin, Hongjin Qian, Zheng Liu, Chaozhuo Li,
Zhicheng Dou, Tsung-Yi Ho, and Philip S Yu. 2024. Trustworthiness in retrieval-
augmented generation systems: A survey. arXiv preprint arXiv:2409.10102 (2024).
[60] Wenhao Zhu, Hongyi Liu, Qingxiu Dong, Jingjing Xu, Shujian Huang, Lingpeng
Kong, Jiajun Chen, and Lei Li. 2023. Multilingual machine translation with large
language models: Empirical results and analysis. arXiv preprint arXiv:2304.04675
(2023).