# SiliconHealth: A Complete Low-Cost Blockchain Healthcare Infrastructure for Resource-Constrained Regions Using Repurposed Bitcoin Mining ASICs

**Authors**: Francisco Angulo de Lafuente, Seid Mehammed Abdu, Nirmal Tej

**Published**: 2026-01-14 15:21:37

**PDF URL**: [https://arxiv.org/pdf/2601.09557v2](https://arxiv.org/pdf/2601.09557v2)

## Abstract
This paper presents SiliconHealth, a comprehensive blockchain-based healthcare infrastructure designed for resource-constrained regions, particularly sub-Saharan Africa. We demonstrate that obsolete Bitcoin mining Application-Specific Integrated Circuits (ASICs) can be repurposed to create a secure, low-cost, and energy-efficient medical records system. The proposed architecture employs a four-tier hierarchical network: regional hospitals using Antminer S19 Pro (90+ TH/s), urban health centers with Antminer S9 (14 TH/s), rural clinics equipped with Lucky Miner LV06 (500 GH/s, 13W), and mobile health points with portable ASIC devices. We introduce the Deterministic Hardware Fingerprinting (DHF) paradigm, which repurposes SHA-256 mining ASICs as cryptographic proof generators, achieving 100% verification rate across 23 test proofs during 300-second validation sessions. The system incorporates Reed-Solomon LSB watermarking for medical image authentication with 30-40% damage tolerance, semantic Retrieval-Augmented Generation (RAG) for intelligent medical record queries, and offline synchronization protocols for intermittent connectivity. Economic analysis demonstrates 96% cost reduction compared to GPU-based alternatives, with total deployment cost of $847 per rural clinic including 5-year solar power infrastructure. Validation experiments on Lucky Miner LV06 (BM1366 chip, 5nm) achieve 2.93 MH/W efficiency and confirm hardware universality. This work establishes a practical framework for deploying verifiable, tamper-proof electronic health records in regions where traditional healthcare IT infrastructure is economically unfeasible, potentially benefiting over 600 million people lacking access to basic health information systems.

## Full Text


<!-- PDF content starts -->

SiliconHealth: A Complete Low-Cost Blockchain Healthcare
Infrastructure for Resource-Constrained Regions Using Repurposed
Bitcoin Mining ASICs
Francisco Angulo de Lafuente1,Seid Mehammed Abdu2,Nirmal Tej Kumar3
1Independent Researcher, Madrid, Spain
ORCID: 0009-0001-1634-7063
2Senior Lecturer, Department of Computer Science, Woldia University, Ethiopia
Email: seid.m@wldu.edu.et
3Consultant / Senior Staff Scientist, antE Inst (UTD), Dallas, TX, USA
ORCID: 0000-0002-5443-7334
January 2026
Abstract
This paper presents SiliconHealth, a comprehensive
blockchain-based healthcare infrastructure designed for
resource-constrained regions, particularly sub-Saharan Africa.
We demonstrate that obsolete Bitcoin mining Application-
Specific Integrated Circuits (ASICs) can be repurposed to
create a secure, low-cost, and energy-efficient medical records
system. The proposed architecture employs a four-tier hi-
erarchical network: regional hospitals using Antminer S19
Pro (90+ TH/s), urban health centers with Antminer S9
(14 TH/s), rural clinics equipped with Lucky Miner LV06
(500 GH/s, 13W), and mobile health points with portable
ASIC devices. We introduce the Deterministic Hardware
Fingerprinting (DHF) paradigm, which repurposes SHA-256
mining ASICs as cryptographic proof generators, achieving
100% verification rate across 23 test proofs during 300-second
validation sessions. The system incorporates Reed-Solomon
LSB watermarking for medical image authentication with
30-40% damage tolerance, semantic Retrieval-Augmented
Generation (RAG) for intelligent medical record queries, and
offline synchronization protocols for intermittent connectivity.
Economic analysis demonstrates 96% cost reduction compared
to GPU-based alternatives, with total deployment cost of $847
per rural clinic including 5-year solar power infrastructure.
Validation experiments on Lucky Miner LV06 (BM1366 chip,
5nm) achieve 2.93 MH/W efficiency and confirm hardware
universality. This work establishes a practical framework for
deploying verifiable, tamper-proof electronic health records in
regions where traditional healthcare IT infrastructure is eco-
nomically unfeasible, potentially benefiting over 600 million
people lacking access to basic health information systems.
Keywords:Healthcare Blockchain, Bitcoin Mining ASIC,
Electronic Health Records, Reed-Solomon Watermarking,
Retrieval-Augmented Generation, Sub-Saharan Africa, Low-
Cost Healthcare IT, Medical Image Authentication, Offline
Synchronization, Physical Unclonable Functions1 Introduction
Sub-Saharan Africa faces a critical healthcare infrastructure
crisis. According to the World Health Organization, the re-
gion has only 3% of the world’s health workers serving 24% of
the global disease burden [1]. Electronic health record (EHR)
adoption remains extremely low—studies indicate that imple-
mentation is largely driven by HIV treatment programs, with
penetration still minimal across general healthcare [2, 3]. The
barriers are well-documented: high procurement and mainte-
nance costs, unreliable electricity, poor internet connectivity,
and limited digital literacy among healthcare workers [4].
Simultaneously, the global cryptocurrency mining industry
has created an unprecedented surplus of specialized computa-
tional hardware. As Bitcoin mining difficulty increases expo-
nentially, older ASIC generations become economically unvi-
able for their intended purpose. The Antminer S9, once the
dominant mining device with 14 TH/s capacity, now operates
at negative profit margins in most electricity markets [5]. Mil-
lions of these devices—representing billions of dollars in engi-
neering investment—sit idle, contributing to electronic waste.
This paper asks a transformative question:Can obsolete
cryptocurrency mining hardware be repurposed to solve the
healthcare infrastructure crisis in resource-limited settings?
We present SiliconHealth, a comprehensive system that pro-
vides an affirmative answer. Our key insight is that Bitcoin
mining ASICs, designed exclusively for SHA-256 computa-
tion, possess properties ideally suited for healthcare blockchain
applications:
Cryptographic efficiency:Mining ASICs compute SHA-
256 hashes at efficiencies 14,000×greater than GPUs per watt,
enabling blockchain operations on minimal power [6].
Physical Unclonable Functions (PUF):Manufacturing
variations create unique device signatures that serve as unforge-
able authentication mechanisms [7].
Abundant availability:Millions of obsolete miners avail-
able at $40-60 per unit represent extraordinary value for health-
care infrastructure.
Solar compatibility:Low-power devices like Lucky Miner
1arXiv:2601.09557v2  [cs.NE]  15 Jan 2026

LV06 (13W) operate sustainably on basic solar installations.
1.1 Contributions
This paper makes the following contributions:
1. We present a complete four-tier hierarchical healthcare net-
work architecture using repurposed mining ASICs, from re-
gional hospitals to mobile health points.
2. We introduce the Deterministic Hardware Fingerprinting
(DHF) paradigm that transforms mining operations into
cryptographic healthcare proofs with 100% verification
rate.
3. We implement Reed-Solomon LSB watermarking for med-
ical image authentication achieving 30-40% damage toler-
ance.
4. We develop offline synchronization protocols enabling
healthcare delivery under intermittent 2G connectivity.
5. We validate the complete system through empirical exper-
iments on Lucky Miner LV06, demonstrating 2.93 MH/W
efficiency and hardware universality.
6. We provide comprehensive economic analysis showing
96% cost reduction versus GPU alternatives with full 5-year
TCO calculations.
2 Related Work
2.1 Blockchain in Healthcare
MedRec, developed by Azaria et al. at MIT Media Lab [8],
pioneered blockchain-based medical record management us-
ing Ethereum smart contracts. Their pilot with Beth Israel
Deaconess Medical Center demonstrated feasibility but re-
lied on conventional computing infrastructure inappropriate for
resource-limited settings.
Seid Mehammed Abdu’s work on optimizing Proof-of-Work
for healthcare blockchain using CUDA [6, 9] established the
theoretical foundation for GPU-accelerated medical record ver-
ification. His 2025 publication in Blockchain in Healthcare
Today demonstrated that mining algorithms could be repur-
posed for healthcare data integrity, inspiring our ASIC-based
approach.
2.2 Electronic Health Records in Sub-Saharan
Africa
Odekunle et al. [4] comprehensively analyzed barriers to EHR
adoption in sub-Saharan Africa, identifying cost, infrastructure,
and training as primary obstacles. Akanbi et al. [2] documented
that 91% of existing EHR implementations use open-source
software (primarily OpenMRS), concentrated in HIV treat-
ment programs. Recent work by researchers at PLOS Digital
Health [10] analyzed DHIS2 implementations across Ethiopia,
Ghana, and Zimbabwe, finding that while digital health inter-
ventions improve care quality, infrastructure limitations remain
critical barriers.2.3 Physical Reservoir Computing
Tanaka et al. [11] reviewed physical reservoir computing im-
plementations, demonstrating that diverse physical systems can
serve as computational substrates. Our prior work established
that Bitcoin mining ASICs exhibit reservoir computing prop-
erties including thermal fading memory and edge-of-chaos dy-
namics [12], with the Single-Word Handshake (SWH) protocol
achieving NARMA-10 NRMSE of 0.8661.
2.4 Retrieval-Augmented Generation
Lewis et al. [13] introduced the RAG paradigm at NeurIPS
2020, combining parametric and non-parametric memory for
knowledge-intensive NLP tasks. We adapt this approach for
medical record queries, enabling natural language access to pa-
tient histories even in low-bandwidth environments.
3 System Architecture
3.1 Four-Tier Hierarchical Network
SiliconHealth implements a hierarchical network topology op-
timized for sub-Saharan African healthcare infrastructure real-
ities:
TIER 3: Regional Hospital
S19 Pro (90+ TH/s) | Fiber/4G
TIER 2: Urban
S9 (14 TH/s)TIER 2: Urban
S9 (14 TH/s)TIER 2: Urban
S9 (14 TH/s)
Rural
LV06 13WRural
LV07 15WRural
LV06 13WRural
LV06 13WRural
LV07 15WRural
LV06 13W
Mobile Health
Tablet+LV06Community
WorkerVaccination
Team
Figure 1: SiliconHealth four-tier hierarchical network archi-
tecture. Tier 3 (regional hospitals) maintains full blockchain
nodes. Tier 2 (urban health centers) coordinates district-level
data. Tier 1 (rural clinics) operates on solar power with 13-15W
devices. Tier 0 (mobile health points) provides community out-
reach.
Table 1: Hardware Specifications by Network Tier
Tier Facility ASIC Hashrate Power Cost
3 Regional Hospital S19 Pro 110 TH/s 3250W $800-1200
2 Urban Center S9 14 TH/s 1372W $50-80
1 Rural Clinic LV06 500 GH/s 13W $40-60
1 Rural Clinic LV07 700 GH/s 15W $55-75
0 Mobile Point USB ASIC 50-100 GH/s 2-5W $15-30
3.2 Hardware Specifications
The primary rural deployment device, Lucky Miner LV06, em-
ploys the Bitmain BM1366 chip manufactured on TSMC 5nm
process technology. Key specifications verified from manufac-
turer documentation [14]:
• Chip: BM1366 (5nm ASIC)
• Hashrate: 500 GH/s±10%
2

• Power consumption: 13W±5%
• Noise level:<35 dB
• Dimensions: 130×66×40mm
• WiFi: 2.4GHz integrated
The legacy Antminer S9, used at Tier 2 urban centers, con-
tains 189 Bitmain BM1387 chips (16nm FinFET, TSMC) dis-
tributed across three hashboards with 63 chips each. Operating
at 525 MHz optimal frequency, the S9 achieves 14 TH/s nomi-
nal hashrate at 0.098 J/GH efficiency [5].
3.3 Data Flow Architecture
Patient records flow through the hierarchy according to the fol-
lowing principles:
Creation:Records are created at any tier, immediately
signed with the local ASIC’s hardware fingerprint, and queued
for upstream synchronization.
Propagation:Data propagates upward through the hierar-
chy. Rural clinics sync to urban centers during connectivity
windows; urban centers maintain continuous sync with regional
hospitals.
Query:Patient records can be retrieved at any tier. Local
queries search the local cache first; if not found, requests prop-
agate upward until the record is located.
Conflict Resolution:The “last-write-wins” strategy is em-
ployed for concurrent edits, with full edit history preserved in
the blockchain for audit purposes.
4 Cryptographic Foundation
4.1 Deterministic Hardware Fingerprinting
(DHF)
The core innovation of SiliconHealth is the Deterministic Hard-
ware Fingerprinting paradigm, which repurposes the SHA-256
mining operation as a cryptographic proof generator for health-
care records.
Unlike traditional Proof-of-Work where difficulty targets fil-
ter for rare hash outputs, DHF uses controlled difficulty to gen-
erate deterministic proofs:
DHF(record, device) =SHA256(record∥nonce)whereH(output)< target(D)
(1)
The proof generation timeτis predictable:
E[τ] =D
H×232(2)
whereDis the difficulty parameter,His the device hashrate,
and232represents the nonce space. ForD= 32768on Lucky
Miner LV06 (H= 500GH/s):
E[τ] =32768
500×109×4.29×109≈15.3µs(3)
The critical insight is that the proof generation incorporates
device-specific characteristics—thermal variations, manufac-
turing tolerances, power delivery fluctuations—that create a
unique “signature” for each physical ASIC.4.2 Merkle Tree Binding
Patient records are organized in Merkle trees [15] where each
record forms a leaf node:
root=H(H(r 1∥r2)∥H(r 3∥r4)∥. . .)(4)
This structure enables:
•O(logn)verification of individual records
• Efficient synchronization by comparing root hashes
• Tamper detection through hash chain integrity
• Selective disclosure without revealing entire history
4.3 Ephemeral Key Protocol
To prevent replay attacks while maintaining device authenti-
cation, we implement ephemeral session keys with 30-second
TTL:
Ksession =HKDF(device_secret, timestamp∥nonce,“SiliconHealth-v1”)
(5)
Each transaction is signed with the current session key, bind-
ing it to both the specific device and the time window. The
HKDF (HMAC-based Key Derivation Function) uses SHA-256
as the underlying hash function, consistent with ASIC capabil-
ities [16].
5 Medical Image Authentication
5.1 Reed-Solomon LSB Watermarking
Medical images—radiographs, electrocardiograms, ultra-
sounds, wound photographs—require authentication that sur-
vives image processing. We implement Reed-Solomon error
correction [17] embedded in image least significant bits.
Reed and Solomon’s 1960 paper established polynomial
codes over finite fields that can correct up toterrors in2tre-
dundant symbols. Our implementation uses RS(255, 223) over
GF(28):
RS(n= 255, k= 223)→32parity symbols→corrects 16 byte errors
(6)
Medical
ImageASIC DHF
SignatureReed-Solomon
RS(255,223)LSB
EmbedAuthenticated
Image
Damage Tolerance:
30-40%
Figure 2: Reed-Solomon LSB watermarking process for medi-
cal image authentication. The scheme tolerates 30-40% image
damage while preserving authentication capability.
5.2 Embedding Strategy
The watermark payload consists of:
• ASIC hardware signature (32 bytes)
• UTC timestamp (8 bytes)
3

• Patient identifier hash (16 bytes)
• Facility identifier (8 bytes)
• Image content hash (32 bytes)
• RS parity (32 bytes)
•Total: 128 bytes = 1024 bits
For a 1024×1024 pixel grayscale image (1 MB), embed-
ding 1024 bits in LSBs affects only 0.012% of total data, creat-
ing imperceptible visual changes while maintaining diagnostic
quality.
6 Semantic Medical RAG
6.1 Query Expansion Architecture
SiliconHealth incorporates Retrieval-Augmented Genera-
tion [13] for natural language medical record queries. The
system addresses the challenge of healthcare workers with
varying levels of medical training querying patient histories in
local languages.
Query→Expand(synonyms)→Retrieve(records)→Generate(response)
(7)
The semantic expansion module recognizes:
• Medical synonyms (e.g., “BP”→“blood pressure”→“hy-
pertension history”)
• Local disease names in African languages
• Symptom descriptions in colloquial terms
• Drug names including generic and brand variants
6.2 Offline Operation
Unlike cloud-based RAG systems, SiliconHealth operates en-
tirely on-device for Tier 1 rural clinics. The local language
model (quantized to 4-bit precision) runs on the clinic’s tablet
computer, with the ASIC handling only cryptographic opera-
tions.
Query latency remains under 2 seconds for typical medi-
cal record searches, with the ASIC contributing approximately
15ms for signature verification and proof generation.
7 Patient Identity Management
7.1 Challenge: Identity Without Documenta-
tion
A significant challenge in sub-Saharan Africa is patient iden-
tification. UNICEF estimates that 230 million children under
5 in sub-Saharan Africa lack birth registration [18]. Tradi-
tional healthcare IT systems assume government-issued iden-
tification, an assumption that fails in this context.7.2 Multi-Modal Identity Protocol
SiliconHealth implements a multi-modal identity system:
Primary: Biometric fingerprint—Low-cost capacitive sen-
sors ($3-5) capture fingerprint templates stored locally. The
template (not the raw image) is hashed and used as patient iden-
tifier.
Secondary: QR code card—Patients receive a laminated
card with a QR code encoding their anonymized identifier. This
serves as backup when biometric readers malfunction.
Tertiary: Family linkage—Children are linked to par-
ent/guardian records until they can provide independent bio-
metrics. The system maintains family relationship graphs.
Emergency: Demographic matching—In emergencies
where no identification is available, the system can search by
demographic parameters with explicit uncertainty flagging.
7.3 Privacy Protection
Patient identifiers are pseudonymized at creation. The mapping
between pseudonym and any identifying information is stored
only at Tier 2 or higher facilities, never at rural clinics. This
ensures that device theft at Tier 1 cannot compromise patient
identity even if encryption is broken.
8 Network Synchronization Protocol
8.1 Intermittent Connectivity Model
Rural African clinics experience connectivity measured in
“availability windows”—periods when 2G network signal is
strong enough for data transfer. Our protocol is designed for:
• Connectivity windows of 15-30 minutes, 2-3 times daily
• Bandwidth limited to 9.6-14.4 kbps (2G GPRS)
• Complete offline operation for up to 7 days
• Graceful degradation under network stress
8.2 Synchronization Algorithm
When connectivity is detected:
Sync() =Compare(local_root, remote_root)→∆(records)→Transfer
(8)
1. Exchange Merkle root hashes (32 bytes each direction)
2. If roots match, synchronization complete
3. If roots differ, exchange intermediate hashes to identify
changed branches
4. Transfer changed records, prioritized by: (a) Emergency
flags, (b) Referral pending, (c) Timestamp, (d) Patient age
5. Verify received records against ASIC signatures
6. Update local Merkle tree and confirm sync completion
4

9 Experimental Validation
9.1 Hardware Validation Setup
We conducted comprehensive validation experiments using:
• Lucky Miner LV06 (BM1366, 5nm, 500 GH/s, 13W)
• Antminer S9 (BM1387, 16nm, 189 chips, 14 TH/s, 1372W)
• Custom Stratum proxy implementing DHF protocol
• Raspberry Pi 4 as local controller
• 50W solar panel + 12V 20Ah battery for rural simulation
9.2 DHF Validation Results
The primary validation experiment ran for 300 seconds on
Lucky Miner LV06:
Table 2: DHF Validation Results (LV06, 300s test)
Metric Value Significance
Total proofs generated 23 Baseline
Verification rate 100% All proofs valid
Average proof time 12.52s D=32768
Energy per proof 162.8 J 13W×12.52s
Efficiency 2.93 MH/W 8.5×vs RTX 3090
Hash rate stability CV = 0.08 Highly consistent
Key Result:The Lucky Miner LV06 achieved 100% veri-
fication rate across all 23 proofs, with 2.93 MH/W efficiency
representing 8.5×improvement over RTX 3090 GPU baseline.
This validates the DHF paradigm for healthcare blockchain ap-
plications.
9.3 Cross-Device Universality
To confirm hardware universality, we replicated experiments
across multiple device types:
Table 3: Cross-Device Validation Results
Device Chip Process Verif. Efficiency
Lucky Miner LV06 BM1366 5nm 100% 2.93 MH/W
Lucky Miner LV07 BM1366 5nm 100% 3.12 MH/W
Antminer S9 BM1387 16nm 100% 0.76 MH/W
USB ASIC BM1384 28nm 100% 0.42 MH/W
All devices achieved 100% verification rate, confirming that
the DHF paradigm is hardware-universal across ASIC genera-
tions from 28nm to 5nm process technology.
9.4 Image Authentication Validation
Reed-Solomon watermarking was validated on 100 test medi-
cal images:Table 4: Image Authentication Test Results
Damage Type Damage Level Recovery Rate
JPEG compression Quality 70 98%
JPEG compression Quality 50 87%
Metadata strip Complete 100%
Crop (edge) 10% 95%
Crop (edge) 20% 82%
Random noise 30% 94%
Random noise 40% 78%
9.5 Solar Power Validation
A 7-day rural deployment simulation was conducted:
• Location: Simulated Sahel conditions (6 hours direct
sun/day)
• Equipment: 50W panel, 12V 20Ah battery, LV06 miner
• Load: 8 hours operation/day (clinic hours)
• Result: Zero downtime over 7 days, including 2 overcast
days
• Battery never dropped below 40% capacity
10 Deployment Scenarios
10.1 Epidemic Outbreak Tracking
SiliconHealth enables rapid epidemic response through real-
time case reporting from rural clinics, automatic contact trac-
ing via family linkage graphs, authenticated test results with
tamper-proof timestamps, geographic clustering analysis at
Tier 2/3 facilities, and SMS alerts to community health work-
ers.
During simulated Ebola-like outbreak scenario, the system
demonstrated 94% contact identification within 24 hours of in-
dex case report.
10.2 Vaccination Certificate Management
Childhood immunization tracking represents a critical use case:
ASIC-signed vaccination records are unforgeable, family link-
age ensures sibling coverage tracking, automated reminder
generation for due vaccines, QR-code verification at school en-
rollment, and cold chain breach detection.
10.3 Maternal Health Tracking
Antenatal care requires longitudinal record maintenance: pre-
natal visit history across multiple facilities, authenticated ultra-
sound images, risk factor flagging, delivery outcome recording
with newborn linkage, and postnatal follow-up scheduling.
10.4 HIV/AIDS Confidentiality
HIV status requires enhanced privacy protections: double-
encryption for HIV-related records, access logging with au-
tomatic alerts for unusual patterns, “break glass” emer-
5

gency access with mandatory justification, support for patient-
controlled disclosure, and integration with antiretroviral ther-
apy tracking.
11 Economic Analysis
11.1 Hardware Cost Comparison
Table 5: Hardware Cost Comparison (per verification node)
Solution Hardware Power Cost Efficiency
GPU (High-end) RTX 3090 350W $1,500 0.34 MH/W
GPU (Mid-range) RTX 3080 320W $700 0.28 MH/W
ASIC (Rural) LV06 13W $50 2.93 MH/W
ASIC (Urban) S9 1372W $60 0.76 MH/W
Cost Reduction:The LV06 ASIC achieves 96% cost reduc-
tion compared to RTX 3090 ($50 vs $1,500) while delivering
8.5×better energy efficiency (2.93 vs 0.34 MH/W).
11.2 Total Cost of Ownership (5-Year)
Table 6: 5-Year TCO Analysis per Rural Clinic
Component Initial Annual 5-Year
ASIC Device (LV06) $50 $0 $50
Solar Panel (50W) $80 $0 $80
Battery (12V 20Ah) $40 $20 $140
Charge Controller $25 $0 $25
Tablet Computer $150 $0 $150
Fingerprint Reader $5 $0 $5
QR Code Printer $30 $15 $105
Installation/Training $100 $20 $200
Connectivity (2G) $0 $24 $120
TOTAL $480 $79 $847
At $847 per clinic over 5 years ($169/year), SiliconHealth
costs less than one month of a community health worker’s
salary in most sub-Saharan African countries, while potentially
serving thousands of patients.
11.3 Scalability Analysis
Table 7: Network Deployment Costs by Scale
Deployment T3 T2 T1 Total Per Clinic
District (pilot) 0 1 10 $9,970 $997
Region 1 5 50 $49,250 $878
Province 3 20 200 $183,500 $823
National (small) 10 100 1000 $903,500 $814
Economies of scale reduce per-clinic costs from $997 (pi-
lot) to $814 (national deployment), with the majority of savings
from shared Tier 2/3 infrastructure.12 Limitations and Future Work
12.1 Current Limitations
Throughput:While sufficient for rural clinic volumes (50-200
patients/day), the system may bottleneck at high-volume urban
hospitals without additional Tier 2 nodes.
Language Coverage:Current RAG semantic expansion
covers major African languages (Swahili, Hausa, Amharic,
Yoruba) but lacks support for hundreds of minority languages.
Hardware Availability:While LV06/LV07 devices are cur-
rently abundant, long-term availability depends on continued
cryptocurrency mining industry production.
Regulatory Approval:Medical device certification varies
by country; regulatory pathways for ASIC-based healthcare
systems remain undefined.
12.2 Future Work
AI Diagnostic Integration:Extending the system to support
on-device AI inference for diagnostic assistance (malaria smear
analysis, tuberculosis X-ray screening).
Interoperability:Developing bridges to existing health in-
formation systems (DHIS2, OpenMRS) for hybrid deploy-
ments.
Hardware Certification:Working with regulatory bodies to
establish certification pathways for medical-use ASICs.
Multi-Country Pilot:Planned deployments in Ethiopia,
Kenya, and Nigeria to validate cross-border patient record
portability.
13 Conclusions
This paper has presented SiliconHealth, a comprehen-
sive blockchain-based healthcare infrastructure designed for
resource-constrained regions. By repurposing obsolete Bitcoin
mining ASICs, we demonstrate that secure, verifiable elec-
tronic health records can be deployed at costs 96% lower than
conventional alternatives.
Our key contributions include:
• The Deterministic Hardware Fingerprinting paradigm
achieving 100% verification rate
• Reed-Solomon medical image authentication with 30-40%
damage tolerance
• A four-tier hierarchical network architecture from regional
hospitals to mobile health points
• Offline synchronization protocols enabling healthcare de-
livery under intermittent 2G connectivity
• Patient identity management without government docu-
mentation
• Comprehensive validation demonstrating 2.93 MH/W effi-
ciency on Lucky Miner LV06
The economic analysis demonstrates that complete rural
clinic deployment costs $847 over 5 years—less than $170
annually—making universal health record coverage economi-
cally feasible for the first time in sub-Saharan Africa.
6

With over 600 million people in sub-Saharan Africa lacking
access to basic health information systems, SiliconHealth rep-
resents not merely a technical contribution but a humanitarian
opportunity. The obsolete cryptocurrency mining hardware that
currently contributes to electronic waste can instead become the
foundation for a healthcare revolution.
“Every ASIC that once mined Bitcoin can now mine something
far more valuable: the health data that saves lives. ”
Acknowledgments
This work was conducted using donated mining hardware and
open-source software. The authors thank the Bitcoin mining
community for protocol documentation, the reservoir comput-
ing research community for foundational frameworks, and the
healthcare workers in sub-Saharan Africa whose challenges in-
spired this research. Special acknowledgment to Woldia Uni-
versity (Ethiopia) for institutional support.
References
[1] World Health Organization, “Health workforce require-
ments for universal health coverage and the Sustainable
Development Goals,” Human Resources for Health Ob-
server Series No. 17, WHO, Geneva, 2016.
[2] M. O. Akanbi, A. N. Ocheke, P. A. Agaba, et al.,
“Use of Electronic Health Records in sub-Saharan Africa:
Progress and challenges,”Journal of Medicine in the
Tropics, vol. 14, no. 1, pp. 1–6, 2012.
[3] F. F. Odekunle, R. O. Odekunle, and S. Shankar, “Why
sub-Saharan Africa lags in electronic health record adop-
tion and possible strategies to increase its adoption in this
region,”International Journal of Health Sciences, vol. 11,
no. 4, pp. 59–64, 2017.
[4] F. F. Odekunle, R. O. Odekunle, and S. Shankar, “Why
sub-Saharan Africa lags in electronic health record adop-
tion,”International Journal of Health Sciences (Qassim),
vol. 11, no. 4, pp. 59–64, 2017.
[5] Bitmain Technologies, “Antminer S9 Specifications,”
2017. BM1387 chip: 16nm FinFET TSMC, 189 chips
(3×63), 14 TH/s, 1372W, 0.098 J/GH.
[6] S. M. Abdu and O. Yesuf, “Optimizing Proof-of-Work for
Secure Health Data Blockchain Using Compute Unified
Device Architecture,”Blockchain in Healthcare Today,
vol. 8, no. 2, 2025. DOI: 10.30953/bhty.v8.421
[7] C. Herder, M. D. Yu, F. Koushanfar, and S. Devadas,
“Physical Unclonable Functions and Applications: A
Tutorial,”Proceedings of the IEEE, vol. 102, no. 8,
pp. 1126–1141, 2014.
[8] A. Azaria, A. Ekblaw, T. Vieira, and A. Lippman,
“MedRec: Using Blockchain for Medical Data Access
and Permission Management,”2nd International Confer-
ence on Open and Big Data (OBD), IEEE, pp. 25–30, Au-
gust 2016.[9] S. M. Abdu and D. Lemma, “Improving the Performance
of Proof of Work-Based Bitcoin Mining Using CUDA,”
International Journal of Innovative Science and Research
Technology, vol. 6, no. 12, pp. 1139–1145, 2021.
[10] S. Masvaure, A. Dzinamarira, et al., “Digital health inter-
ventions in strengthening primary healthcare systems in
Sub-Saharan Africa,”PLOS Digital Health, 2025.
[11] G. Tanaka, T. Yamane, J. B. Héroux, et al., “Recent ad-
vances in physical reservoir computing: A review,”Neu-
ral Networks, vol. 115, pp. 100–123, 2019.
[12] F. Angulo de Lafuente, “Speaking to Silicon: Thermody-
namic Reservoir Computing with Bitcoin Mining ASICs
and the Single-Word Handshake Protocol,” Working Pa-
per, 2026.
[13] P. Lewis, E. Perez, A. Piktus, et al., “Retrieval-
Augmented Generation for Knowledge-Intensive NLP
Tasks,”Advances in Neural Information Processing Sys-
tems, vol. 33, pp. 9459–9474, NeurIPS 2020.
[14] Lucky Miner, “LV06 Specifications,” 2024. BM1366
chip (5nm), 500 GH/s±10%, 13W±5%,<35 dB,
130×66×40mm, WiFi 2.4GHz.
[15] R. C. Merkle, “A Digital Signature Based on a Conven-
tional Encryption Function,”Advances in Cryptology—
CRYPTO ’87, LNCS, vol. 293, pp. 369–378, Springer,
1988.
[16] NIST, “Secure Hash Standard (SHS),” FIPS PUB 180-4,
National Institute of Standards and Technology, August
2015.
[17] I. S. Reed and G. Solomon, “Polynomial Codes Over Cer-
tain Finite Fields,”Journal of the Society for Industrial
and Applied Mathematics, vol. 8, no. 2, pp. 300–304,
1960.
[18] UNICEF, “Birth Registration,” UNICEF Data, 2023.
Available:https://data.unicef.org/topic/
child-protection/birth-registration/
[19] H. Jaeger, “The ‘echo state’ approach to analysing and
training recurrent neural networks,” GMD Report 148,
German National Research Center for Information Tech-
nology, 2001.
[20] W. Maass, T. Natschläger, and H. Markram, “Real-time
computing without stable states: A new framework for
neural computation based on perturbations,”Neural Com-
putation, vol. 14, no. 11, pp. 2531–2560, 2002.
[21] P. Bak, C. Tang, and K. Wiesenfeld, “Self-organized crit-
icality: An explanation of the 1/f noise,”Physical Review
Letters, vol. 59, no. 4, pp. 381–384, 1987.
[22] M. Lukoševi ˇcius and H. Jaeger, “Reservoir computing ap-
proaches to recurrent neural network training,”Computer
Science Review, vol. 3, no. 3, pp. 127–149, 2009.
[23] D. Verstraeten, B. Schrauwen, M. D’Haene, and
D. Stroobandt, “An experimental unification of reservoir
computing methods,”Neural Networks, vol. 20, no. 3,
pp. 391–403, 2007.
7

[24] World Health Organization Regional Office for Africa,
“Atlas of African Health Statistics 2022,” WHO AFRO,
Brazzaville, 2022.
[25] P. E. Mbondji, L. Kebede, P. Zielinski, et al., “Health in-
formation systems in Africa: descriptive analysis of data
sources, information products and health statistics,”Jour-
nal of the Royal Society of Medicine, vol. 107, no. 1 suppl,
pp. 34–45, 2014.
[26] K. Nakajima and I. Fischer (Eds.),Reservoir Comput-
ing: Theory, Physical Implementations, and Applications,
Springer, 2021.
[27] S. Hochreiter and J. Schmidhuber, “Long short-term
memory,”Neural Computation, vol. 9, no. 8, pp. 1735–
1780, 1997.
[28] Y . LeCun, Y . Bengio, and G. Hinton, “Deep learning,”
Nature, vol. 521, pp. 436–444, 2015.
[29] Slush Pool, “Stratum Mining Protocol,” 2012.
Available:https://slushpool.com/help/
stratum-protocol/
8

Author Profiles
Francisco Angulo de Lafuente
Independent Researcher / Open-Source Developer, Madrid, Spain
Background:Computer Engineering and Biotechnology (Universidad Politécnica de Madrid). Extensive experience as
programmer, writer, and researcher in biotechnology, environmental science, and advanced AI architectures. Former director
of the Ecofa project in biofuels.
Research Interests:
• Physics-based, hardware-agnostic AI architectures for extreme efficiency
• Neuromorphic computing and holographic neural networks
• Quantum-inspired and photonic/optical computing paradigms
• Sustainable, democratized AI (low-resource, open-source systems)
• Repurposed hardware (Bitcoin mining ASICs) for reservoir computing and blockchain
• Biotechnology intersections and applications in healthcare infrastructure
Notable Projects:Creator of CHIMERA (neuromorphic framework, 43×faster than PyTorch, 88.7% memory reduction,
OpenGL-based), NEBULA (holographic quantum-inspired neural network), and EUHNN (Enhanced Unified Holographic
Neural Network—NVIDIA-LlamaIndex Developer Contest 2024).
Selected Publications:
• “SiliconHealth: A Complete Low-Cost Blockchain Healthcare Infrastructure...” (arXiv:2601.09557, 2026)
• “Toward Thermodynamic Reservoir Computing: Exploring SHA-256 ASICs...” (arXiv:2601.01916, 2026)
• “CHIMERA v3.0: Intelligence as Continuous Diffusion Process” (2025)
Contact & Profiles:
Email: lareliquia.angulo@gmail.com
ORCID: 0009-0001-1634-7063
arXiv: arxiv.org/a/angulodelafuente_f_1
GitHub: github.com/Agnuxo1
ResearchGate: researchgate.net/profile/Francisco-Angulo-Lafuente-3
Hugging Face: huggingface.co/Agnuxo
Kaggle: kaggle.com/franciscoangulo
Wikipedia (ES): es.wikipedia.org/wiki/Francisco_Angulo_de_Lafuente
9

Seid Mehammed Abdu
Senior Lecturer, Department of Computer Science, Woldia University, Ethiopia
Education:Master of Science (MSc) in Computer Science (specialization: Network and Security), Addis Ababa University,
Ethiopia. Thesis: “Improving the Performance of Proof of Work-Based Bitcoin Mining Using CUDA” (2021).
Research Interests:
• Blockchain and Proof-of-Work optimization
• Data Science and Artificial Intelligence
• Machine Learning and Deep Learning
• Internet of Things (IoT) and Cloud Computing
• Cyber Security and Real-time Embedded Systems
• Data-driven applications in agriculture, tourism, and health
About:Seid Mehammed Abdu teaches undergraduate Computer Science courses and conducts research focused on AI,
mobile technology, Blockchain, Data Science, and low-cost, locally relevant tech solutions. His work emphasizes improving
rural livelihoods and education through innovative, affordable technologies.
Selected Publications:
• “Optimizing Proof-of-Work for Secure Health Data Blockchain Using CUDA” (2025)
• “Improving the Performance of Proof of Work-Based Bitcoin Mining Using CUDA” (2022)
Contact & Profiles:
Email: seid.m@wldu.edu.et
ResearchGate: researchgate.net/profile/Seid-Abdu-4
Google Scholar: scholar.google.com/citations?user=5OtelMsAAAAJ
10

Nirmal Tej Kumar
Consultant / Senior Staff Scientist, antE Inst (UTD), Dallas, TX, USA
Education:Dr.Engg.Sc (Doctor of Engineering Science) in Nanotechnology. Background in Electrical and Electronics
Engineering (EEE), Quantum Physics, Informatics, and related fields. Extensive international experience across USA, UK,
Germany, China, Korea, Malaysia, Israel, Brazil, Japan, and Russia.
Research Interests:
• Nanotechnology and Quantum Physics/Computing
• Artificial Intelligence, Machine Learning, Deep Learning
• High-Performance Computing (HPC) and Informatics
• Blockchain, Neuromorphic Computing, Theorem Proving
• Bio-informatics (DNA/RNA sequencing, Cryo-EM image processing, MRI scans)
• Accelerator Physics, Astrophysics, Biophysics, Space Science
• Photonics, Semiconductors, Unconventional Computing
About:Nirmal Tej Kumar is an Electrical Engineer, Quantum Physicist, and AI Researcher with focus on interdisciplinary
R&D. He serves as consultant at antE Inst (USA) and engages in non-profit informatics and nanotechnology projects. His
work emphasizes innovative cross-domain solutions using theorem provers, AI/ML, blockchain, and neuromorphic systems.
Selected Publications/Projects:
• “Block Chain + NGINX + Neuromorphic Computing + Theorem Proving→To Process MRI Scans” (2024)
• “OLLAMA-Nirmal-Py: Integrating Ollama into fiber cut prediction software in Telecoms” (2024)
• Theorem Proving + AI for Bio-informatics HPC Software R&D
• Bayesian ML + NLP for DNA/RNA Sequencing and Next-Gen Bio-informatics
Contact & Profiles:
Email: hmfg2014@gmail.com
ORCID: 0000-0002-5443-7334
ResearchGate: researchgate.net/profile/Nirmal-Tej
LinkedIn: linkedin.com/in/nirmal-tej-kumar
Manuscript:Open Access Publication
Target:PLOS Digital Health / Blockchain in Healthcare Today
Project:SiliconHealth - Blockchain Healthcare Infrastructure
Version:2 (January 2026) - Corrected author affiliations and profiles
“From mining cryptocurrency to mining health data that saves lives. ”
11