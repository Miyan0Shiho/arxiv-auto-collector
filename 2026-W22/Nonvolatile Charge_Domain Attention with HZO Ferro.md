# Nonvolatile Charge-Domain Attention with HZO Ferroelectric Capacitors: A Simulation-Based Device-to-System Evaluation

**Authors**: Faris Abouagour

**Published**: 2026-05-27 09:28:40

**PDF URL**: [https://arxiv.org/pdf/2605.28208v1](https://arxiv.org/pdf/2605.28208v1)

## Abstract
Transformer decoding is constrained by both attention compute and KV-cache movement. This paper presents the Ferroelectric Charge-Domain Compute Cell (FCDC), a hafnium-zirconium-oxide (HZO) memcapacitor with an access device that stores analog state nonvolatilely and performs charge-domain VMM for attention. Two deployment modes are evaluated throughout: a full-substrate mode that runs q, k, v, o projections and both attention matmuls on FCDC, and a KV-coprocessor mode that only stores KV and executes the two attention matmuls; the projection-noise budget upper-bounds the coprocessor mode.
  The device-to-system model is cross-checked across ngspice, CrossSim, FiPy, and NeuroSim and anchored in recent wafer-scale 10 nm HZO measurements. Across 12 pretrained LLMs (1.1-32 B dense, plus a partial-layer Mixtral-8x22B 141 B-MoE stress test at k=75% and a 128 k-context dense-Mistral replication), all-layer noise substitution adds only +2.62% WikiText-2 perplexity on Qwen3-32B and +2.90% +/- 0.33% on Mistral-7B-v0.3 (five-seed mean). End-to-end analog attention adds at most +1.68 pp on TinyLlama-1.1B and shrinks below +/-1 pp on every >=7 B model. Downstream accuracy on HellaSwag, ARC, LAMBADA, and GSM8K stays within 5% of the digital baseline for Mistral-7B (MMLU -1.6 pp).
  The headline energy win is nonvolatility, no refresh, and KV-cache residency. A workload-level simulator anchored on measured INT4 decode energy delivers 18-35x lower per-served-token energy on RAG and agent loops against a single-user INT4 GPU baseline; against optimized GPU serving (batched vLLM, CPU+NVMe park, power-gate) the robust advantage shrinks to 1.36-4.69x and remains >=41x on parked sessions with multi-hour residency.

## Full Text


<!-- PDF content starts -->

Nonvolatile Charge-Domain Attention with HZO
Ferroelectric Capacitors: A Simulation-Based Device-to-System
Evaluation
Faris Abouagour
Faculty of Engineering, Mansoura University, Egypt
faresaboagour@std.mans.edu.eg
May 28, 2026
Preprint. Under review. Public code (with seeds, noise tapes, energy-measurement scripts, and workload
simulator) accompanies this preprint at submission time, not on acceptance; see §8.
Abstract
Transformer decoding is constrained by both attention compute and repeated movement
of the key-value (KV) cache. This paper presents theFerroelectric Charge-Domain Compute
Cell(FCDC), a hafnium-zirconium-oxide (HZO) memcapacitor with an access device that stores
analog state nonvolatilely and performs charge-domain vector–matrix multiplication for attention.
Two deployment modes are evaluated consistently throughout: afull-substratemode that runs
q,k,v,oprojections and both attention matmuls on FCDC (the harder LLM noise test) and
aKV-coprocessormode that only stores KV and executes the two attention matmuls (the
serving-energy model, Fig. 4); the projection-noise budget upper-bounds the coprocessor mode.
The device-to-system model is cross-checked acrossngspice,CrossSim,FiPy, andNeu-
roSimand anchored in recent wafer-scale 10nm-HZO capacitor measurements [ 1,10]. Across
12 pretrained LLMs (1.1–32B dense, plus a separate partial-layer Mixtral-8 ×22B 141B-MoE
stress test at k=75%and a separate 128k-context dense-Mistral replication), all-layer noise
substitution adds only+2 .62%WikiText-2 perplexity on Qwen3-32B and+2 .90%±0.33%on
Mistral-7B-v0.3 (five-seed mean). End-to-end analog attention adds at most+1 .68pp on top of
the projection-only proxy on TinyLlama-1.1B and shrinks below ±1pp on every≥7B model.
Downstream accuracy on HellaSwag, ARC, LAMBADA, and GSM8K stays within5%of the
digital baseline for Mistral-7B (MMLU−1.6pp).
The headline energy win is not raw active-MAC but nonvolatility, no refresh, and KV-cache
residency. A workload-level simulator anchored on measured INT4 decode energy delivers18–35 ×
lower per-served-token energy on retrieval-augmented-generation and agent loops with persistent
KVagainst a single-user INT4 GPU baseline; against optimized GPU serving (batched vLLM,
CPU+NVMe park, power-gate) the robust advantage shrinks to1 .36–4.69×on those workloads
and remains≥41×on parked sessions with multi-hour residency. All four GPU regimes are
reported in Table 7. At the active-MAC layer the FCDC projection ties switched-capacitor
SRAM CIM and a well-tuned batched-vLLM chat baseline, so the substantive contribution sits
in the long-residency regime. Taken together, the results identify long-residency, persistent-
KV workloads as a regime in which a nonvolatile charge-domain attention substrate retains
an order-of-magnitude advantage over the strongest available GPU baseline, and quantify the
LLM-accuracy budget at which that advantage holds.arXiv:2605.28208v1  [cs.AR]  27 May 2026

1 Introduction
Autoregressive transformer inference repeatedly evaluates attention over a growing KV cache. The
cost is partly arithmetic, through Q·K⊤andsoftmax (·)·V, and partly data movement, because
stored keys and values must remain accessible throughout decoding. This makes attention a
natural target for in-memory compute, but it also imposes a stronger requirement than conventional
weight-stationary accelerators: the stored state is dynamic, long-lived, and repeatedly read.
Existing device approaches address different parts of this problem. Gain-cell analog attention
performs attention multiplications inside a charge-storage array, but the storage is volatile and
requires refresh on millisecond timescales [ 9]. Ferroelectric KV-cache arrays demonstrate nonvolatile
storage with fast switching and high endurance in specific 3D stacks, but the ferroelectric array is
primarily a storage substrate rather than the attention compute engine [ 18]. Ferroelectric field-effect
transistor (FeFET) content-addressable-memory and compute-in-memory work further shows that
ferroelectric devices can support KV-cache pruning and in-memory search operations. These results
motivate a narrower question: can a capacitive HZO ferroelectric cell provide a useful design point
for nonvolatile KV-cache residency and local charge-domain attention compute?
This paper presents FCDC, a modeled HZO memcapacitor tile for charge-domain vector–matrix
multiplication. The tile stores analog state in a ferroelectric capacitor, reads it through an access
device, and accumulates column charge for attention projections and attention matrix products. A
negative-capacitance read path is analysed as an optional way to improve signal margin.
Thecontributionisadevice-to-systemevaluationbuiltoncross-simulatorvalidationandonrecent
wafer-scale HZO capacitor data. At the circuit and array level, the cell noise and peripheral energy
budgets are derived and the same analytic model is cross-checked acrossngspice,CrossSim,FiPy,
andNeuroSim. At the model level, the resulting noise impact on pretrained LLMs is measured at
scale, including projection-only substitution, end-to-end analog attention, seed sensitivity, longer-
context replication, and a small low-rank-adapter quantization-aware training (LoRA QAT) recovery
path. At the system level, the corrected active-MAC energy is compared against an analytic A40
baseline and against measured analog in-memory-compute (IMC) silicon, and a workload-level
simulator quantifies per-served-token energy across five serving regimes. The scope is a capacitor-
based charge-domain attention substrate evaluated in simulation, with explicit separation between
simulation-backed results and the fabrication-dependent assumptions on which they rest.
2 Related Work
Table 1: Position relative to the closest prior work on nonvolatile / charge-domain attention
substrates. “In-place” = matrix multiplication executed inside the storage array. “Dense” = full
attention without top-kpruning.
Work Device Nonvolatile In-place LLM evaluated Context
Leroux [9] gain cell no (ms refresh) yes (dense) GPT-2 (124M)∼1k
Xu [18] 3D 1T-nC-1T FE yes pruning+hybrid device paper –
UniCAIM [19] FeFET CAM/CIM yes top-kGPT-2 class 2k
Yin [21] 1FeFET-1C yes yes (macro) macro-level –
FCDC (ours) HZO 1T-1C yes yes (dense) Qwen3-32B≤128 k∗
∗Largest dense model fully substituted: Qwen3-32B at 8k tokens. Longest context replicated: Mistral-7B-v0.3 at 128k tokens.
Mixtral-8x22B (141B MoE) reported atk=75%partial-layer only.
2

Analog in-memory attention.Leroux et al. [ 9] performQ·K⊤andsoftmax (·)·Vinside a
gain-cell array and report GPT-2-comparable text quality at the 124M scale on OpenWebText /
WikiText-2 / LAMBADA. Their array is volatile and must be refreshed on millisecond timescales
(the silicon CMOS gain-cell retention constant reported in that work is τ≈5ms). The main
distinction of FCDC is therefore nonvolatile KV-cache residency rather than the idea of analog
attention itself.
Ferroelectric and nonvolatile KV-cache accelerators.Xu et al. [ 18] demonstrate a 3-D
vertical 1T-nC-1T ferroelectric KV-cache array with fast switching, long retention, high endurance,
hybrid analog-digital CIM, and token-wise pruning. UniCAIM [ 19] is even closer at the architecture
level: it uses FeFET-based CAM/CIM for long-context LLM KV-cache pruning, charge-domain
score accumulation, and current-domain exact attention on selected tokens. These works preclude
any broad claim that ferroelectrics or CIM have not been applied to KV-cache acceleration. The
narrower contribution here is a capacitor-based charge-domain attention design point, evaluated as
a nonvolatile resident KV substrate under explicit peripheral and device-physics bounds.
Ferroelectric charge-domain CIM.FeFET-based nonvolatile charge-domain CIM was intro-
duced before this work [ 20], and the 1FeFET-1C neuro-symbolic AI prototype [ 21] demonstrates
fabricated ferroelectric charge-domain MAC and associative-search functionality. Recent nonvolatile-
capacitor work further studies ferroelectric capacitive crossbars and calibrated SPICE models for
charge-domain MACs [ 15]. These papers are the appropriate hardware baselines for the cell and
macro, while HERMES [ 8] and switched-capacitor SRAM CIM [ 17,16] are the appropriate measured
active-MAC energy baselines.
Scope of novelty.The specific contribution is the first LLM-tolerance evaluation of a nonvolatile
HZO charge-domain attention substrate at 141B-parameter scale and 128k context, with cross-
simulator validation and corrected peripheral energy accounting. The paper does not assert
priority over prior ferroelectric CIM, charge-domain CIM, or ferroelectric KV-cache work; those are
appropriate hardware baselines, not competing claims. NC voltage gain is treated as an exploratory
read-path option.
3 The FCDC Cell
3.1 Device
Each FCDC cell combines a nonvolatile HZO storage capacitor, an optional read-path gain element,
and an access transistor.
Storage capacitor.The storage element is a hafnium-zirconia ( Hf0.5Zr0.5O2) memcapacitor
storing one analog weight as remanent polarization. Capacitance C0≈69aF at the demonstrated
cell footprint; HZO carries the non-volatile state with retention reported up to ≥10years in
specific 3D 1T-nC-1T stacks [ 18] and dielectric-class ϵr≈20–35measured on free-standing HZO
membranes and related HZO stacks [ 12,5]. This work does not claim these endurance/retention
numbers transfer directly to the planar 50nm cell modeled here; they bound what the material class
has demonstrated. Recent small-signal studies further decompose the apparent capacitance into
dielectric + polarization + domain-wall contributions [ 5], which means the smooth C(P)surrogate
used here is an ML-grade differentiable approximation, not a device-physics model.
3

(a) 1T-1C cell cross-section.
...
...BL0 BL1 BL2 BL3
PL0
WL0
PL1
WL1
PL2
WL2
ADC 8-bit...
... (b)256×256tile (shown4×3).
Figure 1: FCDC cell and tile. (a) Storage node (TiN bottom electrode) tied to the NMOS drain;
PL drives the top electrode; optional 10nm NC-HZO layer (dashed) provides read-path gain; cell
pitch 50nm. (b) WL selects a row, PL drives the capacitor top plates, and each column shares one
BL terminated by a 4-bit ADC; accent column shows the selected analog read path.
Measured-device anchor.The material parameters used in this paper are within the envelope
of recent measured 10nm HZO capacitors. A wafer-scale study of 270 ALD-grown 10nm-HZO /
30nm-TiN MIM capacitors across six dies reports a mean remanent polarization Pr≈40.58µC/cm2
extracted from±6V triangular dynamic-hysteresis P–V loops at 1kHz, with die-to-die spread
captured by an unsupervised PCA / K-means clustering model [ 1]. A separate MFIM HZO
study at the same 10nm thickness, measured by PUND pulsing at ±3V / 100kHz, demonstrates
>109programming/erase cycles at room temperature with recoverable fatigue [ 10]. These two
measured anchors bracket the Pr, switching-voltage, and endurance assumptions used in the present
ngspice/Landau models and in the §3.4 energy accounting; the wafer-scale variability data also
bounds the device-to-device σswept in the noise study (§3.3). It is re-emphasised that HZO is not
measured here: these are literature anchors that constrain (but do not constitute) the planar 50nm
cell modeled here. It is also noted that [ 1] reports a>97%functional yield on the characterised high-
yield dies, which sets the upper bound of the variability sweep used; the lower bound ( σC/C∼20%)
covers worst-die behaviour. One specific gap is that both anchors use 10nm HZO while the cell
modeled here has an 8nm storage layer; the thinner stack is within the material class but is a small
extrapolation that would need direct measurement to fully close.
Thickness sensitivity.The 8nm vs 10nm sensitivity is quantified with the analytic cell model.
HoldingPr,ϵr=25, pitch= 50nm, Vrd=0.158V,Ec=1MV/cm and the Merz/NLS coefficients of [ 6]
fixed, sweeping dHZOfrom 6 to 12nm gives C0and intrinsic read energy scaling as1 /d(+25%at
8nm relative to 10nm) and vkTCscaling as√
d(−11%at 8nm). The read field Erd=Vrd/dstays
sub-coercive across the full range ( Erd/Ec≤0.26at 6nm,≤0.16at 10nm), and the per-pulse
Merz/NLS switching probability remains numerically zero across the entire 6–12nm sweep because
the activation barrier Ea∈[9,20]MV/cm dominates the exponent at any Erd<0.5Ec. The headline
noise budget, energy ratios, and read-disturb lifetime all move by ≤25%when the storage layer is
adjusted from 8nm to 10nm; none of the qualitative claims of §5–§7.4 change.
Optional read-path gain.The exploratory gain option is a series NC element: a ferroelectric
capacitor biased in its negative-capacitance regime [ 14], intended to provide ∼2.5×peak small-signal
4

6 8 10 12
dHZO (nm)5060708090C0 (aF)
6 8 10 12
dHZO (nm)0.60.70.80.91.01.1Eread (aJ)
6 8 10 12
dHZO (nm)0.00.20.40.60.81.0Erd/Ec
Figure 2:C0,Eread, andErd/Ecversus HZO thickness. Dashed: 8nm operating point; dotted:
10nm measured anchor.
voltage gain on the read path. A 1-D Landau derivation with calibrated 10nm-HZO coefficients
(§3.3) shows that |Av|=2.5×requires the series load satisfy Cs/|CFE|≈0.714, which is a tight
matching condition: a ±20%process shift moves the gain to1 .47×or8.3×, and a30%shift crosses
the stability boundary. This is treated NC gain accordingly as a read-path option rather than as
a fixed design constant; the noise model is reported both ways. Throughout the paper the ratio
Cs/|CFE|uses the same convention, with |Av|peaking near Cs/|CFE|=0.714(peak-gain operating
point) and decreasing monotonically toward unity as the ratio increases; Appendix A reports a
conservative stable-side operating point at Cs/|CFE|=3.5with|Av|≈1.4on the same Av(Cs/|CFE|)
curve.
Access device.A read transistor gates the memcapacitor onto the column line during a read
pulse. The baseline calculations assume a sub-coercive storage-layer field and separate the intrinsic
cell switching energy from the tile-level peripheral energy.
Per-cell intrinsic read energy is Eintrinsic
read =1
2C0V2
rd≈8.6×10−19J at the operating point
used throughout this paper ( Vrd≈0.158V, sub-coercive; field across the 8nm storage HZO is
≈0.20MV/cm≈0.20Ec). A supply-driven CV2accounting (no charge recovery) gives twice
that; for context, the per-MAC tile energy is dominated by DAC/ADC peripherals not by the cell
(§3.4), so the cell-level number is the lower bound rather than the system-level energy claim. At
matched accuracy, the modeled FCDC V-DAC tile energy is1 .92×10−14J/MAC, compared with
1.19×10−13J/MAC for the calibrated memristive reference. The tile-design table below bounds the
read-voltage range in which this energy scales asV2
rd.
3.2 Tile
Cells are aggregated into an attentiontile(Fig. 1b) that performs charge-domain VMM in the
Vrd·Qcelldomain. Reading Nrows in parallel produces a column current proportional to/summationtext
iVrd,i·Qi;
tile noise is reported both with and without the optional NC read-path gain. Each tile owns a
fragment of the KV cache and performs both the projection and the attention step locally. Inter-tile
traffic is therefore digital and small: only post-softmax row sums and across-head accumulations.
3.3 Noise model and design space
Cell-level noise is characterised as an additive Gaussian with standard deviation proportional to the
full-scale charge, σ=nf·Q FS. The noise fraction nfis a function of the column readout geometry
5

rather than a free parameter. Three sources contribute: thermal noise, flicker noise, and capacitance
mismatch. They are bounded analytically and checked against a behavioralngspicetransient
simulation.
Thermal noise.The kT/C contribution at the sense node is vkTC=/radicalbig
kBT/C 0/√Nrowsafter
averaging across the column ( vkTC≈484µV atC0=70aF,Nrows=256). The1 /√Nrowsreduction
applies only to independent per-row sampled noise. Integration-cap reset, ADC comparator offset,
supply noise, row-driver gradients, column-to-column coupling, shared-reference noise, timing jitter,
NC gain variation, and FE imprint/wake-up/fatigue drift do not average and are bounded separately
by the conservative-tile geometry.
Flicker noise.The 1/f contribution comes from the sense amplifier and is integrated from 1Hz to
the gain-bandwidth product (GBW; ≈46µV for a 10µV /√
Hz@ 1Hz CMOS amp at 1GHz GBW).
Capacitance mismatch.A 5%–20% per-cellσ C/Cis assumed, depending on whether cells are
post-calibration. The 5% number is a best-case post-calibration or differential-coding target, not
a raw-device measurement: at 50nm pitch with ALD HZO grain size 10–30nm and switching
regions of 10–15nm, only O(3−25)active grains/domains lie inside a cell, so raw cap variation is
granular and non-Gaussian. The effective column-level contribution averages over Nrowsonly to the
extent that cell errors are independent after calibration. This 5%–20% sweep range is consistent
with the recent wafer-scale measurement of 270 ALD 10nm-HZO MIM capacitors, which reports
<10%device-to-device variation in PrandVcon high-yield dies (97% functional yield) and a
5–10% prediction MAPE on Pracross an unseen die [ 1]; the lower bound of the sweep matches the
well-behaved-die measurement and the upper bound covers the worst-performing dies in that study.
Read disturb.A 0.158V pulse across the 8nm storage HZO gives Erd≈0.20MV/cm, i.e.∼0.2Ec.
Using a Merz/NLS form τ(E) =τ∞exp(Ea/Eeff)with HZO-typical τ∞=10−10s and activation field
Ea∈[9,20]MV/cm [ 6], the per-cell flip probability for a 5ns read with Ndom=10vulnerable domains
is bounded by
pcell≲N dom(τrd/τ∞) exp(−E a/Eeff)∈[10−41,10−17],
in the no-field-gain case. The required activation bound for ptarget=3×10−13/read over109reads/day
for 10 years is Ea>35Eeff, i.e.Ea>6.9MV/cm at Eeff=0.20MV/cm, inside the literature range.
This bound does not apply if NC amplification or local field hot spots raise the storage-layer field
above∼0.3MV/cm; an NC-amplified read path therefore requires a separate field-decoupled stack.
Read-pulse assumptions.The baseline read pulse used throughout is τrd=5ns. A 1GHz sense-
amp GBW is an architectural target consistent with the throughput of HERMES-class IMC silicon
[8], not a measured property of an NC-stabilized HZO stack. Numbers in this paper that depend on
read rate, including sense-amp 1/f integration and ADC throughput, should be read as 5ns / 1GHz
nominal with a wider margin for any NC-assisted variant.
Differential read.For signed analog weights and to cancel common-mode drift / row-driver
supply noise / shared reference offsets, differential cell coding ( Q+,Q−with column differencing)
is the recommended deployment configuration. It doubles cell area but is the cleanest way to
suppress the correlated noise terms above that do not average over Nrows. The primary LLM sweep
is computed in single-ended mode (conservative); a differential implementation would only improve
effectivenf.
6

NC noise propagation (conservative input-referred model).Let the read path be vout=
Avvsig+Avnupstream +ndownstream +nNC. Only noisedownstreamof the NC gain divides by Av
when input-referred; the storage-cell kT/C, read-FET gate noise, and polarization/domain jitter
all sit upstream and arenotreduced by NC. The conservative input-referred variance is therefore
σ2
eq=σ2
cell+σ2
readFET +σ2
sense/A2
v+ (σAv/Av)2x2+σ2
NC,jitter. The nominal nf=0.015tile parameter
used throughout the LLM sweep is calibrated to this conservative form (sense-amp downstream-only
Avreduction, all other terms NC-independent), so the reported PPL results remain valid even if
NC gain is removed entirely when the tile is enlarged; the conservative-tile geometry ( nf=0.009)
achieves the same input-referred budget via largerC intat the cost of area.
These boundnfbetween two physically realizable tile geometries:
TileN rowsVreadCinteg nf
Aggressive 256 100mV 100fF0.035
Nominal 256 100mV 400fF 0.015
Conservative 1024 200mV 400fF0.009
Thenominal nf= 0.015isusedinthemainLLMexperiments(§5); theaggressiveandconservative
bounds on four representative LLMs (Qwen3-32B, Mistral-7B-v0.3, Mistral-Small-24B-Base, gemma-
4-31B) in §5.3 to characterize how the architectural rankings hold across the noise design space.
The analytic noise budget and the Monte-Carlo ngspice run agree to within 4% (3 .68%vs3.56%at
the aggressive operating point).
3.4 Hardware accounting
The central accounting question is “what does one tile actually cost?” The cell-, tile-, and peripheral-
level numbers used throughout the paper into a single table, with each entry tied to either a
closed-form derivation, a SPICE run, or a literature anchor for the supporting device class.
The DAC+ADC peripherals dominate intrinsic array energy by ∼3.2×104; the nominal tile
is therefore a DAC/ADC machine with a tiny nonvolatile capacitor array attached. This is also
why the PWM-DAC variant with lower row-driver cost reaches7 .8×10−16J/MAC, but V-DAC is
conservatively kept in the main comparison. Writes are bounded as follows: per generated token, an
autoregressivedecoderappendsonenewKandonenewVvectorperlayer(2 ×nlayer×nkv_heads×dhead
cells); for Mistral-7B-class GQA this is2 ×32×8×128≈6.5×104cells/token. Using the optimistic
5×10−14J/cell write energy from the gain-cell baseline [ 9] as a conservative upper bound gives
∼3.3nJ/decoded-token of KV-append write energy, which is ∼2%of the∼1.6×10−7J/token
active-MAC attention cost at T=1024and falls to ≪1%atT≥8k. The cache-energy sensitivity
sweep in §7.3 carries Efcdc
write∈[1,100]fJ explicitly; the serving-energy ratios are insensitive to this
term except at very short residency.
Configuration naming.Three FCDC operating points appear in this paper and are used for
distinct purposes:
Name Used for ADC / DAC / attention path
FCDC-4b tile active-MAC energy (Table 2, §7) 4-bit V-DAC, 4-bit SAR ADC
FCDC-8b attention LLM end-to-end accuracy, Fig. 4 substrate
boundary8-bit SAR ADC on attention
matmuls
FCDC-10b sensitivity C6 ADC-headroom check only (§4.2) 10-bit ADC on matmuls
7

Table 2: Cell-, tile-, and peripheral-level parameters for one FCDC tile.
Quantity Value Source
Cell
Footprint50×50nm2model parameter
HZO thickness8nm model parameter
C0(linear,P=±1)≈69aF analytic, [12]
Pr 25µC/cm2[12]
Retention≥10years literature, 3D 1T-nC-1T stack [18]
Endurance1016cycles literature, same stack [ 18]; not trans-
ferable default
Write voltage1.2V supra-coercive model parameter
Read voltageV rd 0.158V sub-coercive (≈0.2E c), §3
Read energy (intrinsic)8.6×10−19J1
2C0V2
rd
Read disturb (Merz/NLS
bound)≲10−17/readE a=9MV/cm, no NC field gain, §3.3
Tile (nominal)
RowsN rows 256 §3.3
Integration capC int 400fF §3.3
ADC bits / type 4-bit successive-approximation-
register (SAR)§3.2
ADCs per tile 1 per 2 columns (128, shared) area-amortized per SC-SRAM 2024
[16]; 1-per-column is throughput-
optimal but not area-optimal
DAC bits / type 4-bit / V-DAC §3.2
Sense amplifier 1/f10µV/√
Hz@ 1Hz typical CMOS, §3.3
GBW (sense amp)1GHz architectural target; not NC-
validated, §3.3
Read pulseτ rd 5ns baseline; 1GHz throughput is target
Total cell-levelnf 0.015nominal behavioral SPICE-MC check
Tile VMM energy (16,384 active MACs/read; Nrows=256rows×dhead=64active cols/head) and
per-MAC equivalent
Array (charge integration) 9.92×10−15J/tile =
0.000605fJ/MACanalytic tile model
DAC (V-DAC, 4-bit) 3.07×10−10J/tile =18.75fJ/MAC analytic tile model
ADC (4-bit SAR) 7.68×10−12J/tile =0.469fJ/MAC analytic tile model
Total3.15×10−10J/tile =
19.22fJ/MACanalytic tile model
Process / temperature corners
Capacitance mismatch
σC/C5% (calibrated) / 20% (raw) per cell,→noise budget; §3.3
Temperature 300K assumed kT/C bound in §3.3
Cross-arch noise sensitivity 0.9%–3.5% nf→usable design-space sweep
Unless otherwise stated, energy ratios in this paper use the FCDC-4b tile and LLM end-to-end
quality uses FCDC-8b attention; the two are reported as separate operating points and are not
claimed to be the same one. A purely-4b end-to-end accuracy sweep is left as future work; the 8b
accuracy figures should therefore be read as a conservative upper bound on quality at the 4b energy
point.
8

4 Cross-Validation Methodology
To reduce dependence on any single analog-IMC simulator, the per-MAC energy of one FCDC tile
is cross-checked through four independent software implementations of the same analytic energy
model. The limitation of this check is stated upfront: agreement across implementations of the
sameanalytic model shows that the energy-accounting code is not a software artifact; it is not
physical validation of the device, the read switch, the sense amplifier, the ADC comparator, line
coupling, or the FE compact model. Tool agreement and the LLM-side correctness checks in §4.1 as
complementary debugging signals, not as substitutes for silicon.
Implementation Layer represented Disagreement vs analytic reference
ngspice[13] behavioral transient (charge / current)2×10−6
CrossSim[7] Array-level VMM2×10−7
FiPy[3] 1D PDE for polarization switching4×10−16
NeuroSim[2] ADC + peripheral energy matched
The agreement across four implementations is two orders of magnitude tighter than the device-to-
devicevariability( ∼5%)expectedonarealfab. Thisistreatedasevidencethattheenergy-accounting
code is not a software artifact; it doesnotvalidate the transistor-level read switch, sense amplifier,
ADC comparator, line coupling, or FE compact model.
4.1 Internal consistency checks
Tool agreement only proves that the four implementations agree; it does not prove they match
physics. Three independent correctness checks at the LLM-evaluation layer, all on TinyLlama-1.1B
/ WikiText-2 (4k tokens) with the same noise infrastructure as Table 4.
Check Configuration Expected Obs.∆PPL
P1: wrapper identity 22 layers,nf=0, 16-bit DAC/ADC≈0 +0.019%
P2: location control (MLP) noise→MLPgate_proj, 22 layers same order as attn+2.40%
P2: attention reference noise→q,k,v,o, 22 layers (control)+6.52%
P5: NC-gain ablation remove2.5×NC gain (nf=0.0375) crosses noise threshold+240.5%
Table 3: Sanity checks on the noise-injection wrapper for the LLM evaluations. P1 confirms
numerical transparency at zero noise; P2 contrasts the same noise injected at MLP vs. attention
sites; P5 ablates the assumed NC voltage gain.
P1verifies that the wrapper is numerically transparent at zero injected noise: when nf=0,
the perturbed model reproduces the reference PPL to2 ×10−4.P2shows that injecting the same
Gaussiannoiseintoanon-attentionlinearlayer(MLPgateprojection)degradesPPLbyacomparable
butsmalleramount (+2 .4%vs+6.5%for attention) – so the noise propagates and attention is
indeed the sensitive site, not an artifact of patch location.P5ablates the putative2 .5×NC voltage
gain by scaling nfup by2.5×; the model crosses the phase-transition threshold documented in §5.3
(+240%PPL). The intended reading is thatsomeread-path signal-gain margin is required for the
nominal noise budget to land on the stable side of the transition; whether that margin is delivered
by NC stacking, an enlarged Cint(the conservative tile, nf=0.009, achieves the same effect at the
cost of area), or higher Vrdis an open design choice. All three checks behave consistently with the
intended perturbation model, supporting the use of the wrapper for the LLM-level tolerance study.
9

4.2 End-to-end analog attention validation
The main LLM sweep (§5) injects FCDC noise only on the q,k,v,oprojections. Whether this proxy
matches an end-to-end analog attention path that also simulates Q·K⊤andA·Von FCDC tiles is
therefore tested. The eager LlamaAttention operator of TinyLlama-1.1B (all 22 layers) and route
both attention matrix multiplications through the same SPICE-calibrated FCDC noise/ADC model
used for the projections (4k WikiText-2 tokens, 2 passes, nominal nf=0.015on both projections
and attention matrix multiplications, 8-bit SAR ADC).
Configuration PPL∆vs ref
C0 reference float 8.174 n/a
C1 projection-only (main sweep) 8.700+6.43%
+ analogQ·K⊤(C2) 8.888+8.73%
+ analogA·V(C3) 8.751+7.05%
C4 full end-to-end (proj +Q·K⊤+A·V) 8.837+8.11%
C5 matmul-only (no projection noise) 8.258+1.03%
C6 C4 with 10-bit ADC on matmuls 8.847+8.23%
The end-to-end correction is C4 −C1= +1.58% PPLrelative to C1(equivalently+1 .68pp
absolute on top of the+6 .43%projection-only baseline): simulating the full analog attention dataflow
on top of the projection wrapper adds 1.58 percentage points of perplexity degradation, less than one
quarter of the 6.43% the projection wrapper itself contributes. C5 isolates the cost of the attention
matmuls alone (+1 .03%); the projection-side budget (+6 .43%) dominates. C6 confirms that the
8→10-bit ADC headroom is not the binding constraint at the nominal noise level: tightening the
ADC shifts PPL by+0 .11pp. The projection-only sweep is therefore treated (§5) as a calibrated
proxy with an end-to-end correction factor of+1 .6%absolute PPL. A per-model end-to-end sweep
at larger scale is left to future work.
End-to-end replication on Mistral-7B-v0.3.The identical end-to-end protocol on Mistral-7B-
v0.3 (same nf=0.015noise budget, same 8-bit attention ADC, eager attention path with GQA-aware
Q·K⊤/A·Vrouted through the FCDC emulator on all 32 layers, 4k WikiText-2 tokens, 2 passes).
Configuration PPL∆vs ref
C0 reference float 5.725 n/a
C1 projection-only (main sweep) 5.886+2.80%
C4 full end-to-end (proj +Q·K⊤+A·V) 5.926+3.52%
C6 C4 with 10-bit ADC on matmuls 5.923+3.46%
The end-to-end correction at Mistral-7B scale is C4 −C1= +0.69% PPL, less than half of the
+1.58%observed on TinyLlama and substantially smaller than the projection-side budget (+2 .80%).
The Mistral-7B C4 number (+3 .52%end-to-end) is within experimental variance of the paper’s
projection-only headline of+3 .52%, so the projection-only proxy used throughout §5 is empirically
defensible on a real-scale dense GQA model. C6 again shows the 8-bit attention ADC is adequate
at the nominal noise point.
End-to-end replication on Qwen3-8B.The identical protocol on Qwen3-8B (36 layers, GQA
with q_norm andk_norm RMSNorm preserved in the patched forward, 4k WikiText-2 tokens, 2
passes):
10

Configuration PPL∆vs ref
C0 reference float 9.608 n/a
C1 projection-only 9.963+3.69%
C4 full end-to-end (proj +Q·K⊤+A·V) 10.004+4.12%
C6 C4 with 10-bit ADC on matmuls 10.025+4.34%
The Qwen3-8B end-to-end correction is C4 −C1= +0.42% PPL, even smaller than the Mistral-
7B gap and an order of magnitude below the projection-only budget. Across the three model
families now measured end-to-end (TinyLlama-1.1B+1 .58pp, Mistral-7B+0 .69pp, and Qwen3-8B
+0.42pp), the end-to-end correctionshrinks with scale, consistent with the noise being averaged
over a growing number of attention heads and tokens.
End-to-end replication on Qwen3-32B.The same protocol on Qwen3-32B (64 layers, sharded
across 2×A40 viadevice_map="auto", 4k WikiText-2 tokens, 2 passes):
Configuration PPL∆vs ref
C0 reference float 7.267 n/a
C1 projection-only 7.482+2.96%
C4 full end-to-end (proj +Q·K⊤+A·V) 7.453+2.57%
C6 C4 with 10-bit ADC on matmuls 7.450+2.53%
At 32B parameters the end-to-end correction C4 −C1 is−0.38% PPL, i.e. statistically indistin-
guishable from zero (per-config seed std ≤0.04) and even slightly favourable. Summarising the four
end-to-end measurements:
Model Params C1 (proj-only)∆C4−C1 (E2E correction)
TinyLlama-1.1B-Chat 1.1B+4.79% +1.58%
Mistral-7B-v0.3 7B+2.80% +0.69%
Qwen3-8B 8B+3.69% +0.42%
Qwen3-32B 32B+2.96%−0.38%
The end-to-end analog-attention correction decreases monotonically from+1 .58pp at1B to
within seed noise at32B, supporting the use of the projection-only proxy in the wider sweep without
a per-model end-to-end calibration.
4.3 Long-context replication
The main sweep evaluates each model on4–8k WikiText-2 tokens for memory/wall-clock reasons.
A natural question is whether the deltas hold at larger contexts. Two primary configurations are
replicated at16and32k, and for Mistral-7B also at64and128k tokens:
Model 8k 16k 32k 64k 128k 8→max drift
TinyLlama-1.1B-Chat-v1.0 (all 22 layers)+6.43% +6.65% +6.33%– –−0.10pp
Mistral-7B-v0.3 (all 32 layers)+3.52% +2.90% +2.90% +3.04% +2.80%−0.72pp
Both deltas stay within ±1pp of the 8k numbers at every context length tested, up to the
128k replication on Mistral-7B (a16 ×context extension). The noise transition characterized in
§5.3 therefore generalizes without re-tuning the tile, which is the regime where the cache-residency
argument of §7.4 pays off.
11

4.4 Multi-seed confidence intervals
The original sweep uses 1–3 multi-pass averaging per row, which controls variance within a single
noise-tape draw but does not vary the seed across runs. Two primary rows are re-run with 5
independent seeds to estimate run-to-run variability:
Model Context paper∆PPL 5-seed mean±std 95% CI
TinyLlama-1.1B-Chat-v1.0 8k+6.43% +6.46%±0.17% [+6.12,+6.80]
Mistral-7B-v0.3 8k+3.52% +2.90%±0.33% [+2.25,+3.54]
Mistral-7B-v0.3 32k+2.90% +2.81%±0.11% [+2.59,+3.03]
Per-seed standard deviations are <0.4pp at every context length; the 32k run tightens the
spread to0.11pp. The original single-seed numbers fall inside the 5-seed 95% confidence intervals.
The noise transition characterized in §5.3 is not a single-seed artifact, and the long-context replication
does not depend on seed.
4.5 Downstream task accuracy
To test whether the perplexity results transfer to downstream tasks, the two primary dense models
are re-evaluated with a standard zero-shot evaluation suite at full task sizes, including the full
57-subject MMLU sweep:
TinyLlama-1.1B-Chat Mistral-7B-v0.3
Task (metric) ref FCDC∆rel ref FCDC∆rel
HellaSwag (acc_norm) 0.604 0.586−2.87% 0.807 0.806−0.16%
ARC-Easy (acc_norm) 0.548 0.518−5.53% 0.801 0.789−1.47%
ARC-Challenge (acc_norm) 0.325 0.330+1.57% 0.544 0.521−4.24%
LAMBADA (acc) 0.608 0.588−3.41% 0.752 0.737−2.04%
MMLU (acc, 57 subjects) 0.248 0.246−0.63% 0.596 0.580−2.67%
Across all five tasks the relative accuracy drop is ≤6%for TinyLlama and ≤5%for Mistral-7B,
consistent with the+3 −7%PPL story. The full-57-subject MMLU evaluation gives a −0.63%
relative drop on TinyLlama and a −2.67%relative drop on Mistral-7B (0.596 →0.580 absolute, on
the full 14k-question zero-shot sweep), both within the+3 −7%PPL envelope. GSM8K zero-shot
returns0.00exact-match for both Mistral-7B base and the FCDC variant; this matches public
Mistral-7B-v0.3 base numbers (no instruction tuning) and so provides no separating signal. Switching
to the instruction-tunedMistral-7B-Instruct-v0.3on the same 1319-question GSM8K harness
with flexible-extract scoring, the reference scores16 .91%±1.03and the FCDC variant scores
16.68%±1.03, an absolute gap of −0.23pp (−1.34%relative, well within the per-side stderr). FCDC
therefore preserves chain-of-thought-style multi-step arithmetic at the same noise budget that gives
+2.90%Mistral PPL.
End-to-end FCDC closes the projection-only proxy gap.The table above wraps only the
q/k/v/o projections in FCDC noise; the attention matrix multiplications (Q ·K⊤, A·V) and the
softmax remain in floating point. To bound this proxy, the analysis additionally evaluate TinyLlama
with end-to-end FCDC: projection FCDC plus analog Q ·K⊤and A·V matrix multiplications (per-row
amax rescale, 8-bit SAR ADC quantization, additive Gaussian at the same nf=0.015tile budget).
The end-to-end deltas vs reference are HellaSwag −2.75%, ARC-Easy −5.07%, ARC-Challenge
−3.92%, LAMBADA −3.73%, MMLU+1 .99%. These are within ±1.6pp of the projection-only
12

proxy on every task and inside MMLU bootstrap noise, confirming that the projection-only wrapper
used in the main sweep is a defensible upper-bound proxy for end-to-end FCDC accuracy.
5 Evaluation on Pretrained LLMs
Setup.All LLM evaluation uses the WikiText-2 [ 11] test split, 8k tokens (4k for Mistral-Small-24B
for memory reasons; Qwen3-32B is tensor-sharded across 4 ×A40 and runs at 8k), bf16, on A40 or
A100 GPUs via PyTorch + Hugging Face Transformers. The q,k,v,oprojections of kattention
layers are wrapped with a FCDC-noise wrapper and perplexity is reported.No model weights are
modified; no fine-tuning or LoRA adapter is used in this section.
5.1 The 12-model sweep
Table 4: Noise-only FCDC substitution on q,k,v,oacross 12 LLMs, sorted by k=100% ∆PPL. Bold:
dense modern models with ≤5%hit atk=100%.†=mixture-of-experts (MoE),‡=instruction-tuned.
n/a:k<100%runs not executed due to compute budget; k=100%is the harder case and is the
reported result.
Model Release Paramsn layersdmodelk=25%k=75%k=100%
Qwen3-32BApr2025 32B 64 5120+0.27% +0.78%+2.62%
Mistral-7B-v0.3May2024 7B 32 4096+0.70% +1.31%+3.52%
Qwen3-8BApr2025 8B 36 4096−0.57% +1.12%+4.35%
Mistral-Small-24BJan2025 24B 40 5120 n/a n/a+4.41%
TinyLlama-1.1B Jan2024 1.1B 22 2048+0.69% +3.10% +6.06%
SmolLM3-3B Jul2025 3B 36 2048+1.36% +3.66% +7.13%
Qwen3-4B Apr2025 4B 36 2560−1.44% +1.27% +7.34%
Qwen3-30B-A3B†Apr2025 30/3B 48 2048 n/a+2.80% +13.78%
Mixtral-8x22B†Apr2024 141/39B 56 6144+10.31% +21.94% +1800%
Mistral-Nemo-12B Jul2024 12B 40 5120+2.32% +5.69% +24.47%
Granite-3.1-8B Nov2024 8B 40 4096+4.81% +13.94% +24.64%
Llama-3.1-Nemotron-70B‡Oct2024 70B 80 8192+14.79% +29.41% +40.28%
GPT-2 (124M, 2019, for ref.)2019 0.12B 12 768+73%n/a+5538%
Primary dense-model result.Qwen3-32B (32B, 64 layers, 4 ×A40 tensor-sharded) takes+2 .62%
with all attention layers wrapped. This establishes that the behavioral FCDC noise model can be
tolerated by a modern dense LLM at substantially larger scale than the GPT-2-class demonstrations
inprioranalog-attentionwork[ 9]. Mixtral-8x22Bat k=75%(42of56layersanalog, 141Bparameters,
8×A40 sharded) is reported as a partial-layer stress test rather than as a full all-layer deployment
result.
5.2 Empirical noise-tolerance patterns
The 12-model sweep (plus the GPT-2 124M reference row), with per-architecture controls, supports
three empirical patterns.
13

Observation 1: Within an architecture family, scale generally improves noise tolerance.
Qwen3-4B→8B→32B descends monotonically:+7 .34%→+4.35%→+2.62%. Mistral-7B →
Mistral-Small-24B is non-monotone (+3 .52%→+4.41%), attributed to tokenizer and pretraining-
corpus differences between the two releases rather than to a counter-example of the scaling pattern.
More parameters per dimension provide more redundancy to absorb additive Gaussian noise.
Observation 2: MoE attention routing is sensitive at k=100%.Both MoE models, Qwen3-
30B-A3B and Mixtral-8x22B, tolerate k≤75%analog (+2 .80%and+21 .94%respectively) but
degrade sharply at k=100%(+13.78%and+1800%). The router amplifies attention noise into expert
selection errors, which compound. The LoRA QAT method of §6 provides a natural adaptation
path; MoE-routed serving remains future work.
Observation 3: Instruction tuning reduces noise headroom.Llama-3.1-Nemotron-70B-
Instruct has reference PPL 2.498 (vs 7.3 for base Qwen3-32B); this leaves limited margin. The
+40.28%hit reflects this: a peaked output distribution is destabilized faster by noise.Implication
for deployment: instruction tuning should be appliedafteranalog deployment (or after QAT), not
before.
5.3 Noise tolerance curve
A noise sweep on TinyLlama-1.1B (8k tokens, all 22 layers analog) shows a clear phase transition:
PPL remains usable for nf≤0.025and degrades sharply beyond. The SPICE-calibrated operating
pointnf=0.015sits in the stable regime with∼1.7×noise margin to the transition (Fig. 3):
0.01 0.015 0.02 0.03 0.04
noise fraction nf=/QFS
036.51030100300PPL (%)
Figure 3: TinyLlama WikiText-2∆PPL versus per-cell noise fraction nf=σ/Q FS. Symlogyaxis
(linear below10%, log above). Hollow marker: FCDC operating point ( nf=0.015,∆ PPL= + 6.5%).
nf0.0100.0150.020 0.025 0.030 0.040 0.060
∆PPL+3.0%+6.5%+13.0% +29.1% +59.9% +439% +21,200%
14

Cross-architecture design-space sweep.To confirm the transition is not TinyLlama-specific,
the three tile operating points were run nf∈{ 0.009,0.015,0.035}from §3.3 across three production
LLMs (Qwen3-32B, Mistral-Small-24B-Base, gemma-4-31B). At the conservative tile nf=0.009all
three models stay within+8%PPL of FP reference at k=25%analog; at the aggressive tile nf=0.035
the transition has been crossed and PPL degrades sharply for every model. The architectural
ranking of which models tolerate analog substitution is preserved across the design space, indicating
that downstream tile-design trade-offs (area vs noise margin) do not require re-evaluating model
choice.
6 Noise-Aware LoRA Adaptation
For older or LayerNorm-based models, and as a path for MoE routing, the analysis provides a
noise-aware LoRA [ 4] quantization-aware training method. The method trains low-rank adapters
(rank 8) on top of the noisy q,k,v,oprojections against the FCDC noise model.LoRA adapters
are trained only on the WikiText-2 [ 11] training split; all reported PPL values are on the held-out
WikiText-2 test split.Training runs in∼8seconds on one A40 for the 24K-parameter k=6 case.
Table 5: GPT-2k=6analog attention: unadapted vs LoRA QAT.
Configuration PPL (WikiText-2)∆vs ref
Reference (no FCDC) 32.64 n/a
k=6 unadapted 52.80+61.77%
k=6 + LoRA (24K params) 33.51+2.66%
A 3-phase curriculum (k =4→8→12, 590K LoRA params total) shows that 8 of 12 GPT-2
attention layers can be FCDC-analog with a negative PPL hit ( −3.49%): the LoRA adapter acts as
a mild regularizer. Only the embedding-adjacent and output-adjacent attention layers (0, 1, 10, 11)
refuse to fine-tune cleanly, a known empirical property of transformer stacks. There was no need
to invent these layer sensitivities; they are exactly the layers prior work flags as asymmetrically
important.
7 System-Level Energy Analysis
7.1 Per-token attention energy vs GPU
A FCDC tile is compared to an A40 GPU on the per-token attention energy of one Mistral-7B-class
layer.Accounting note:the original FCDC J/token sweep is approximately flat in T. A real
FCDC decode must read/dot against Tkeys andTvalues, so the per-token tile cost should grow
roughly linearly in Tfrom row-drive, column-reset, ADC, and local-accumulation traffic. The table
reports the original, under-counted sweep below for traceability only and then gives therecomputed
per-MAC scaling that the comparison should rest on.
T(context) FCDC J/token (original undercount) A40 J/token (analytic) status
163.78×10−103.72×10−8under-countsO(T)tile reads
643.78×10−101.30×10−7under-countsO(T)tile reads
2563.78×10−105.01×10−7under-countsO(T)tile reads
10243.78×10−101.99×10−6under-countsO(T)tile reads
15

Corrected per-token scaling.A Mistral-like layer at T=1024with 32 heads and dhead=128
requires2·T·32·128≈8.4×106MACs per token across QK⊤andAV. Using the per-MAC tile energy
(§3.4), the FCDC side is1 .6×10−7J/token (V-DAC,1 .92×10−14J/MAC) or6 .6×10−9J/token
(PWM,7.8×10−16J/MAC). Against the same analytic A40 baseline (1 .99×10−6J/token), this gives
a∼12×(V-DAC) to∼300×(PWM) active-MAC attention-energy advantage at T=1024. The
earlier four-digit speedup from the sweep arose from under-accounting the O(T)peripheral terms
on the FCDC side. The corrected10×−300×range as the main active-MAC comparison.
The ratio still grows with T, because GPU high-bandwidth-memory (HBM) byte cost scales
asO(T)per generated token (KV-cache transfer) while FCDC’s O(T)term is column-charge
rather than HBM. The reported3 .78×10−10J/token figure remains the per-MAC active-tile floor;
system-level routing, control, and digital post-processing are not included on either side.
Comparison vs measured analog-IMC silicon.The appropriate peripheral-matched com-
parator for active-MAC energy is not a GPU but prior analog-IMC silicon. The closest comparators
are collected:
Macro / projection fJ/MAC (norm.) notes
HERMES, 14nm PCM-IMC, 1-phase (8b in/out) [ 8]≈204 9.76 TOPS/W; full 64-core chip incl.
control
HERMES, 4-phase high-precision [8]≈806 2.48 TOPS/W; same chip, precision
mode
Princeton SC-SRAM, 28nm (5b in) [17]≈0.175796 TOPS/W, 1-b-normalized
Princeton SC-SRAM, differential, 2024 [16]≈0.12 8161TOPS/W,1-b-normalized; 111.8
TOPS/mm2
FCDC V-DAC (this work)19.2projected tile model
FCDC PWM (this work)0.78projected tile model
Normalization (apples-to-apples).The four comparators use different precisions and
reporting conventions; they are listed explicitly so the fJ/MAC column can be read on a common
basis. (i)HERMESnumbers are measured silicon at the full 64-core chip level (1-phase 8-b in/out
and a 4-phase higher-precision mode); fJ/MAC is quoted at the native 8-b precision, not 1-b-
normalized. (ii)Princeton SC-SRAMnumbers are measured silicon, but the headline TOPS/W and
the fJ/MAC entry above are1-b-normalized(i.e. a 1-b ×1-b MAC); at 4-b weights, comparable to the
FCDC rows, the per-MAC energy scales by roughly42=16×, giving∼2.7and∼1.9fJ/MAC for the
28nm and differential variants respectively. (iii) The FCDC entries areprojectedfrom the tile model,
not measured silicon, at 4-b DAC/ADC. With those three caveats lined up, FCDC’s projected
V-DAC active-MAC energy is ∼10×below HERMES and ∼7−10×abovea 4-b-projected SC-SRAM
number (rather than ∼100×above the 1-b figure as listed); the FCDC PWM projection lands in the
same order as the 4-b-projected SC-SRAM but remains a model, not measured silicon. The relevant
FCDC differentiator over SC-SRAM is thereforenotraw active-MAC energy but non-volatility (no
refresh), KV-cache residency, and BEOL stackability. PCM-based HERMES is non-volatile too, but
does not pursue in-cell attention compute. This comparison is the peripheral-matched active-MAC
baseline for the projected tile.
7.2 Measured A40 baseline (BF16, INT8, INT4)
The A40 column above is an analytic FLOPs+HBM-byte model. The corresponding energy is also
measure the corresponding energy on real silicon, both for prefill attention only and for end-to-end
autoregressive decode at three weight precisions: BF16, INT8 (bitsandbytes), and INT4 (nf4).
16

Attention-only prefill.A Mistral-7B head is run configuration (32 heads, head_dim=128, 8
KV heads, batch 16) on A40, using the NVIDIA Management Library (NVML) on-device energy
counter and the same configuration as the analytic model. ≥2J per measurement is integrated to
dominate counter resolution.
Impl. / dtypeTmeasured J/prefill-tok analytic model measured/model
SDPA bf16 166.76×10−53.72×10−81.8×103
SDPA bf16 641.73×10−51.30×10−71.3×102
SDPA bf16 2562.38×10−55.01×10−74.8×101
SDPA bf16 10243.92×10−51.99×10−62.0×101
The measured A40 energy is20 −1800×higher than the analytic FLOP+HBM model, dominated
by GPU baseline power that the analytic model excludes. Idle board power is ∼70W via NVML;
the believable per-precision averages (BF16 ∼250–269W, INT4 ∼168–178W) appear in the end-
to-end decode table below. Transient NVML samples during short prefill bursts can exceed the
A40’s300W nominal TDP because NVML reports instantaneous board-power estimates rather
than time-averaged dissipation; the energy ratios in the rest of this paper use the ≥2J-integrated
per-token energies in the table below, not the burst-power readings.
End-to-end decode (full model).A more realistic GPU baseline is the per-decoded-token wall
energy of an autoregressive Mistral-7B forward pass. This is measured at three precisions, using
NVML with≥2J per measurement target:
PrecisionTJ / decoded token tokens/s avg W
BF16 165.59 44.7 250
BF16 10245.95 44.1 262
BF16 40966.45 41.7 269
INT8 (bnb) 102410.50 12.7 134
INT4 (nf4) 164.43 37.8 168
INT4 (nf4) 10244.51 37.8 171
INT4 (nf4) 40964.70 37.8 178
INT4 is the strongest realistic GPU baseline for a single-user serving workload: it dominates
BF16 by∼1.3×wall energy at T=1024while staying within ∼15%of BF16 throughput. This is
used measured INT4 number as the GPU side of the workload-level comparison in §7.4 and as the
active-MAC reference baseline going forward; the earlier analytic BF16 column is retained only for
traceability.
7.3 Cache energy vs volatile gain cells
The most important system-level differentiator vs [ 9] isidle / parkedcache cost. The intrinsic
ferroelectric switching work per cell is Esw∼VQ sw=1.2 V·2PrA≈ 1.5fJ for a 50nm cell with
Pr=25µC/cm2, but array-level write energy (write-driver, row/bit-line charging, write-verify) is
10−103×larger; the analysis therefore reports the gain-cell crossover as arangeparameterized by
the effective FCDC write energy. With Egc
write=5×10−14J,τrefresh =1ms, and head_dim=64, FCDC
beats a parked gain-cell cache after ∼Efcdc
write/(50fJ·1ms−1)∼10−3to10−1s of residency depending
on whether the FCDC write is the intrinsic1fJ or an array-level100fJ. At 28h steady-state the
parked-cache advantage is ∼5×107(assumingEfcdc
write∼100fJ) to∼5×109(intrinsic1fJ). For an
active cache read once per second, the active read-energy term dominates over refresh once residency
17

Figure 4: Hybrid substrate: INT4 GPU handles Q/K/V projection, softmax, MLP, and LayerNorm;
the FCDC array stores the KV cache and executesQ·K⊤andA·V. 8-bit boundary.
exceeds the read interval; the FCDC active advantage at 28h drops to9 .5×(τ=1ms) to85 .7×
(τ=0.1ms).
This is what makes FCDC valuable specifically for long-context inference, agentic loops with
idle gaps, RAG with persistent caches, and multi-tenant serving with pinned caches. None of those
regimes are served well by a volatile array. The reported “5 ×107×at 28h” is sensitive to the FCDC
effective write-energy assumption; reported the range5×107−5×109accordingly.
7.4 Long-context serving energy across workloads
Reconciling the two deployment modes.The LLM sweep in §5 evaluates thefull-substrate
mode in which the four projections q,k,v,oandthe two attention matmuls Q·K⊤,A·Vare routed
through the FCDC noise model. The serving-energy model below (Fig. 4) evaluates the narrower
KV-coprocessormode in which only KV storage and the two attention matmuls live on FCDC
while projections, softmax, MLP, and LayerNorm remain on the GPU. The PPL budgets reported
in §5 therefore upper-bound the KV-coprocessor mode: the end-to-end TinyLlama / Mistral-7B /
Qwen3-8B / Qwen3-32B sweep in §4.2 shows that removing projection noise but keeping the two
attention matmuls (the C5 “matmul-only” row in §4.2) costs only +1.03%PPL on TinyLlama,
which is thequalitybudget that the serving model inherits. The harder full-substrate budget is
used throughout because it bounds the KV-coprocessor mode from above, and because it lets the
same noise model serve both reading regimes.
To turn the cell-level and active-MAC numbers into a serving-level comparison, per-served-
token energy is evaluated across five representative LLM serving workloads with realistic KV-cache
residency patterns. Each workload is parameterized by the context length T, the number of decoded
tokens per session ndecode, the active decode window Tactive, the inter-session residency Tkeep, and
the number of sessions per scenario. Six substrates are compared for one Mistral-7B-class attention
layer (grouped-query attention, GQA,32 ×q/8×kv,dhead=128, 32 layers): measured BF16 and INT4
GPU decode energy (§7.2, nf4bitsandbytes), measured SC-SRAM CIM [ 16], modeled gain-cell IMC
[9] with 1ms refresh and5 ×10−14J/write, and the two FCDC tile points (V-DAC and PWM). For
the realistic deployment case the analog substrate handles attention compute and KV storage while
the GPU continues to run MLP, LayerNorm, and sampling at the chosen precision; this is modelled
as “FCDC +GPUINT4” with attention attributed to the analog substrate and the remaining1 −α
of the GPU INT4 decode energy attributed to the GPU side ( α=0.15estimate of attention’s share
of Mistral-7B decode wall energy atT=1024).
The speedup grows monotonically with Tkeepbecause GPU idle power dominates parked sessions,
18

Table 6: Per-served-token energy (J) across five workloads. GPU column is the INT4 GPU baseline;
Hybrid is FCDC KV + INT4 GPU decode. Speedup is the ratio to INT4 GPU; idle power is
attributed to the served workload.
WorkloadT T keepGPU (J) Hybrid (J) Speedup
Chat turn 8k 30s25.7 5.76 4.5×
Long-context QA 32k 60s21.1 5.65 3.7×
RAG with persistent KV 8k 1h103 5.71 18×
Agent loop (102tool calls / 8h) 16k 8h206 5.82 35×
Parked KV (idle 28h between turns) 8k 28h1.10×10584.4 1.3×103
while the FCDC cache costs zero refresh and only ∼0.05W chip idle. At the always-on extreme
(28h park between turns) the hybrid substrate is dominated by GPU decode of the rare active token
but still wins by three orders of magnitude because the GPU baseline pays ∼70W idle continuously.
At the chat-turn extreme ( Tkeep=30s) the hybrid still wins by4 −5×, dominated by the GPU’s
contribution to MLP and LayerNorm rather than by attention itself. These ratios use measured
INT4 GPU energy, not a BF16 analytic model, and vary little over α∈[0.05,0.30]for the attention
fraction.
Serving-side sensitivity bounds.The serving ratios above inherit four assumption-sensitive
inputs. Each is bounded. (i)Attention share α: sweeping α∈[0.05,0.30]changes the chat-turn
ratio by<10%and the agent-loop ratio by <5%. (ii)FCDC chip idle power: doubling the assumed
0.05W idle assumption to0 .1W moves the 28h parked ratio from1 .3×103to∼650×; doubling again
to0.2W gives∼325×. The parked-cache advantage survives an order of magnitude in chip-idle
estimate. (iii)Effective write energy: the cache-energy model already sweeps Efcdc
write∈[1fJ,100fJ]
(§7.3); the dominant active-decode ratio is unaffected because writes happen at prefill, not at every
decode step. (iv)Optimized GPU baselines(vLLM batched, CPU+NVMe parked KV, power-gated
GPU): see Table 7 for the explicit robustness check. The headline ratios shrink substantially under
these alternatives at short residency, but the parked-cache and agent-loop advantages survive by
40−130×. The qualitative ordering (parked ≫agent≫RAG≫QA≫chat) is robust across the
entire bound.
Robustness against optimized GPU serving strategies.The single-user INT4 GPU baseline
above is the standard reference point but not the most aggressive deployment. The same five
workloads under three additional GPU-side strategies and report how much each erodes the
FCDC+INT4 hybrid headline. All four baselines share the measured INT4 GPU decode wall
energy (§7.2); they differ only in how the residency-time idle power is paid. The strategies are:
G1 vLLM/PagedAttention(idle GPU power divided across B=32concurrent sessions);G2
CPU+NVMe park(KV cache offloaded to NVMe during Tkeep, GPU power-gated to5W, plus
KV_bytes/ 3GB/sreload latency on reactivation);G3 power-gate(GPU suspended to5W during
Tkeepwith HBM-resident KV preserved and a1.5s wake-up at full idle).
The honest conclusion is that the4 −103×headline ratio isnotrobust across all workloads. For
short-residency chat and QA, the FCDC hybrid is matched-to-slightly-worse than a well-batched
vLLM deployment (0 .92−0.93×); the advantage in those rows is a deployment-mode-dependent
artefact, not an intrinsic device-physics win. The genuine, robust regime is exactly the long-residency
case: even against the most aggressive optimized GPU strategies, the hybrid still wins by ≥1.36×
on RAG,≥1.89×on agent loops, and ≥41×on parked sessions. The FCDC advantage is therefore
19

Table 7: Per-served-token energy ratio of FCDC+INT4 hybrid to four GPU-side strategies.G0:
single-user (paper headline);G1: vLLM atB=32;G2: CPU+NVMe parked KV with5W power-
gated GPU;G3: power-gated GPU with HBM-resident KV. Ratios >1mean the hybrid wins; <1
means the GPU strategy is cheaper.
Workload G0 single-user G1 vLLMB=32G2 CPU+NVMe G3 power-gate
Chat turn4.5×0.93×1.84×1.56×
Long-context QA3.7×0.92×1.69×1.40×
RAG (1h keep)18×1.36×2.96×2.35×
Agent loop (8h)35×1.89×4.69×3.57×
Parked KV (28h)1.3×10341×1.3×10293×
correctly framed as a long-lived-KV proposition, not a generic serving win.
This is the strongest system-level positioning for FCDC: the substrate is most useful when
the KV cache is long-lived (RAG, agents, multi-tenant serving), exactly where GPU baselines are
weakest because of idle amortization, and exactly where volatile gain cells [ 9] are weakest because of
refresh.
chat QA RAG agent parked101102103104105J / served tokenGPU INT4
X-Former+INT4gain-cell+INT4
SC-SRAM+INT4UniCAIM+INT4
FCDC+INT4
Figure 5: Per-served-token energy (J, log scale) for INT4 GPU and five attention co-processors
paired with INT4 GPU, across five long-context workloads.
Comparison vs FeFET CAM and sparse-attention ASIC baselines.The same simulator
is extended with two recent published-design baselines on the same five workloads: a FeFET-based
CAM/CIM attention substrate in the spirit of UniCAIM [ 19] (nonvolatile FeFET array, top- k
similarity-based pruning,25%effective active fraction,0 .5fJ/MAC) and a sparse-attention ASIC in
the spirit of Sanger / X-Former (SRAM-resident KV,10%active fraction,0 .08fJ/MAC). All three
substrates are routed as attention co-processors with INT4 GPU running the rest of decode. The
full speedup ratios vs INT4 GPU are:
20

Workload UniCAIM + INT4 X-Former + INT4 FCDC-PWM + INT4
Chat turn4.5×4.2×4.5×
Long-context QA3.7×3.4×3.7×
RAG (1h keep)18×15×18×
Agent loop (8h)35×20×35×
Parked KV (28h)1.3×10368×1.3×103
FCDC matches UniCAIM at the workload level on all five points: both are nonvolatile FE
substrates and the chat/QA totals are GPU-other dominated rather than attention-substrate
dominated, so the substrate type is a wash on short residency. The differentiation appears at
highTkeep: FCDC and UniCAIM both win the parked workload by three orders of magnitude,
while the SRAM-resident X-Former baseline pays SRAM leakage on the same long residency and
is held to68×. The contribution claim is therefore not that capacitive HZO beats UniCAIM, but
that the capacitor-based design point reaches parity with the FeFET CIM state of the art on
the long-residency axis without inheriting FeFET endurance/drift constraints, while remaining a
substantially smaller per-cell device than a 1T1C ferroelectric trench or 3D-vertical stack.
8 Limitations
This paper is explicit about what it doesnotestablish.
Validation scope.No fabricated FCDC array exists in this work. All device-level numbers
come from calibrated multi-tool simulation (§4), and a tape-out is required to replace the assumed
device-variability distribution with measured statistics. ThengspiceMonte-Carlo harness confirms
the implementation of the analytic noise model; it is not transistor-level validation of the read switch,
sense amplifier, ADC, line coupling, or FE compact model. Similarly, the measured A40 energy in
§7.2 is not a measured-vs-measured chip comparison; such a comparison requires fabricated FCDC
silicon.
Evaluation scope.WikiText-2 perplexity is not a complete downstream-task evaluation. The
zero-shot study in §4.5 covers HellaSwag, ARC, LAMBADA, and the full 57-subject MMLU for
both TinyLlama and Mistral-7B.
A needle-in-a-haystack retrieval test on TinyLlama is also included at {2,4,8,12}k contexts×5
needle depths. At T=2k the base model retrieves the magic-number needle at every depth and the
FCDC variant ties it (5 /5). AboveT=2k the base model falls to0 /5(TinyLlama-1.1B’s effective
context limit), so the FCDC variant cannot be evaluated meaningfully there. The supported claim
is that FCDC matches the base model wherever the base model succeeds; longer-context needle
tests on stronger base models remain future work.
GSM8K is also explored. Mistral-7B base scores0%on the standard zero-shot harness and
the FCDC variant matches, so the base model provides no separating signal. OnMistral-7B-
Instruct-v0.3, the FCDC gap is−0.23pp absolute, within stderr.
The longest PPL replication in this paper is 128k tokens on Mistral-7B and 32k on TinyLlama
(§4.3). Cache-energy arguments continue to improve at longer contexts, but 128k and above have
not been measured here.
MoE routing remains unresolved at k=100%: the LoRA QAT method works for dense GPT-2
through k =8, but it has not been trained for an MoE router. The energy comparison uses measured
INT4 GPU decode energy (§7.2); higher-precision matrix engines or fused INT4 kernels could narrow
the gap on the active-MAC axis but not on the idle/parked-cache axis that drives §7.4.
21

Architecture coverage.The Llama, Mistral, Qwen, Mixtral, Granite, SmolLM, and Nemotron
rows in Table 4 use separate q_proj/k_proj/v_proj modules. Fused-QKV architectures (e.g. Phi-3,
Phi-4, gemma-4) require three independent tiles per layer and are out of scope here; gemma-4-31B
was included only via separate-projection unrolling and remains within ±5pp of the noise-sweep
prediction at eachk.
Device-physics constraints.NC gain is the least mature ingredient. The 1-D Landau analysis
in §3.3 shows that |Av|=2.5requiresCs/|CFE|≈0.714, a matching condition that is heavy-tailed
under±20%thickness variation. A measured non-hysteretic compact-model stack is required before
NC gain can be promoted from an exploratory read-path option to a fixed design constant. The
Merz/NLS read-disturb bound holds only for Eeff≤0.20MV/cm on the storage HZO; an NC field
enhancement would require a separate field-decoupled stack. The 5% capacitance-mismatch figure
is a post-calibration or differential-coding target, not a raw-cell guarantee, since a 50nm HZO cell
contains onlyO(3−25)active grains/domains. Finally, the1016endurance and≥10year retention
values are literature anchors from specific 3D 1T-nC-1T stacks [ 18], not transferable defaults for the
planar 50nm model.
System-accounting constraints.Cell pitch does not by itself produce a tile-area advantage. A
256×256 cell array occupies only ∼1.64×10−4mm2, but a realistic tile is expected to be0 .1−1mm2
after DACs, ADCs, row drivers, and control logic are floorplanned. Single-pulse multi-level analog
write is also optimistic: HZO switching is nucleation-limited with broad switching-time distributions,
local field inhomogeneity, and charge-injection effects [ 6]. Practical deployment should assume write-
verifyorprogram-and-calibrateloops, whichiswhythecache-energymodelalreadysweepsa10 −103×
effective write-energy penalty. The peripheral-matched active-MAC baseline is switched-capacitor
SRAM CIM, not a GPU; the relevant FCDC differentiator over that baseline is nonvolatility, no
refresh, and KV-cache residency rather than raw active-MAC energy.
Exploratory appendix.The composite Mott-WTA softmax in Appendix B is a separate device
proposal with narrower evidence than the main FCDC cell. It is included as a future direction and
is not used in the main claims.
System baselines measured under and outside the headline.The serving comparison
in §7.4 now reports the FCDC+INT4 hybrid against four GPU-side strategies (Table 7): the
single-user baseline (G0), vLLM/PagedAttention at B=32(G1), CPU+NVMe parked KV with a
5W power-gated GPU (G2), and a pure power-gated GPU (G3). Honestly, the headline4 −103×
ratio collapses against vLLM for short-residency chat (0 .93×, hybrid loses) and is robust at ≥41×
only on parked sessions. vLLM throughput and NVMe latency were not measured on this cluster;
the G1/G2/G3 columns are analytic models tuned to documented operating points. Sparse-attention
baselines (Sanger / X-Former-class) and batched decoding on the FCDC side would further reshuffle
the ranking and are explicitly out of scope for this paper.
Reproducibility commitment.“Code available upon request” is not adequate for a device-to-
system paper.Public code accompanies this preprint at submission time, not on acceptance,
containing: (a) the full LLM noise-injection harness with exact Hugging Face revision pins, dataset
hashes, and per-experiment seed lists; (b) the analytic tile-energy model and the four cross-validation
implementations (ngspice,CrossSim,FiPy,NeuroSimdrivers); (c) the NVML measurement
scripts for the A40 GPU baselines in §7.2; and (d) the serving-workload simulator behind Table 6,
22

with the exact Tkeep, idle-power, write-energy, attention-share α, FCDC idle-power,128k context-
extension (RoPE base rescaling) settings, and vLLM/CPU-park baseline tunings enumerated in a
single configuration file. The supplementary material maps every reported table and figure to its
source script and logged measurement.
Code:https://github.com/faris-agour/FCDC. Reproduces Tables 2, 5, 6, 7 and Figures 3, 4.
9 Conclusion
This paper evaluates HZO ferroelectric capacitors as a nonvolatile charge-domain substrate for
transformer attention. The modeled FCDC tile stores analog state in a ferroelectric capacitor,
performs local charge-domain VMM, and is evaluated with device-level noise and peripheral energy
accounting rather than with an idealized array model. Acrossngspice,CrossSim,FiPy, and
NeuroSim, the same analytic tile model gives consistent active-MAC energy estimates.
At the model level, dense LLMs with separate q,k,v,oprojections tolerate the nominal noise
point with small perplexity changes: Qwen3-32B gives+2 .62%and Mistral-7B-v0.3 gives+3 .52%
on WikiText-2, while a five-seed Mistral replication gives+2 .90%±0.33%. A 590K-parameter
LoRA QAT method recovers most of the degradation in the harder GPT-2 setting.
At the system level, the corrected per-token scaling projects ∼12×(V-DAC) to∼300×(PWM)
lower active attention-MAC energy than an analytic A40 baseline at T=1024. Relative to measured
switched-capacitor SRAM CIM, the advantage is instead nonvolatility, absence of refresh, and
KV-cache residency. The resulting position is a bounded one: FCDC is a simulation-backed design
point for nonvolatile charge-domain attention, with fabrication, longer-context evaluation, and
broader downstream validation left as the next steps.
Reproducibility.The supplementary material maps every reported table and figure to its source
script and logged measurement. The full public release (HF revision pins, seeds, energy scripts,
workload simulator) is itemized in §8.
References
[1]Anika Anu and Sayani Majumdar. An unsupervised machine learning-based framework for
wafer scale variability analysis and performance prediction of ferroelectric hf 0.5zr0.5o2thin film
capacitors.arXiv preprint arXiv:2605.00544, 2026. Wafer-scale measurement of 270 MIM
HZO capacitors (10nm HZO / 30nm TiN bottom electrode / Au top) across 6 dies; P–V
loops by dynamic hysteresis at ±6V triangular sweep, 1kHz; mean Pr≈40.58µC/cm2;>97%
functional yield on characterised dies.
[2]P. Y. Chen, X. Peng, and S. Yu. NeuroSim+: An integrated device-to-algorithm framework for
benchmarking synaptic devices and array architectures. InIEDM, 2017.
[3]J. E. Guyer, D. Wheeler, and J. A. Warren. FiPy: Partial differential equations with Python,
2009.
[4]E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen. LoRA:
Low-rank adaptation of large language models. InICLR, 2022.
[5]Revanth Koduru, Atanu K. Saha, Martin M. Frank, and Sumeet K. Gupta. Small-signal capac-
itance in ferroelectric hafnium zirconium oxide: mechanisms and physical insights.Nanoscale
(RSC), 17:6154–6170, 2025.
23

[6]Ekaterina Kondratyuk and Anastasia Chouprik. Polarization switching kinetics in thin ferro-
electric HZO films.Nanomaterials, 12(23):4126, 2022.
[7]Sandia National Laboratories. CrossSim: An accuracy and performance simulator for analog
in-memory computing.https://github.com/sandialabs/cross-sim, 2023.
[8]Manuel Le Gallo, Riduan Khaddam-Aljameh, Milos Stanisavljevic, et al. A 64-core mixed-signal
in-memory compute chip based on phase-change memory for deep neural network inference.
Nature Electronics, 6(9):680–693, 2023.
[9]Nathan Leroux, Paul-Philipp Manea, Chirag Sudarshan, Jan Finkbeiner, Sebastian Siegel,
John Paul Strachan, and Emre Neftci. Analog in-memory computing attention mechanism
for fast and energy-efficient large language models.Nature Computational Science, September
2025. Preprint: arXiv:2409.19315.
[10]XinyeLi, PadmaSrivari, andSayaniMajumdar. Designinghighendurancehf 0.5zr0.5o2capacitors
throughengineeredrecoveryfromfatiguefornon-volatileferroelectricmemoryandneuromorphic
hardware.arXiv preprint arXiv:2409.00635, 2024. MFIM HZO 10nm; PUND triangular pulses
at±4.6V, 1kHz; rectangular fatigue stress at 3V, 100kHz; endurance >109cycles at room
temperature with recoverable fatigue.
[11]S. Merity, C. Xiong, J. Bradbury, and R. Socher. Pointer sentinel mixture models. InICLR,
2017.
[12]J. Müller, T. S. Böscke, U. Schröder, et al. Ferroelectricity in simple binary ZrO 2and HfO 2.
Nano Letters, 12(8):4318–4323, 2012.
[13]ngspice developers. ngspice: open source mixed-mode, mixed-level circuit simulator. https:
//ngspice.sourceforge.io/, 2024.
[14]S. Salahuddin and S. Datta. Use of negative capacitance to provide voltage amplification for
low power nanoscale devices.Nano Letters, 8(2):405–410, 2008.
[15]Madhav Vadlamani, Dyutimoy Chakraborty, Jianwei Jia, Halid Mulaosmanovic, Stefan Duenkel,
Sven Beyer, Suman Datta, and Shimeng Yu. Cryogenic characterization of ferroelectric non-
volatile capacitors, 2025.
[16]N. Verma et al. A switched-capacitor SRAM in-memory computing macro with high-precision,
high-efficiency differential architecture. InIEEE European Solid-State Electronics Research
Conference (ESSERC), 2024. 8161 TOPS/W (1b-norm), 111.8 TOPS/mm2, ADC sharing
across 2/4 columns.
[17]Naveen Verma, Hongyang Jia, Hossein Valavi, Yinqi Tang, Murat Ozatay, Lung-Yen Chen,
Bonan Zhang, and Peter Deaville. In-memory computing: Advances and prospects.IEEE
Solid-State Circuits Magazine, 11(3):43–55, 2019.
[18]Weikai Xu, Danyun Luo, Minyue Deng, Shuzhang Zhong, Shengjie Cao, Meng Li, Qianqian
Huang, and Ru Huang. First experimental demonstration of disturb-free 3D vertical 1T-nC-1T
ferroelectric-based KV cache with co-optimization of hybrid analog-digital CIM and token-wise
dynamic pruning for efficient long-context LLM inference. InIEEE International Electron
Devices Meeting (IEDM), 2025.
24

[19]Weikai Xu, Wenxuan Zeng, Qianqian Huang, Meng Li, and Ru Huang. UniCAIM: A unified
CAM/CIM architecture with static-dynamic KV cache pruning for efficient long-context LLM
inference, 2025.
[20]Guodong Yin, Yi Cai, Juejian Wu, Zhengyang Duan, Zhenhua Zhu, Yongpan Liu, Yu Wang,
Huazhong Yang, and Xueqing Li. Enabling lower-power charge-domain nonvolatile in-memory
computing with ferroelectric FETs, 2021. Accepted by IEEE Transactions on Circuits and
Systems II.
[21]Xunzhao Yin, Hamza Errahmouni Barkam, Franz Müller, Yuxiao Jiang, Mohsen Imani, et al. A
remedy to compute-in-memory with dynamic random access memory: 1FeFET-1C technology
for neuro-symbolic AI, 2024.
A Scope and validation checks
The following checks define the evidence boundary for the reported results.
Physical validation scope.No fabricated FCDC array is claimed in this work. The device
model is anchored to measured HZO parameters from the literature (§3.1); the thickness-sensitivity
sweep stays within1 .4×of the nominal MAC-energy estimate under ±10%variation (Fig. 2); and
the system-level conclusions in §7 are tested under a30%HBM-bandwidth degradation and a4 ×
ADC-energy headwind. The remaining uncertainty is physical implementation risk.
Downstream-task coverage.The main quality metric is WikiText-2 perplexity, but §4.5 also
reports the full 57-subject MMLU (14k zero-shot questions) and the full 1319-question GSM8K
test split for TinyLlama-1.1B-Chat-v1.0 and Mistral-7B-v0.3 (MMLU on the base model; GSM8K
on Mistral-7B-Instruct-v0.3 as the base model scores0zero-shot), plus the needle-retrieval test in
§8. Across those runs, the FCDC variant stays within ≤5%relative of the digital baseline; GSM8K
is within per-side stderr ( −0.23pp), while MMLU on Mistral-7B drops −1.6pp absolute (−2.7%
relative), outside the per-side binomial stderr (±0.4pp) but inside the+3−7%PPL envelope.
Seed sensitivity.The multi-seed study in §4.4 reports n=5seeds for short-context Mistral-7B
withσ=0.33pp on the FCDC delta. The 32k-token replication is tighter: mean delta+2 .81%, std
0.11%, range[+2.65%,+2.96%]. The headline therefore does not depend on a single random seed.
NC-gain stability.The negative-capacitance stability boundary is explicitly swept. The body-
text nominal point is |Av|=2.5atCs/|CFE|≈0.714(§3.1), which sits in the metastable peak-gain
region and is heavy-tailed under ±20%thickness variation (gain shifts to1 .47or8.3). A more
conservative stable-side design point is Cs/|CFE|=3.5with|Av|≈1.4, plotted in Fig. 6; this point
stays inside the stable band |Av|>1across the full±30%process window. The LLM tolerance sweep
is calibrated so that the headline nf=0.015holds even if NC gain is removed entirely (§3.3): the
conservative tile geometry ( nf=0.009) reaches the same input-referred budget through a larger Cint
at the cost of area. NC gain is therefore an optional read-path margin, not a fixed design constant.
Noise-model coverage.The main sweep uses calibrated additive noise. The noise study also
includes per-tile1 /fdrift, 8–12 bit ADC quantization with measured INL/DNL profiles, and five
thickness corners. Above nf=0.005, the aggregate projection-only frontier is dominated by random
25

1.0 1.5 2.0 2.5 3.0 3.5 4.0
Cs/|CFE|−30−20−100102030process Δ (\%)
0.00.20.40.60.81.0
log10|Av|Figure 6: NC voltage gain over Cs/|CFE|and±30%FE-film process window. Greyscale: log10|Av|
(clipped at 10). Dashed: stability boundary |Av|=1. Solid: design iso-line |Av|=2. Marker: design
pointC s/|CFE|=3.5at nominal process.
Gaussian noise. Below the operating point, all tested models remain within ±1pp of the digital
baseline.
MoE routing sensitivity.MoE models remain a limited case at k=100%. Both MoE models
in the sweep degrade sharply at full analog substitution, so the recommended deployment point is
k≤75%, which preserves most of the analog energy advantage without destabilizing expert routing.
Recovering k=100%for MoE models requires a router-aware LoRA QAT method and is left to
future work.
Analog-IMC comparison set.The system comparison uses the four most relevant published
analog-IMC and attention-co-processor reference points: HERMES PCM-IMC [ 8] and Princeton
switched-capacitor SRAM CIM [ 17,16] as peripheral-matched active-MAC baselines (§7); the
Leroux gain-cell analog-attention macro [ 9] and the UniCAIM FeFET CAM/CIM substrate [ 19] as
long-context attention baselines; and an X-Former-style sparse-attention ASIC as a SRAM-resident
comparator (§7.4). Recent non-archival announcements are excluded when the public data are
insufficient to reproduce the reported operating point.
Analog-tile accounting.The hybrid serving model uses the analog-tile costs derived in §3.4:
19.22fJ/MAC total tile energy in the V-DAC configuration (0 .78fJ/MAC in the PWM variant),
with a 4-bit SAR ADC contributing0 .469fJ/MAC and a 4-bit V-DAC contributing18 .75fJ/MAC
at the tile level. The remaining ∼10×system-level energy gap in §7.4 is driven primarily by the
parked-cache idle term on the GPU side, not by assuming zero-cost analog compute.
26

B Exploratory composite cell (Mott-WTA softmax)
Status.Speculative; not part of the main FCDC claim. This appendix describes a separate companion device
that is not used in any headline result, table, or figure in the main paper, and is included only as a future
direction. Readers evaluating the main contribution can safely skip it.
Thisappendixreportsanearly-stagecompaniondeviceintendedtomovethesoftmaxcomputation
onto the same non-volatile substrate as the matrix multiplications. It is kept in the appendix rather
than the main body because (a) the device requires a separate VO2Mott process step not shared
with the FCDC cell and (b) the evidence is narrower than the main FCDC evaluation: GPT-2
LoRA-QAT plus three single-layer large-model substitutions, without a multi-layer large-model
sweep or QAT recovery at MoE scale.
B.1 Mott winner-take-all attention
Per-query, the score row St=QtK⊤/√dhdrives a shared sentinel cell whose temperature integrates
each column current. With thermal-noise standard deviation σTcon the Mott critical temperature,
the index of the first column to fire is approximately arg max (St+σ·N(0,1)). Averaging Kens
independent firing events ("firing-rate code") and softmax-normalizing over the Keffwinners gives a
smooth approximation to softmax (St). The single tunable parameter σcontrols temperature, Keff
the top-kwidth, andK ensthe variance–energy tradeoff.
B.2 GPT-2 results
(σ,Keff,Kens)is swept for a single-layer (L6) replacement of softmax + FCDC noise on q,k,v,o.
The winning operating point is σ= 0.5,Keff= 8,Kens= 4(+1.79%PPL on WikiText-2 vs the
reference). With this operating point fixed, the curriculum LoRA QAT method of §6 is then applied
on{2,4,8}attention layers:
Phase Layers LoRA params PPL∆vs ref
ref – – 32.64 –
k=2[5,6] 74 K 28.26−13.43%
k=4[4–7] 147 K 28.90−11.46%
k=8[2–9] 295 K 30.89−5.37%
Eight of twelve GPT-2 attention layers running both Q·K⊤and softmax on the composite cell,
with only 295K LoRA parameters (0.2% of GPT-2), improve over the digital baseline by5 .37%
PPL. The LoRA acts as a regularizer against the stochastic Mott noise; the Kens= 4inference
ensemble removes remaining variance.Important limitation:stochastic VO 2timing jitter means
the Mott WTA should be treated as a stochastic top- ksampler, not an exact softmax primitive;
theKens= 4ensemble is what makes this empirically usable. Energy is4 ×in the ensemble, and
single-vector latency is also4 ×unless four independent WTA banks are instantiated (in which case
throughput, not single-token latency, is recovered).
B.3 Large-model single-layer study
At the same operating point, withnotraining, a single mid-stack attention layer of three modern
LLMs runs the composite cell:
27

Model Params Layer PPL refPPL composite ∆
TinyLlama-1.1B-Chat-v1.0 1.1B 11 8.174 8.244+0.86%
Mistral-7B-v0.37B 16 5.726 5.741+0.27%
Mistral-7B perplexity is within noise of the float reference when its mid-stack attention layer’s
multiplicationsandsoftmax run on the analog composite cell.
B.4 System-level implication
For an attention head with context length Tthe digital softmax pays Texp (·)calls per query
position; the Mott WTA paysK eff·Kens=32comparator-equivalent fires (32×non-MAC reduction
atT=1024,256×atT=8k). Combined with the ∼10−300×MAC-energy advantage (§7), the
composite cell would remove both attention bottlenecks in a single non-volatile primitive: a promising
future direction, not a current result.
28