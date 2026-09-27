# Computation Over Geometry: Meaning Identity Is Computed, Not Shipped in the Embeddings

**Authors**: Jiaqi Deng

**Published**: 2026-09-23 15:36:20

**PDF URL**: [https://arxiv.org/pdf/2609.28290v1](https://arxiv.org/pdf/2609.28290v1)

## Abstract
Meaning identity (whether two sentences say the same thing after wording changes) is treated in retrieval and RAG as a geometric fact about independently encoded sentence vectors. We show that, for frozen off-the-shelf encoders and language models, it is not: identity is computed when both sentences share one forward pass, and is not a property of the embedding geometry those systems ship. On overlap-matched PAWS-X, purpose-built encoders (BGE, E5, GTE, MiniLM, E5-Mistral-7B) reach English confirm AUC only 0.55-0.65 (dense peak 0.70). Independently encoded last-token states of Llama 3, Mistral, and Qwen do no better; late fusion of the two vectors stays near chance. The same probe on a joint forward pass reaches 0.90-0.96 from 1.5B to 32B, collapses under partner shuffle, is mid-depth, saturates near 0.94 by 3B, and appears more weakly in GPT-2 XL (0.76). The gap holds beyond Llama-style models on other causal LMs, bidirectional encoders (DeBERTa, RoBERTa), and encoder-decoders (Flan-T5, T5, BART). Fixed or linear readers over frozen independent encodings never unlock identity; nonlinear pair readers recover part of it only on the full 49k-pair PAWS train split (0.68-0.87). Off-the-shelf rerankers split: BGE-reranker-large reaches 0.94, while MS-MARCO and Jina stay at 0.55-0.64. Independently trained families compute the same relation and a 1.5B joint reader can distill it from unlabelled teacher scores, while no linear function of the teachers own independent vectors can. Bi-encoders can be fine-tuned to fit PAWS (0.87-0.93), but transfer and STS-B suffer. Cosine compares wording neighbourhoods; identity is a cheap computed operator, not a property of either sentence vector.

## Full Text


<!-- PDF content starts -->

Computation Over Geometry:
Meaning Identity Is Computed,
Not Shipped in the Embeddings
Jiaqi Deng
Independent Researcher
djq627@163.com
Abstract
Meaningidentity—whethertwosentencessaythesamethingafterthewordinghaschanged—istreated
throughout retrieval and RAG as a geometric fact about independently encoded sentence vectors. We
showthat,forfrozenoff-the-shelfencodersandlanguagemodels,itisnot: identityiscomputedwhenboth
sentencesoccupythesameforwardpass,anditisnotapropertyoftheembeddinggeometrythosesystems
ship. Onoverlap-matchedPAWS-X,purpose-builtencoders(BGE,E5,GTE,MiniLM,andE5-Mistral-7B)
reach English confirm AUC only 0.55–0.65(dense peak 0.70;𝑛=900). Independently encoded last-token
states of Llama 3, Mistral, and Qwen do no better; a linear probe on the concatenation of the two vectors
(late fusion) remains near chance. The same probe on the last token of ajointforward pass over both
sentences reaches 0.90–0.96from 1.5B through 32B, and shuffling sentence B collapses it to chance; the
signal is mid-depth, saturates near 0.94by 3B, and is already present, more weakly, in GPT-2 XL ( 0.76).
ThepatternisnotaLlama-familyartefact: thesamefourreadoutsonnon-Qwen/Llama/Mistralcausal
models,onbidirectionalencoders(DeBERTa, RoBERTa),andonencoder–decoders(Flan-T5,T5, BART)
again give joint≫late fusion with shuffle near chance. No fixed or linear reader over frozen independent
encodings—cosine, token-level MaxSim, late fusion—makes identity accessible at any training size;
nonlinear pair readers trained from scratch, an MLP over the two vectors or a cross-attention module over
the frozen token sequences, recover part of it only on the full 49k-pair PAWS train split ( 0.68–0.87), forty
times the pairs the joint probe needs. Off-the-shelf rerankers split the same way: BGE-reranker-large
reaches 0.94, but MS-MARCO and Jina rerankers stay in the dense band ( 0.55–0.64): joint scoring is
necessary, not sufficient. Independently trained families compute thesamerelation: their joint scores
agree, their errors co-occur, and a 1.5B joint reader distilled from unlabelled teacher-scored pairs recovers
it, while no linear function of the teacher’s own independent vectors can. A bi-encoder can be fine-tuned
to fit PAWS (AUC 0.87–0.93), but the fit is a PAWS-specific criterion: transfer to overlap-matched QQP
drops and STS-B Spearman falls by about 0.25. Off-the-shelf cosine compares wording neighbourhoods;
identity is a cheap computed operator, not a property of either sentence’s vector.
1 Introduction
Asentenceembeddingisaninvitationtotreatmeaningasapoint. Onceeachsentencehasavector,“same
meaning” becomes cosine, and the industrial stack—dense retrieval, duplicate-question detection, RAG—is
built on that geometry (Reimers and Gurevych, 2019; Muennighoff et al., 2023; Gao et al., 2024). The
invitation is so familiar that it is easy to miss the empirical claim it encodes: that identity of meaning is a
property of a single sentence’s representation, recoverable by comparing two such properties.
That claim can be false even if embeddings are useful. They can rank topical neighbours, cluster
documents, and win STS (Cer et al., 2017) while still being nearly blind to the distinction PAWS was built to
isolate: two sentences that share almost all of their words, one of which is a paraphrase and one of which is
1
arXiv:2609.28290v1  [cs.CL]  23 Sep 2026

Table 1:English confirm AUC at the layer of peak joint probe. Late fusion and shuffle are the two falsifiers of a
computational reading. Chance=0.5. Qwen2.5-32B joint 95% CI is[0.897,0.969]; 14B is[0.925,0.971].
Model layer Cos Sep LateJointShuf
Qwen2.5-14B L36.614.615.541.950.496
Qwen2.5-7B-Inst L21.571.612.529.961.526
Qwen2.5-1.5B-Inst L21.565.609.542.947.509
Qwen2.5-7B L21.556.572.483.941.520
Qwen2.5-32B L58.569.557.493.938.473
Qwen2.5-3B L27.526.538.555.937.521
Qwen2.5-1.5B L21.560.549.529.916.458
Mistral-7B-v0.3 L16.553.563.508.915.499
Llama-3-8B L16.530.514.473.897.542
Phi-2 L24.607.597.524.893.520
GPT-J-6B L14.565.508.532.868.538
Qwen2.5-0.5B-Inst L18.563.557.497.863.494
SmolLM2-1.7B L18.528.616.581.843.498
OPT-6.7B L24.607.585.561.829.466
GPT-Neo-1.3B L12.539.582.546.815.511
Qwen2.5-0.5B L18.529.544.568.789.503
TinyLlama-1.1B L16.631.641.543.778.441
Phi-1.5 L12.548.577.539.778.497
GPT-2 XL L24.521.566.513.756.457
SmolLM2-360M L24.582.550.513.753.530
not(Zhangetal.,2019;Yangetal.,2019). Ifidentityisnotinthevectors,theneverysystemthatfirstencodes
and then compares is looking in the wrong place for a large class of hard pairs.
We separate two pictures, on identical items, with a metric whose chance level is exactly0.5:
Geometry.Each sentence carries an identity. Cosine, a pair probe, or late fusion of the two vectors should
recover it.
Computation.Identityisarelationthenetworkcomputeswhenbothsentencesoccupythesamecontext. It
is then readable from the last token of a concatenated forward pass, andnotfrom any fixed or linear function
of the two separately encoded states.
The pictures are not rhetorical: late fusion gives a linear probe both vectors and no cross-attention, so if
geometryweremerely“misalignedwithcosine”itwouldclosethegap;ashuffled-partnerjointpasspreserves
format, language, and the marginal of sentence B while destroying pairing, so a probe reading a template
rather than the pairing would survive it.
Wefindcomputation,notgeometry,acrossmodernopenfamilies(Table1,Figure1). Onoverlap-matched
English PAWS-X, Llama 3 8B, Mistral 7B, and Qwen2.5 from 1.5B to 32B all yield joint confirm AUC
0.90–0.96atmid-depth, whileseparatecosine, aseparatepair probe, andlatefusionremainin 0.47–0.61;
partner shuffle returns to ≈0.50; dedicated encoders never leave the 0.55–0.70band. GPT-2 is not a
counterexample species: the same joint circuit appearsmore weakly, and only XLclearly pulls away ( 0.76).
Outside that stack the same protocol holds for Phi-2, GPT-J-6B, OPT-6.7B, GPT-Neo, Pythia, and SmolLM2
(joint 0.75–0.90),formean-pooledDeBERTa/RoBERTaencoders(DeBERTa-v3 0.92,nli-DeBERTa 0.94),
and for encoder–decoders (Flan-T5-base 0.95; T5-small / BART-base 0.64–0.66). A nonce-substitution
control keeps joint above late fusion with shuffle near chance on Qwen and on non-Qwen causals. Qwen base
models can alsosaythe answer (zero-shot verdict AUC 0.92–0.99from 3B to 32B); Llama 3 and Mistral
compute the relation at0.90while their verbal verdict stays≈0.61.
Two further findings sharpen the claim from “not geometry” to “one computed operator”. First, six
2

0.4 0.5 0.6 0.7 0.8 0.9 1.0
English confirm AUCQwen2.5-7B-Inst
Qwen2.5-14B
Qwen2.5-7B
Qwen2.5-32B
Qwen2.5-3B
Qwen2.5-1.5B-Inst
Qwen2.5-1.5B
Mistral-7B
Llama-3-8B
Phi-2
GPT-J-6B
Qwen2.5-0.5B-Inst
SmolLM2-1.7B
GPT-Neo-1.3B
Qwen2.5-0.5B
TinyLlama-1.1B
Phi-1.5
SmolLM2-360M
GPT-2 XL
Pythia-410M
GPT-2 medium
Pythia-1.4B
Pythia-1B
GPT-2 large
GPT-Neo-125M
GPT-2 small
OPT-6.7B
OPT-350M
OPT-1.3B
Pythia-160MJoint probe Late fusion Separate cosine Shuffled jointFigure 1:English confirm AUC at each model’s peak joint layer (overlap-matched PAWS-X). Joint sits above late
fusion, cosine, and shuffle, which stay near chance. Same numbers as Table 1.
independentlytrainedmodels(Qwen,Llama,Mistral)computemeasurablythesamerelation: theiritem-level
scores agree far beyond the labels, their errors co-occur, and the relation is symmetric and graded (§6).
Second, the operator is cheap and deployable: a 1.5B joint reader trained only on theconsensusof three
cross-family teachers over unlabelled text matches the 7B teacher on PAWS ( 0.959) and repairs a cosine FAQ
stack on wording twins (trapped top-10.47→0.88) without touching the retrieval index (§9).
The implication is not that embeddings should be discarded, but that they should not be asked to
certify identity: a stack that retrieves “the same question asked differently” is asking a bi-encoder to do a
cross-encoder’s job, and the shipped geometry of current LMs does not contain that job.
2 Related work
Sentence embeddings trained with siamese or contrastive objectives (Reimers and Gurevych, 2019; Gao
et al., 2021; Wang et al., 2022; Xiao et al., 2024) dominate retrieval benchmarks (Thakur et al., 2021;
Muennighoff et al., 2023). The retrieval literature already knows thatcross-encoders, which read the pair
jointly,outperformbi-encodersonreranking(NogueiraandCho,2019;KhattabandZaharia,2020). Thatfact
is usually treated as a compute/quality tradeoff. We show it is sharper: BGE rerankers reach the joint-probe
3

band on overlap-matched PAWS, while MS-MARCO and Jina rerankers do not—joint scoring alone does not
imply identity. We treat it as evidence aboutwhere the information is accessible: if every fixed or linear
reader over two frozen encodings stays near chance at any training size, while pair computations trained from
scratchoverthesameencodingsneedfortytimesthepairsofajoint-passprobetoapproachit,thegapisnota
scoring detail—identity becomes accessible only through a computation over the pair (the claim is about
accessibilitytotrainedreadouts,notinformation-theoreticabsence). Lateinteraction(KhattabandZaharia,
2020) still encodes independently, then scores a function of the two sequences; our late-fusion probe and
MaxSim are its sentence- and token-level analogues.
ProbingworkaskswhatislinearlyreadablefromLMstates(AlainandBengio,2017;Tenneyetal.,2019;
Hewitt and Manning, 2019; Pimentel et al., 2020; Belinkov and Glass, 2019). The linear representation
hypothesis (Park et al., 2024) concerns directionsinsidea model. Recent probes on modern open models
sharpen the same map: lexical identity is linearly strongest early and weakens with depth (Li and Subramani,
2026), while semantic and syntactic signals occupy partly separable mid-depth regimes (Acevedo et al.,
2026), and lexical overlap continues to distort independent-embedding similarity under paraphrase stress
tests (Rizwan et al., 2026). Our question is narrower and prior to that geometry: whether therelation“these
two sentences are paraphrases” is a function of two independent points at all.
PAWS and PAWS-X (Zhang et al., 2019; Yang et al., 2019) adversarialise bag-of-words overlap. Broader
paraphrase evaluationhas since movedbeyond a singleclassification set (Michailet al., 2025); westay with
overlap-matched PAWS-style identity because that is exactly where shipped bi-encoder geometry is asked to
certify sameness and fails. We go further andmatchoverlap between paraphrase and non-paraphrase cells, so
a scorer that only sees word overlap is at chance—the setting in which identity, if it exists as geometry, would
have to show itself.
3 Protocol
Data.WesamplePAWS-XEnglishandChinesepairsintodiscovery/development/confirmsplits,and
insideeachlanguage ×splitcellwematchlexicaloverlapofparaphrasesandnon-paraphrases(Englishconfirm
overlap gap typically <0.02), so the overlap-only AUC on English confirm is ≈0.47–0.55. Joint-probe
confirmcellscontain 150–500pairs( 600–2000discovery);the32Bcellisthesmallest( 150),anditsintervalis
reportedwhereveritisused. WereportAUCratherthanforced-choiceaccuracysothataconstantanisotropic
offset cannot move the number (Ethayarajh, 2019), and bootstrap 95% intervals on confirm; chance is exactly
0.5regardless of class balance.
Four pair readouts, one item set.For each frozen causal LM we score the same pairs four ways.
Separate cosine: last-tokenstatesfromindependentforwards,cosine.Separate probe: alogisticprobeon
(h1⊙h 2,|h 1−h2|)afteraPCAmapfitondiscovery.Late fusion: thesameprobeclasson [h1;h2]—both
vectorspresent,nomixinginsidetheLM.Joint probe: thetwosentencesareconcatenatedwithalanguage-
specific template that isnota yes/no question (A: .../B: ...); we probe the last token of that joint pass.
Probesarefitondiscovery,regularisationisselectedondevelopment,andscoresarereportedonlyonconfirm.
Table 1 lists each model at its layer of peak confirm joint AUC; locking the layer on development instead
picksthesamelayerfor26of30modelsandchangesconfirmjointAUCby 0.001onaverageandatmost
0.011(32B: L48,0.927instead of L58,0.938).
Shuffle control.Sentence B is replaced by another sentence from the same language and split: format,
language, and the marginal of B are preserved; pairing is destroyed. A probe that reads “two sentences
appeared” rather than “these two belong together” will survive shuffle.
4

Depth.We record last-token states at a fixed grid of relative depths, plus the final-norm input (prenorm,
captured by hook—HuggingFacehidden_states[-1]is already post-norm) and output (postnorm).
Generative channel, separately.A zero-shot yes/no verdict, length-normalised PMI, and conditional
surprisal are scored on a smaller overlap-matched draw; unlike the joint probe, these ask the model tospeak.
Encodersandrerankers.Nineteenpublicdensecheckpoints(BGE,E5,GTE,MiniLM,mpnet,multilingual
E5, E5-Mistral-7B) and six cross-encoders (three BGE rerankers, two MS-MARCO MiniLM rerankers, and
Jinareranker-v2)arescoredonadedicatedoverlap-matchedbankwith 𝑛=900confirmpairsperlanguage,
fixed sampling seeds, and English overlap matching (confirm overlap gap −0.002); bi-encoders and rerankers
score the same items.
Pre-registeredreadings. (P1)Computation: joint ≫latefusion,andlatefusionstaysneartheseparate-
probe band.(P2)Geometry: late fusion ≈joint, both well above chance.(P3)Shuffle must fall to chance.
(P4)Report the first depth whose confirm 95% interval excludes0.55.
4 Geometry does not carry identity
Dedicated encoders fail on overlap-matched confirm. The strongest English cosine is 0.701(BGE-large-zh);
thestrongestEnglish-trainedcheckpointismpnet-base( 0.653). Typicalretrievalencoderssitat 0.55–0.65;
scalingtoE5-Mistral-7Bdoesnotbreaktheband( 0.613). Alinearprobeonfrozenencodervectorsmoves
English BGE-small only from 0.588to0.607, and BGE-large-zh from 0.701to0.710: the information is not
“there but rotated”.
Noteverycross-encodercomputesidentity. BGE-reranker-large( 0.940),v2-m3( 0.902),andbase( 0.888)
align with the joint-probe band of §5. Jina reranker-v2 ( 0.639) and MS-MARCO MiniLM rerankers ( 0.545–
0.558)alsoreadthepairjointlybutstayinthedenseband—theyscorepassagerelevance,notPAWS-style
identity. Thegapisthereforenot“cross-encoderbeatsbi-encoder”;itisthatonlysomejointcomputations
implement the identity operator.
Independently encoded LM last-token states tell the same story (Table 1, columns Cos and Sep; cf.
Figure 1). Across Llama 3 8B, Mistral 7B, Qwen2.5 (0.5B–14B), GPT-2, GPT-Neo-125M, and Pythia,
Englishseparatecosineatthejoint-peaklayeris 0.52–0.61,andthebestpairprobeonthosetwovectorsis
0.51–0.62. Late fusion, the upperbound of “justhavingtwovectors”, remains 0.47–0.57for every modern
3–14B model we ran (Qwen2.5-14B 0.541): P2 fails, and the 14B cosine of 0.614, the highest geometric
number in the table, is still four tenths of AUC below the joint probe.
Identitymightinsteadliveinthetoken sequences: onthesameconfirmitems,ColBERT-styleMaxSim
of last-layer tokens reaches 0.62–0.71across Llama 3 8B, Mistral 7B, Phi-2, GPT-J-6B, OPT-6.7B, and
Qwen2.5 3B–32B (best: Qwen7B at 0.708, 95% CI[0.649,0.767] ); mean-pool cosine sits in 0.57–0.64, and
a two-layer MLP on the concatenated last-tokens, fit on the 1.2k discovery pairs, stays at chance ( 0.47–0.56).
Scaledoesnotmoveitintoafixedsimilarity: 14BMaxSim( 0.628)islowerthan7B’s,32BMaxSim 0.694is
0.24belowitsjoint 0.938,andPhi-2’sMaxSimequals7B’s( 0.701vsjoint 0.893): MaxSimsaturatesearly
as residual lexical matching.
A stronger independent-encoding reader is a small transformer with bidirectional cross-attention over the
twofrozentoken sequences—capacity, nonlinearity, and token-level interaction, but no joint LM pass; it fits
a relatedness control (true vs shuffled partner) at confirm 0.98–0.99on Qwen2.5-7B encodings, so it can
read the sequences. Trained for identity on the leak-filtered PAWS-X train split it stays at confirm 0.46–0.58
through 20kpairsoneverymodel;thefull 49kpairsliftitto 0.68–0.70(Llama38B,Mistral7B,Qwen2.5
3B/7B)and 0.87(Qwen14B).Alogisticlatefusionofthetwosentencevectorsstaysat 0.52–0.58onthesame
5

0.0 0.2 0.4 0.6 0.8 1.00.50.60.70.80.91.0English confirm AUC
Qwen2.5-32B
0.0 0.2 0.4 0.6 0.8 1.0
Qwen2.5-14B
0.0 0.2 0.4 0.6 0.8 1.0
Relative depth (postnorm at 1.05)0.50.60.70.80.91.0English confirm AUC
Llama-3-8B
0.0 0.2 0.4 0.6 0.8 1.0
Relative depth (postnorm at 1.05)
GPT-2 XLJoint Late fusion Cosine ShuffleFigure2:Depthofthefourreadouts. Jointrisesatmid-depth;theotherthreestaynearchance. Qwen14Bkeepsthe
signal into postnorm (0.950→0.940); Llama spends it (0.897→0.731).
49k pairs; an MLP on the pooled pair reaches 0.72–0.75. The token sequences carry the material; unlocking
it takes a pair computation and forty times the labels the LM’s own joint pass needs (0.90–0.95from1.2k).
Fine-tuningtheencoderitselfisadifferentquestion: afterleak-filteringPAWS-Xtrain(near-duplicates
and generation-template overlap with eval dropped; hyperparameters locked on overlap-matched validation),
MiniLM, BGE-small, and BGE-base reach overlap-matched confirm 0.871,0.884, and 0.930. Zero-shot
transferof thosecheckpointsto overlap-matchedQQPfallsrelativetothe untunedencoders( 0.813→0.701 ,
0.824→0.739 ,0.829→0.693 ),andSTS-BSpearmanfallsfrom 0.87–0.90to0.61–0.62. AMiniLMcross-
encoderonthesamesliceshitsPAWSconfirm 0.906onlyatfulltrainsize(chanceat 500pairs, 0.67at8k)
andstilltransferstooverlap-matchedQQPat 0.651. VectorscanbetrainedtocarryaPAWSidentitycriterion;
they do not, in this protocol, acquire a general meaning-identity geometry, and the fit taxes aboutness.
5 Identity appears when the pair is computed
Table1(andFigure1)reports,foreachmodel,thedepthofpeakEnglishjointconfirmAUC,togetherwith
late fusion, separate cosine, and shuffled joint at that same depth.
Joint is not late fusion, and shuffle kills it.For Llama, Mistral, and Qwen ≥1.5B, joint exceeds late
fusion by 0.37–0.46AUC (Qwen2.5-14B: 0.950versus 0.541); late fusion never leaves the encoder band, so
P1holdsandP2doesnot. Atthesamelayer,shuffleAUCis 0.46–0.54: theprobereadsthepairing,which
shuffle destroys, so P3 holds.
6

Not a decoder-only family effect.Table 1 is dominated by Llama-style causal models. We therefore re-ran
thesamefourreadoutsonthesameoverlap-matchedEnglishbank( 𝑛=200confirm)onothercausalstacks,
bidirectionalencoders, andencoder–decoders. EveryloadedmodelsatisfiesP1andP3: jointexceedslate
fusion,and partnershufflereturnsnearchance. Non-causalpeakssit inthemain-tableband—Flan-T5-base
0.953,nli-DeBERTa 0.940,DeBERTa-v3 0.921—withsmallerT5/BARTlowerbutsame-signed( 0.64–0.66).
Nonce substitution of content words preserves the gap.
The wrapper is not the result.Four concatenation templates— A:/B:, “Sentence 1/2”, a raw separator,
a blank line—yield joint confirm AUC 0.919–0.950on Qwen2.5-7B (shuffle 0.47–0.52),0.933–0.961on
14B,and 0.924–0.967on32B.Mistral7BandLlama38Batanoff-peaklayeronasmallerbankaremore
wrapper-sensitive ( 0.776–0.881and0.716–0.863), but every wrapper still beats late fusion and collapses
under partner shuffle.
QQP is not a PAWS artefact.On overlap-matched QQP (English confirm), joint AUC is 0.88–0.91for
every≥3B model we ran (Qwen2.5 3B–32B, Llama, Mistral, Phi-2; Qwen7B 0.910, 95% CI[0.877,0.943] ),
while late fusion rises to 0.72–0.77: duplicate questions shareaboutness, which PAWS’s word-scramble
matching forbids and separate vectors can see. To remove every single-sentence cue by construction we also
built ananchor-balancedQQPbank: eachanchor questionappears exactly twice, oncewith aduplicate and
once with an overlap-matched non-duplicate, so the first sentence carries no label information and lexical
overlap alone scores 0.52. There, with the layer locked on development, joint AUC is 0.80–0.83for all seven
models (Qwen2.5 3B/7B/14B/32B, Llama 3 8B, Mistral 7B, Phi-2; 95% CIs ≈±0.05), late fusion 0.55–0.60,
separate cosine 0.58–0.62, and partner shuffle 0.42–0.46(𝑛=300confirm). Geometry can notice that two
questions are about the same topic; certifying that they are the same question still takes the joint pass.
Thecomputeddirectiontransfers;thefittedgeometrydoesnot.TrainthelinearprobeonPAWSjoint
states,locklayerandregulariseronPAWSdev,thenapplythefrozenprobezero-shot: onoverlap-matched
QQP confirm it scores 0.72–0.75(Qwen7B / Llama / Mistral / Qwen14B); onChinesePAWS confirm—
differentlanguage, differentwrapper—it scores 0.745–0.818, nearlythein-domainChinesejointnumbers
(0.766–0.851); partner shuffle collapses it ( 0.48–0.53). The reverse direction is sharper: a probe trained only
onQQPreadsPAWSconfirmat 0.834/0.848/0.724/0.696(Qwen7B/Qwen14B/Llama/Mistral)without
everseeingaPAWSpair—theQwendirectionsaboveeveryindependent-encodingreadoutwecouldconstruct
on PAWS itself (best MaxSim 0.708), Llama and Mistral at its level. One linear direction in mid-depth joint
states carries the relation across datasets and across languages; the PAWS-tuned bi-encoder, by contrast,lost
QQPaccuracyrelativetoitsownuntunedcheckpoint. Transferispartial( 0.72–0.75vs0.88–0.91in-domain):
part of any probe is dataset-specific.
The relation is a distillable operator; the vectors cannot imitate it.If identity is a computed relation, it
shouldbehavelikeanoperator: teachabletoasmallerjointreader,andunteachabletoanyfunctionofthe
two independent vectors—even with the teacher’s own answers as supervision. We treat the mid-depth joint
directionofafrozenteacher(Qwen7B/Qwen14B/Mistral7B/Llama38B:PAWSconfirm 0.934/0.945
/0.898/0.895, zero-shot QQP 0.736/0.710/0.812/0.737) as a soft labeller. Ridge-fitting the teacher’s
scoresfromthesame model’sindependentlyencodedlast-tokens(cosine,concatenation,orproduct/difference
features) yields confirm AUC 0.46–0.62and correlation with the teacher between −0.04and0.20: the points
cannotreproducetherelationthesamenetworkcomputesoverthem. AQwen2.5-1.5Bjointstudentfitted
tothesamesoftscores—nogoldlabels—reachesPAWSconfirm 0.85–0.90undereveryteacher,thesame
band as a student trained on gold labels ( 0.86–0.90); a 3B student from the 7B teacher reaches 0.938(teacher
0.949)—theoperatorisalreadycheap,andteacherscoresareasgoodaslabels. DistillingonPAWSpairsonly,
7

the student’s zero-shot QQP lags the teacher ( 0.59–0.71); adding teacher-scoredunlabelledQQP pairs to
the distillation set (still no gold) closes most of the gap ( 0.71→0.79 under Mistral, teacher 0.81;0.69→0.73
under Llama; 0.63→0.67 under 7B). The operator distils from unlabelled pairs while the aboutness encoder
stays frozen; installing it in the vectors is the STS-taxed fit above.
The circuit is mid-depth.Peak joint is not the final residual (Figure 2). Llama and Mistral peak at layer 16
of 32; Qwen7B at layer 21 of 28; Qwen14B at layer 36 of 48; GPT-2 XL at layer 24 of 48. Postnorm joint is
systematicallylowerthanthemid-depthpeak,butfamiliesdifferinhowmuchtheyoverwrite: Llama 0.73
vs0.90(Qwen14B only 0.940vs0.950)—Qwen keeps what it computed, Llama spends it on next-token
prediction. Eitherway,theobjectusedasasentenceembedding—thefinalresidualofaseparatelyencoded
string—never held the relation, and a final-norm edit that wrecks next-token prediction does not create it:
permutingtheRMSNormgainonQwen2.5-7ByieldsKL 36againsttheuntouchedhead(top-1agreement
0.4%)whileseparatecosinestays 0.558versus 0.561;Llama,Mistral,14B,andthemutesmallmodelsall
stay in0.55–0.62under the same edits.
Chineseconfirmrepeatsthepatternatloweramplitudeformultilingualmodels(joint 0.77–0.85vslate
fusion0.54–0.60) and vanishes for GPT-2, which was not trained on Chinese.
6 One operator across independently trained models
Ifidentityisacomputedrelationratherthanacriterionfittedtoadataset,thenmodelstrainedbydifferent
organisations on different corpora should compute thesamerelation. Each of six models (Qwen2.5
1.5B/3B/7B/14B,Llama38B, Mistral7B)getsitsjointdirectionlocatedonce,onPAWSEnglishdiscovery,
then frozen; all six score the same held-out items.
The scores agree beyond the labels.On PAWS confirm, pairwise Spearman between models’ item scores
averages 0.855; restricted to cross-family pairs (Qwen vs Llama vs Mistral) it is 0.834. Agreement is not
explained by the binary label: rankingwithinthe paraphrases alone, or within the non-paraphrases alone,
cross-family agreement is still0.61–0.62—six models agree onhow much the sameeach pair is.
Errors co-occur; consensus adds nothing.Where one model errs, the others err: 𝑃(B wrong|A wrong)
exceeds the base rate by 2.9–5.7×on PAWS, and the seven items that all six models get wrong are three
orders of magnitude more frequent than independent errors would allow. Averaging the six z-scored outputs
does not beat the best single model ( 0.963vs0.970on PAWS; 0.773vs0.778on QQP): the models share the
residual as well as the signal.
Therelationissymmetricandgraded.Swappingthesentencesbarelymovesthescore( score(𝐴,𝐵) vs
score(𝐵,𝐴) Spearman 0.89–0.93on PAWS), and the forward–swap discrepancy carries no label information
(AUC 0.50–0.61). Applied zero-shot to SNLI, every one of the six directions orders PAWS paraphrase >
entailment>neutral>contradiction >PAWSnon-paraphrase,monotonicallyinthemean,withentailment-
vs-contradiction AUC0.79–0.98: a direction located with binary labels measuresdegree.
Unanimous disagreement with QQP gold.On overlap-matched QQP confirm, all six models contradict
thepublishedlabelon 18items( 57×theindependencerate). Sixofthem( 2.4%ofthebank)areconfident
(|𝑧|>1). ConsensusAUCagainstthosepublishedlabelsis 0.773. TransferofthefrozenPAWScoordinate
ispartialonoverlap-matchedMRPC( 0.48–0.61),andwhereittransferslessthesixmodelsalsoagreeless
(frozen-coordinate QQP: pairwise Spearman 0.34–0.84, error lift 1.3–2.3×): the invariant is the relation, and
each dataset’s coordinate onto it is only partly shared.
8

Theinvariantisasufficientteacher.Finallywetrainamodelontheinvariantitself: aQwen2.5-1.5Breader
(four mid-depth blocks unfrozen, linear head on the mid layer) regresses theconsensusof three cross-family
teachers(Qwen7B,Llama38B,Mistral7B;pairwisepoolcorrelation 0.73–0.82)on∼10kunlabelledEnglish
pairs. No gold pair label enters anywhere past locating each teacher’s coordinate. The student reaches PAWS
confirm 0.959—matching the 7B teacher’s own probe ( 0.956) and above every single-teacher distillation
of §5—QQP 0.774, andChinesePAWS 0.758zero-shot from English-only training text: what survives
distillation across families and languages is the computed quantity.
7 Scale, family, and a fossil that grows
Two scale stories are easy to confuse: computation as a Qwen idiosyncrasy, versus computation as the default
organisation of modern LMs, which older English LMs grow more slowly; Llama 3 and Mistral—not Qwen,
notinstruction-tuned—close thefirst. Instruction tuningisnot thecauseeither: Qwen2.5-1.5Bbasealready
reaches0.916(instruct0.947; 7B-Instruct0.961with late fusion still0.529).
InsideQwen2.5thejointcircuitisascalelawwithanearlyceiling,nota7Baccident: 0.789(0.5B)→
0.916(1.5B)→0.937 (3B)→0.941 (7B)→0.950 (14B)→0.938 (32B, 95% CI[0.897,0.969] ), with late
fusion at 32B still chance ( 0.493). Once the model is large enough to compute the pair, more parameters
compute it more cleanly, then stop gaining, and never move the relation into the vectors.
Theolderfamiliesrunthesamecurvemoreslowly(Table1,lowerblock): GPT-2climbs 0.665/0.688/0.652/0.756
(small/medium/large/XL) with late fusion pinned near chance—a long plateau, then a jump at XL. The
plateau-and-jumprepeats ineverylineage weran: TinyLlama-1.1BandPhi-1.5 alreadysplitjoint 0.778vs
late≈0.54andPhi-2isthatfamily’sjump( 0.893);SmolLM2goes 0.753→0.843 ;GPT-Neo 0.680→0.815
and GPT-J-6B 0.868; OPT sits flat at 0.658–0.674through 1.3B and jumps to 0.829at 6.7B; Pythia stalls at
0.611–0.727; none puts the relation in the vectors.
8 The mouth is optional
Ajointprobereadsastate. Auserofachatbotreadstokens. Onasmalleroverlap-matcheddraw( 𝑛=150–300
confirm), the two channels come apart by family.
Qwen base models cansaythe answer: zero-shot verdict AUC 0.915at 3B, 0.938at 7B (matching its
jointprobe 0.941),0.960at14B, 0.987at32B—whilelength-normalisedPMIstays 0.47–0.61throughout
(7B-Instruct verdict0.915: instruction tuning is not what opens the mouth).
Llama38BandMistral7Baretheotherhalfofthesplit: jointprobes 0.897and0.915,verdicts 0.614
and0.606—theycompute identityandthen refuseto utterit. Everyolder orsmaller familyweran doesthe
same(GPT-2verdicts 0.47–0.51againstjointupto 0.756;Phi 0.56–0.61against 0.778–0.893). Promptingis
not a substitute for the joint state: PMI does not recover it, and conditional surprisal sits at 0.55–0.76for
every model, unrelated to the joint probe (OPT-350M0.76against joint0.66).
9 What this is worth in the world
Anend-to-endtestwherethestakesarevisible.Denseretrievalcomparessentencevectorsbecauseit
ischeap;onwordingtwinsthatcomparisonisthewrongobject. WebuilttwofullyautomaticFAQstacks
(frozen BGE index, top-10 recall, then a certification step; abstention thresholds set on a disjoint dev split; no
human scoring anywhere). TheQuora bank(800 canonical QQP questions, 425 same-topic hard negatives,
1,500 distractors; queries are held-out duplicate partners) is the world where restated questions still share
words,sosurfaceoverlapislegitimateevidence: recall@10 0.994,cosinetop-1 0.862,in/outabstentionAUC
0.985. There the PAWS-located direction usedaloneis worse than cosine (top-1 0.427): a coordinate located
9

Table2:Trap-bankFAQsimulation. KBof 2,222entries( 1,289canonical, 418meaning-changedtwins, 515distractors);
1,289in-KBqueries( 375withatwinintheKB)and 300out-of-KBquerieswhosetwinisintheKB;arandom 30%
of queries sets thresholds, the rest are reported. All systems see the same cosine top-10. Conf. wrong =confidently
answered with the wrong entry; false out=answered an out-of-KB query at all.
System Top-1 Top-1 (trap) Conf. wrong False out
cosine.714.468.284.990
cosine→7B joint probe.840.831.100.453
cosine+7B joint probe.898.896.096.552
cosine→7B prompted verdict.903.903.088.547
cosine→1.5B consensus reader.844.842.109.502
cosine+1.5B consensus reader.900.878.098.606
on word-scramble pairs does not rank restated questions, whose twins differ in topic granularity rather than
wordorder;locatedon500in-domainpairsitcertifiesat 0.800,andfusedwithcosineeveryoperatormatches
cosine’stop-1whileloweringfalseanswersonout-of-KBqueries( 0.125→0.094 ): theoperatorcertifiesa
candidate list, it does not replace recall. Thetrap bankis the same stack over PAWS sentences (KB of 2,222
entries): 375of1,289canonicalentrieshaveaword-swaptwininsidetheKB,andeachof300out-of-KB
queries has its meaning-changed twininthe KB. Cosine collapses exactly there: top-1 0.468on trapped
queries, and it confidently answers 99%of out-of-KB queries—always with the wrong twin. The frozen
PAWS direction of §5, with zero in-domain labels, certifies the same top-10 at 0.83on trapped queries; fused
withcosineitreaches 0.90overall,matchingaprompted7Bverdict( 0.903)—achannelonlytheQwenfamily
can speak (§8)—while confident wrong answers fall from 0.284to0.096. The consensus-trained 1.5B reader
of §6 reproduces this end to end (0.844alone,0.900fused) with no gold labels and no 7B at inference.
The practical reading: retrieve with embeddings; certify identity with a joint pass. The operator is cheap
to own, and the aboutness index stays frozen (BGE-base STS-B Spearman stays0.895).
10 Limitations
Scope is PAWS-style paraphrase under overlap matching. Independent-encoding readers trail the joint probe
even at 49k training pairs; cross-architecture amplitude varies while the joint-versus-late sign does not;
Llama 3 / Mistral compute the relation (0.90) without verbalising it (0.61).
11 Conclusion
Meaning identity on overlap-matched PAWS-X is a joint-pass computation, not a property of frozen
independentembeddings: thegapholdsacrosscausal,encoder,andencoder–decoderfamilies,survivesnonce
substitution, is shared across training houses, and distils into a cheap reader that repairs a cosine FAQ stack.
Code, scripts, and result dumps will be released in a public repository in a subsequent update of this
preprint.
References
Santiago Acevedo, Alessandro Laio, and Marco Baroni. Differential syntactic and semantic encoding in
LLMs. InICML, 2026.
10

Guillaume Alain and Yoshua Bengio. Understanding intermediate layers using linear classifier probes.ICLR
Workshop, 2017.
YonatanBelinkovandJames Glass. Analysis methodsinneural language processing: Asurvey.TACL,7:
49–72, 2019.
Daniel Cer, Mona Diab, Eneko Agirre, Iñigo Lopez-Gazpio, and Lucia Specia. SemEval-2017 task 1:
Semantic textual similarity multilingual and crosslingual focused evaluation. InSemEval, 2017.
Kawin Ethayarajh. How contextual are contextualized word representations? InEMNLP, 2019.
TianyuGao,XingchengYao,andDanqiChen. SimCSE:Simplecontrastivelearningofsentenceembeddings.
InEMNLP, 2021.
YunfanGao,YunXiong,XinyuGao,KangxiangJia,JinliuPan,YuxiBi,YiDai,JiaweiSun,MengWang,
andHaofenWang. Retrieval-augmentedgenerationforlargelanguagemodels: Asurvey.arXiv preprint
arXiv:2312.10997, 2024.
John Hewitt and Christopher D. Manning. A structural probe for finding syntax in word representations.
NAACL, 2019.
OmarKhattabandMateiZaharia. ColBERT:Efficientandeffectivepassagesearchviacontextualizedlate
interaction over BERT. InSIGIR, 2020.
MichaelLiandNishantSubramani. Modelinternalsleuthing: Findinglexicalidentityandinflectionalfeatures
in modern language models. InACL, 2026.
AndrianosMichail,SimonClematide,andJuriOpitz. PARAPHRASUS:Acomprehensivebenchmarkfor
evaluating paraphrase detection models. InCOLING, pages 8749–8762, 2025.
Niklas Muennighoff, Nouamane Tazi, Loïc Magne, and Nils Reimers. MTEB: Massive text embedding
benchmark. InEACL, 2023.
Rodrigo Nogueira and Kyunghyun Cho. Passage re-ranking with BERT. InarXiv:1901.04085, 2019.
KihoPark,YoJoongChoe,andVictorVeitch. Thelinearrepresentationhypothesisandthegeometryoflarge
language models.ICML, 2024.
TiagoPimentel,JosefValvoda,RowanHallMaudslay,RanZmigrod,AdinaWilliams,andRyanCotterell.
Information-theoretic probing for linguistic structure. InACL, 2020.
Nils Reimers and Iryna Gurevych. Sentence-BERT: Sentence embeddings using siamese BERT-networks. In
EMNLP, 2019.
Hammad Rizwan, Muhammad Umair Haider, Nishant Subramani, Mona T. Diab, A. B. Siddique, and Hassan
Sajjad. On the persistent effects of lexicality in large language models.arXiv preprint arXiv:2606.02750,
2026.
Ian Tenney, Dipanjan Das, and Ellie Pavlick. BERT rediscovers the classical NLP pipeline. InACL, 2019.
Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. BEIR: A
heterogeneousbenchmarkforzero-shotevaluationofinformationretrievalmodels. InNeurIPS Datasets
and Benchmarks, 2021.
11

Liang Wang, Nan Yang, Xiaolong Huang, BinxingJiao, Linjun Yang, Daxin Jiang, RanganMajumder, and
Furu Wei. Text embeddings by weakly-supervised contrastive pre-training.arXiv:2212.03533, 2022.
Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muennighoff, Defu Lian, and Jian-Yun Nie. C-Pack: Packed
resourcesforgeneralChineseembeddings. InProceedings of the 47th International ACM SIGIR Conference
on Research and Development in Information Retrieval, 2024.
Yinfei Yang, Yuan Zhang, Chris Tar, and Jason Baldridge. PAWS-X: A cross-lingual adversarial dataset for
paraphrase identification. InEMNLP, 2019.
Yuan Zhang, Jason Baldridge, and Luheng He. PAWS: Paraphrase adversaries from word scrambling. In
NAACL, 2019.
12