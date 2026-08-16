# RAG-Audio: Retrieval-Augmented Generation for Faithful Brain-to-Audio Reconstruction

**Authors**: Ambuj Mehrish, Sebastiano Vascon

**Published**: 2026-08-10 09:10:15

**PDF URL**: [https://arxiv.org/pdf/2608.09331v1](https://arxiv.org/pdf/2608.09331v1)

## Abstract
Brain-to-audio reconstruction is limited by \emph{prior domination}: when a pretrained generator is conditioned on a weak neural signal, it produces realistic but stimulus-inaccurate audio. We introduce RAG-Audio, which decodes fMRI into a semantic audio embedding, retrieves a matching real-audio exemplar, and initializes the frozen generator's sampling trajectory from that exemplar while retaining the decoded embedding as conditioning. On Brain2Music, RAG-Audio improves 10-way stimulus identification from $0.14$--$0.18$ for direct generation, near the $0.10$ chance level, to $0.40$--$0.43$, comparable to retrieval. It also reduces Fréchet Audio Distance by roughly an order of magnitude, from $13.49$ to $1.25$ for AudioLDM. RAG-Audio approaches nearest-neighbor retrieval in identification while remaining generative; its higher FAD is expected because retrieval directly replays real audio. An autoregressive negative control, which lacks an initializable latent trajectory, shows no comparable gain, attributing the improvement to trajectory initialization. These results suggest that retrieval-guided initialization can mitigate prior domination in brain-to-audio generation.

## Full Text


<!-- PDF content starts -->

RAG-Audio: Retrieval-Augmented Generation for Faithful
Brain-to-Audio Reconstruction
Ambuj Mehrish
CVML Lab
Ca’ Foscari University of Venice
ambuj.mehrish@unive.itSebastiano Vascon
CVML Lab
Ca’ Foscari University of Venice
sebastiano.vascon@unive.it
Abstract
Brain-to-audioreconstructionislimitedbypriordomination:whenapretrainedgenerator
is conditioned on a weak neural signal, it produces realistic but stimulus-inaccurate audio.
We introduce RAG-Audio, which decodes fMRI into a semantic audio embedding, retrieves
a matching real-audio exemplar, and initializes the frozen generator’s sampling trajectory
fromthatexemplarwhileretainingthedecodedembeddingasconditioning.OnBrain2Music,
RAG-Audioimproves10-waystimulusidentificationfrom 0.14–0.18fordirectgeneration,
nearthe 0.10chancelevel,to 0.40–0.43,comparabletoretrieval.ItalsoreducesFréchetAudio
Distance by roughly an order of magnitude, from 13.49to1.25for AudioLDM. RAG-Audio
approaches nearest-neighbor retrieval in identification while remaining generative; its higher
FAD is expected because retrieval directly replays real audio. An autoregressive negative
control, which lacks an initializable latent trajectory, shows no comparable gain, attributing
the improvement to trajectory initialization. These results suggest that retrieval-guided
initialization can mitigate prior domination in brain-to-audio generation.
1 Introduction
Reconstructingperceptualexperiencefrombrainactivityisademandingchallengeformoderngenerative
models. In vision, recent methods can recover recognizable images from functional magnetic resonance
imaging (fMRI), even though fMRI measures stimulus-related changes in blood oxygenation rather than
neuralactivitydirectly(Nishimotoetal.,2011;TakagiandNishimoto,2023;Scottietal.,2023;Huoetal.,
2024).Audioreconstructionhasreceivedfarlessattention,especiallyformusic.Mostbrain-to-audio
systems use a two-stage pipeline: they first map the fMRI response to a semantic audio embedding
and then use this embedding to guide a pretrained audio generator (Denk et al., 2023; Ciferri et al.,
2025; Park et al., 2023). Brain2Music (Denk et al., 2023), for example, predicts a joint music–language
embedding (Huang et al., 2022) from fMRI and supplies it to a text-to-audio model. This design appears
reasonable because it separates neural decoding from waveform generation. Yet it also assumes that the
generator will preserve whatever stimulus information survives the decoding stage.
Ourresultssuggestthatthisassumptionoftenfails.Werefertothisfailureaspriordomination:when
a powerful pretrained generator receives a weak or noisy brain-derived condition, its learned distribution
overplausibleaudiocanoutweightheevidencecarriedbythatcondition.Thisistheconditioning-strength
1
arXiv:2608.09331v1  [cs.SD]  10 Aug 2026

RAG-Audio: Faithful Brain-to-Audio Reconstruction Preprint
Figure 1: Overview of RAG-Audio and two baselines.Given a brain-decoded CLAP embedding ˆ𝑧, direct
generation samples from noise and may be dominated by the generator prior; retrieval returns the nearest training
clip but is non-generative. RAG-Audio initializes the frozen generator from the exemplar latent 𝑧0perturbed to an
intermediate time, 𝑧𝑡0, preserving stimulus-related structure while the anchoring strength 𝑠(with𝑡0=𝑠𝑇) controls
the trade-off between faithfulness and novelty.
problemfamiliarfromdiffusionguidance,whereaguidanceweightbalancesaconditionalsignalagainst
theunconditionalprior(DhariwalandNichol,2021;HoandSalimans,2022);herethebrain-derived
condition is fixed and weak, so no reweighting of a text prompt can recover it. The generated audio may
sound realistic but still fail to match the music the participant actually heard. This gap is clear on the
public Brain2Music dataset (Denk et al., 2023), which contains fMRI recordings from five subjects
listeningtoGTZANmusicclips(TzanetakisandCook,2002).Inourre-implementation,decodingfMRI
into a CLAP embedding identifies the heard clip with 0.43 accuracy in a 10-way test, well above the
0.10chance level. However, when the same embedding is passed to AudioLDM (Liu et al., 2023a),
TangoFlux(Hungetal.,2024),orMusicGen(Copetetal.,2023),identificationaccuracyforthegenerated
audio falls to 0.14–0.18, including in the published Brain2Music generation setting (Denk et al., 2023).
Theconditioningsignalthereforecontainsrecoverablestimulusinformation,butdirectgenerationretains
littleofit.Brain-to-imagestudieshavereportedarelatedqualitativepattern,withrealisticreconstructions
that drift from the target (Takagi and Nishimoto, 2023; Scotti et al., 2024) (Fig. 7), although it remains
unclear whether the same behavior can be measured systematically in audio or whether it depends on a
particular generator.
We investigate a targeted way to reduce prior domination (Figure 1). Instead of asking the generator
toreconstructthestimulusfromthedecodedembeddingalone,wefirstretrievethetrainingaudioclip
whose CLAP embedding is closest to the decoded representation. We then use this clip to initialize the
generator at an intermediate point in its sampling trajectory. For a diffusion-based generator such as
AudioLDM (Liu et al., 2023a), we follow the SDEdit principle (Meng et al.) by adding partial noise
to the exemplar latent before reverse diffusion. For a flow-based generator such as TangoFlux (Hung
etal.,2024),weusethecorrespondingdeterministicinterpolationalongtherectified-flowtrajectory(Liu
et al., 2023b; Lipman et al.; Rout et al., 2024; Kulikov et al., 2025). The generator can therefore
refine the retrieved audio rather than synthesize entirely from its prior. The anchoring strength controls
howcloselytheoutputfollowstheexemplar:weakerperturbationpreservesmoreofit,whilestronger
2

RAG-Audio: Faithful Brain-to-Audio Reconstruction Preprint
perturbation allows greater variation and novelty. This approach differs from plain retrieval because the
final output does not simply copy the stored clip. It also differs from direct brain-conditioned generation
because sampling begins from stimulus-related audio structure. We therefore view exemplar anchoring
as a tunable trade-off between faithfulness and novelty, rather than as a replacement for retrieval or
unconstrained generation (Lewis et al., 2020; Blattmann et al., 2022; Yuan et al., 2024).
These design choices yield a consistent improvement: anchoring recovers identification to the level
of a strong non-generative retrieval baseline (Ciferri et al., 2025) while keeping the output newly
generated,andanautoregressivenegativecontrol(MusicGen(Copetetal.,2023)),whichexposesno
initializabletrajectory,showsnocomparablegainisolatingtrajectoryinitializationratherthanretrieval
as the mechanism. Section 6 reports the full results. In summary, our contributions are:
•Weidentifyandquantifypriordomination,afailuremodethedecode-then-generaterecipeinherits
and that sharpens as pretrained generators grow stronger.
•We introduce exemplar anchoring, a retrieval-augmented scheme that initializes the generator
partway alongitssampling trajectoryfroma retrievedreal-audio exemplar,recovering retrieval-
level identification while keeping generation genuinely generative and reducing FAD by ∼10.8×
for AudioLDM and∼3.3×for TangoFlux.
•WeusetheautoregressivegeneratorMusicGen(Copetetal.,2023)asanegativecontrol,attributing
the recovery to latent-trajectory initialization rather than to retrieval alone.
•We provide an anatomical validity check that localizes the decoder signal to auditory cortex.
2 Related Work
Brain-to-audio reconstruction.Perceived sound has been decoded from ECoG, MEG, EEG, and
fMRI,spanningspeech,music,andsemanticlanguage(Anumanchipallietal.,2019;Metzgeretal.,2023;
Bellieretal.,2023;Défossezetal.,2023;Postolacheetal.,2025;Tangetal.,2023).ForfMRImusic,
mostmethodsdecodeanaudiorepresentationandconditionapretrainedgenerator(Denketal.,2023;
Parketal.,2023;Liuetal.,2024).Existingsystemsthusrangefromnearest-neighbourretrieval(Ferrante
et al., 2024) to unconstrained generation, including prior-guided diffusion (Ciferri et al., 2025). We
re-implement both endpoints in a common evaluation framework and bridge them through exemplar
anchoring.
Prior domination and cortical localization.Brain-to-image reconstruction exhibits a closely related
tension. Latent-diffusion decoders reconstruct images from fMRI on the Natural Scenes Dataset (Allen
etal.,2022)bymappingactivityintoagenerator’sconditioningspace(TakagiandNishimoto,2023;Chen
et al., 2023), and the MindEye family separates a retrieval pathway from a diffusion-prior reconstruction
pathway(Scottietal.,2023,2024);arecurringobservationisthatoutputslookrealisticyetdriftfrom
the stimulus when the brain signal is weak relative to the generator prior. We term this regime prior
dominationand,unlikethelargelyqualitativeimage-domainreports,quantifyitforaudioandshowit
holdsacrossgeneratorfamilies.Ourdecoder-localizationanalysisfollowsvoxelwisemodelingofauditory
cortex, where deep-network features predict responses along the auditory hierarchy and music-selective
populations occupysuperior temporal cortex (Kellet al., 2018;Norman-Haignere et al., 2015;Tuckute
et al., 2023).
3

RAG-Audio: Faithful Brain-to-Audio Reconstruction Preprint
Audio generators and embeddings.Brain-to-imagereconstructionfacesasimilartrade-off.Diffusion-
based fMRI decoders can produce realistic images but often drift from the stimulus when the generator
prioroverwhelmsweakneuralevidence(Allenetal.,2022;TakagiandNishimoto,2023;Chenetal.,
2023;Scottietal.,2023,2024).Wequantifythispriordominationforaudioacrossgeneratorfamilies
and localize decoder contributions using established models of auditory-cortex organization (Kell et al.,
2018; Norman-Haignere et al., 2015; Tuckute et al., 2023).
Editing and retrieval-augmented generation.SDEdit starts from a guide signal, adds noise up to a
chosenintermediatediffusionstep,andthendenoisesfromthatpointtoproduceanewsample(Meng
et al.). The noise level controls the balance between preserving the guide and allowing the generator
to introduce new content (Ho et al., 2020; Song et al., 2020b; Dhariwal and Nichol, 2021; Ho and
Salimans,2022; Rombachet al.,2022).Related methodsguide frozendiffusionmodels withreference
images (Choi et al., 2021; Lugmayr et al., 2022), while retrieval-augmented generators condition on
externalexamplesacrosslanguage,image,andaudiosynthesis(Khandelwaletal.,2019;Lewisetal.,
2020; Borgeaud et al., 2022; Blattmann et al., 2022; Yuan et al., 2024). RAG-Audio combines these
ideasbyinitializinggenerationfromaretrievedaudioexemplar,preservingstimulus-relevantcontent
without copying the retrieved clip verbatim.
Figure 2: The RAG-Audio pipeline.The decoder 𝑓𝜃maps an fMRI response 𝑥to a CLAP embedding ˆ𝑧; the
nearest exemplar 𝑎∗is retrieved and encoded to 𝑧0, perturbed to an intermediate time 𝑡0=𝑠𝑇(𝑧𝑡0), and the frozen
generator denoises from𝑧 𝑡0conditioned onˆ𝑧to produceˆ𝑎. The strength𝑠trades faithfulness against novelty.
3 Problem Setup
Let𝑎denote an audio clip heard by a subject while functional magnetic resonance imaging (fMRI)
records the corresponding brain response 𝑥. Given a paired dataset D=(𝑥𝑖,𝑎𝑖)𝑀
𝑖=1, the goal is to
reconstructthestimulusassociatedwithaheld-outscan.SincefMRIisanindirectandtemporallycoarse
measurement of neural activity, we formulate reconstruction in a semantic audio space rather than at the
waveform level. Each stimulus is represented using CLAP (Wu et al., 2023):
4

RAG-Audio: Faithful Brain-to-Audio Reconstruction Preprint
𝑧𝑎=CLAP(𝑎)∈R512.(1)
Following the standard decode-then-generate pipeline (Denk et al., 2023), a learned decoder 𝑓maps
the fMRI response to a predicted semantic embedding:
ˆ𝑧=𝑓(𝑥).(2)
A frozen pretrained audio generator𝐺then synthesizes the reconstruction:
ˆ𝑎=𝐺(ˆ𝑧).(3)
We defer the architectures and training procedures for𝑓and𝐺to the Method section.
Wemeasuresemanticfaithfulnessusing 𝑁-wayidentification.Let 𝑎+bethetruestimulus,andlet
A−=𝑎−
1,...,𝑎−
𝑁−1contain𝑁−1distractors. Writing 𝑠(·,·)for cosine similarity, the identification
accuracy is
acc𝑁=E"
I 
𝑠 CLAP(ˆ𝑎),CLAP(𝑎+)
>max
𝑎−∈A−𝑠(CLAP(ˆ𝑎),CLAP(𝑎−))!#
.(4)
Chanceperformanceis 1/𝑁.Thismetricevaluateswhetherthereconstructionpreservesthesemantic
content of the heard stimulus; it does not measure sample-accurate waveform recovery.
Thetwostagescancontributeunequallytothefinaloutput.Thegenerator 𝐺representsastrongprior
𝑝(𝑎)learnedfromlargeaudiocorpora,whereasthebrain-derivedcondition ˆ𝑧iscomparativelyweakand
noisy. We useprior dominationto describe the regime in which 𝐺(ˆ𝑧)remains acoustically plausible but
depends only weakly on ˆ𝑧, behaving approximately like an unconditional sample from the generator
prior.Generationmaythereforediscardstimulusinformationretainedbythedecodedembedding.Our
objectiveistopreservetherealismsuppliedby 𝐺whileimprovingthesemanticfaithfulnessofitsoutput.
4 Method
We proposeRAG-Audio, a retrieval-augmented pipeline for brain-to-audio reconstruction (Fig. 2).
RAG-Audio first decodes an fMRI response into a CLAP embedding, retrieves the nearest real training
exemplar from a memory bank, and then initializes a frozen generator’s sampling trajectory from
that exemplar at an intermediate time. For AudioLDM, this follows the SDEdit principle of partially
noising theexemplar latent beforereverse diffusion (Menget al.), whileTangoFlux (Hung etal., 2024)
appliesthecorrespondingdeterministicinterpolationalongitsrectified-flowtrajectory.Theretrieved
clipsuppliesplausibleaudiostructurethatcansurvivethegenerator’sprior,turningpriordomination
into a controllable trade-off between faithfulness to the retrieved content and generative novelty.
4.1 fMRI-to-CLAP Decoder
For each stimulus audio clip 𝑎, we use its CLAP representation (Wu et al., 2023) as the decoding target,
with𝑧𝑎=CLAP(𝑎)∈R512.WeapplyrigidmotioncorrectionwithANTsPy(Avantsetal.,2011)and
5

RAG-Audio: Faithful Brain-to-Audio Reconstruction Preprint
retaineachsubject’sdatainnativespace.Toaccountforthehaemodynamicresponse,weusealagof
𝑙=3repetition times (TRs), corresponding to4.5seconds, and a temporal window of𝑤=8TRs.
WerankvoxelsaccordingtotheirassociationwiththeCLAPtarget.Forvoxel 𝑣,let𝑥𝑣denoteits
temporal response and let𝑧(𝑑)denote the𝑑-th CLAP dimension. We define the relevance score as
𝑠𝑣=max
𝑑∈{1,...,512}corr(𝑥𝑣,𝑧(𝑑)).(5)
Weretainthe 𝐾=2000 highest-scoringvoxels.Thesamescoresdefinethelocalizationmapanalyzed
in Figure 3.
Thedecoder 𝑓𝜃:R2000→R512mapstheselectedfMRIfeaturestoapredictedembedding ˆ𝑧=𝑓𝜃(𝑥).
Rather than minimizing dimension-wise reconstruction error, we train 𝑓𝜃with InfoNCE (Oord et al.,
2018)sothatdecodedscansremainclosetotheirpairedaudioembeddingsandseparatedfromother
clips in the batch. For a batch of size𝐵and temperature𝜏=0.07, the objective is
L=−1
𝐵𝐵∑︁
𝑖=1logexp cos(𝑓𝜃(𝑥𝑖),𝑧𝑎𝑖)/𝜏
Í𝐵
𝑗=1exp cos(𝑓𝜃(𝑥𝑖),𝑧𝑎𝑗)/𝜏.(6)
Thisobjectivedirectlyoptimizestheretrievalgeometryusedforsemanticidentification.Wecompare
it with ridge, linear, and MLP decoders in Figure 8; the contrastive decoder is selected because its
training criterion matches the downstream identification objective.
4.2 Memory Bank and Retrieval
We construct a memory bankB={𝑧 𝑎:𝑎∈A train}from CLAP embeddings of real training-set audio,
whereAtraindenotes the training clips. Test stimuli are excluded. Each bank entry retains a reference to
its corresponding audio clip.
Givenaheld-outfMRIresponse 𝑥,thedecoderproduces ˆ𝑧=𝑓𝜃(𝑥).Weretrievethetrainingexemplar
whose CLAP embedding has the largest cosine similarity toˆ𝑧:
𝑎∗=arg max
𝑎∈Bcos(ˆ𝑧,𝑧𝑎).(7)
Here,𝑎∈Bindexes the audio clip associated with a bank embedding. Returning 𝑎∗directly defines
theRetrievalbaseline, corresponding to the published linear-contrastive fMRI-to-CLAP retrieval
setting(Ferranteetal., 2024).Thisbaselineprovidesastrongnon-generative reference:itcanpreserve
stimulus-level content, but its output is a verbatim training clip. RAG-Audio instead treats 𝑎∗as an
anchor from which to generate a new reconstruction.
4.3 Exemplar-Anchored Generation via Intermediate-Time Initialization
Let𝐸denote the encoder of a frozen latent generative model. We first map the retrieved exemplar to the
generator latent, 𝑧0=𝐸(𝑎∗). A anchoring strength 𝑠∈(0,1] determines the intermediate initialization
time𝑡 0=𝑠𝑇, where𝑇is the full denoising horizon.
For a latent-diffusion generator such as AudioLDM (Liu et al., 2023a), the anchoring strength
𝑠∈(0,1]determines the diffusion initialization step
𝑡0=⌊𝑠𝑇⌋,(8)
6

RAG-Audio: Faithful Brain-to-Audio Reconstruction Preprint
where𝑇denotesthetotalnumberofdiffusionsteps.FollowingtheSDEditprinciple(Mengetal.),we
partially corrupt the exemplar latent 𝑧0according to the forward diffusion process and then run the
reverse process from𝑡 0, conditioned on the decoded embeddingˆ𝑧:
𝑧𝑡0=√︁
¯𝛼𝑡0𝑧0+√︁
1−¯𝛼𝑡0𝜖, 𝜖∼N(0,𝐼).(9)
Flow based generator such as TangoFlux (Hung et al., 2024) follows a normalized rectified-flow
trajectory. For this model, the anchoring strength directly determines the initialization time:
𝜏0=𝑠.(10)
We initialize the trajectory by deterministically interpolating between the exemplar latent and Gaussian
noise (Song et al., 2020a):
𝑧𝜏0=(1−𝜏 0)𝑧0+𝜏0𝜖, 𝜖∼N(0,𝐼),(11)
and then integrate the learned flow from𝜏 0to0.
Thestrength 𝑠controlshowmuchexemplarstructureremainsatthestartofgeneration.As 𝑠→0,the
outputapproaches 𝑎∗,favoringfaithfulnessatthecostofnovelty.As 𝑠→1,theinitializationincreasingly
erases the exemplar and approaches direct generation from the model prior. Sweeping 𝑠therefore traces
the faithfulness–novelty continuum shown in Figure 4; empirically, we consider the working range
𝑠≈0.25–0.40.
This mechanism requires a continuouslatent trajectory that can be initialized through Equation (9).
Autoregressive MusicGen (Copet et al., 2023) does not provide such a latent. Its retrieved exemplar can
enteronlythroughmelodyconditioning,whichremainssubjecttotheautoregressivetokenprior.We
usethiscaseasanegativecontrolintheResultssectiontodistinguishtheeffectofintermediate-time
trajectory initialization from the mere availability of an exemplar.
Table 1:Main comparison on Brain2Music ( 𝑛=300;10-way chance 0.10), all methods run in a single harness.
Exemplar anchoring lifts identification from 0.14–0.18(direct) to the retrieval level ( 0.40–0.43) and cuts FAD by
up to∼10×for AudioLDM and TangoFlux, but not for autoregressive MusicGen.
Method Type Gen 10-way↑FAD↓
Published baselines (re-implemented)
R&B (Ferrante et al., 2024) retrieval no 0.400.89
MusicGen (Copet et al., 2023)(decode→gen) direct yes 0.18 8.06
Direct generation, prior-only (ours)
AudioLDM (Liu et al., 2023a) direct yes 0.14 13.5
TangoFlux (Hung et al., 2024) direct yes 0.14 7.89
MusicGen (Copet et al., 2023) direct yes 0.18 8.06
RAG-Audio (ours)
AudioLDM + RAG retr.-aug. yes0.431.25
TangoFlux + RAG retr.-aug. yes 0.40 2.36
MusicGen + RAG retr.-aug. yes 0.20 6.21
5 Experimental Setup
Dataset.WeevaluateonBrain2Music,releasedasOpenNeurods003720(Nakaietal.,2022;Denk
et al., 2023). The dataset contains fMRI recordings from five subjects listening to 15-second music clips
7

RAG-Audio: Faithful Brain-to-Audio Reconstruction Preprint
Figure 3:Decoderlocalization:voxelrelevancescores(red)overlaptheHarvard–Oxfordauditory-cortexmask
(blue), with78%of top voxels in bilateral superior temporal gyrus.
drawn from the ten GTZAN genres (Tzanetakis and Cook, 2002), with a repetition time of TR=1.5,s .
Wefollowtheofficialsubject-wisesplitof480trainingclipsand60held-outtestclips.Allmetricsare
averaged across subjects, corresponding to a pooled set of𝑛=300test generations.
Audio generators.Weconsiderthreepretrainedgenerators:AudioLDM( cvssp/audioldm-s-full-v2 )
(Liu et al., 2023a), TangoFlux ( declare-lab/TangoFlux ) (Hung et al., 2024), and MusicGen
(facebook/musicgen-melody ) (Copet et al., 2023). All generator parameters remain frozen. Au-
dioLDM and TangoFlux use latent-diffusion and rectified-flow formulations, respectively, and therefore
supportintermediate-timetrajectoryinitialization.MusicGenisautoregressiveandservesasamechanism
control for which exemplar initialization is unavailable.
Comparison arms.Wecomparethreereconstructionstrategies.Directperformsbrain-conditioned
generation using only the pretrained generative prior.Retrievalreturns the nearest real clip from the
training bank (Ferrante et al., 2024).RAG-Audioapplies our intermediate-time exemplar-anchoring
procedure. All methods, including published baselines, are run within a single evaluation harness using
the same test scans, candidate sets, checkpoints, and metric implementations. Their reported values are
therefore directly comparable.
Faithfulness.Semantic faithfulness ismeasured by the 𝑁-way identification of Eq. 4, computedin
CLAPspaceusingtheLAIONcheckpoint laion/clap-htsat-unfused .Wereport𝑁∈{2,5,10,50} ,
with corresponding chance levels of0.50,0.20,0.10, and0.02.
Realism and novelty.AudiorealismisevaluatedusingFADcomputedwithVGGishfeatures(Kilgour
et al., 2019; Hershey et al., 2017), together with FD and KL divergence computed from the PANNs
Cnn14 audio tagger (Kong et al., 2020). Lower values indicate better agreement with real audio. We
quantifynoveltyasoneminusthecosinesimilaritybetweeneachgeneratedclipanditsnearestclipin
the training bank. Plain retrieval has novelty near zero by construction because it returns a training clip
verbatim. We additionally report pairwise diversity among generated samples.
Implementation.We apply rigid-motion correction with ANTsPy (Avants et al., 2011), select the top
2000voxels,sweeptheanchoringstrengthover {0.2,...,0.5} ,andtrainthecontrastivedecoderwith
8

RAG-Audio: Faithful Brain-to-Audio Reconstruction Preprint
Figure 4:Ten-wayidentificationversusanchoringstrength 𝑠:low𝑠staysneartheretrievalbound,high 𝑠decaysto
the0.10chance level; working range𝑠≈0.25–0.40.
temperature 𝜏=0.07.Completepreprocessing,optimization,andgenerationsettingsareprovidedin
Appendix C.
6 Results
Decoder PerformanceThe contrastive fMRI-to-CLAP decoder achieves the strongest stimulus
identificationdespitethelowestdimension-wise embeddingcorrelation.Averagedacrossfivesubjects,
itreaches 0.43accuracyin10-wayidentification(chance 0.10)and 0.80in2-wayidentification,with
every subject above 10-way chance. It outperforms ridge, linear, and MLP decoders even though its
Pearson correlation is only 0.20, versus 0.68–0.71for regression-based alternatives. This contrast likely
reflects the objectives: InfoNCE (Oord et al., 2018) directly preserves the relative geometry required
Figure 5:Ten-wayidentificationandFADpergenerator:anchoringrestoresAudioLDMandTangoFluxtoretrieval
level but not autoregressive MusicGen.
9

RAG-Audio: Faithful Brain-to-Audio Reconstruction Preprint
foridentification,whereascorrelationandmean-squarederroremphasizedimension-wiseagreement.
Per-subject results, peak relevance, and the full decoder comparison appear in Tables 3 and 4 and Fig. 8.
The selected voxels also exhibit an anatomically plausible distribution. Figure 3 shows that the relevance
Figure 6:Faithfulness (identification) versus novelty: direct is novel but unfaithful, retrieval faithful but verbatim,
RAG-Audio faithful and generative.
scores concentrate in the bilateral superior temporal gyrus, a region associated with auditory processing.
UndertheHarvard–Oxfordatlas(Desikanetal.,2006),78%ofthetop-rankedvoxelsfallwithinauditory
cortex acrossthe two hemispheres. This localizationsupports the interpretationthat the decoderrelies
primarily on stimulus-related auditory activity rather than scanner artefacts or head motion.
Prior DominationDirect brain-conditioned generation discards much of the stimulus information
recovered by the decoder. Although the decoded embedding reaches 0.43 identification accuracy,
generationreducesperformanceto0.14withAudioLDM,0.14withTangoFlux,and0.18withMusicGen,
onlymodestlyabovethe0.10chancelevel.Table1reportsthefullcomparison,withthecomplete 𝑁-way
breakdown in Table 2. The MusicGen result re-implements the decode-then-generate formulation of
(Denketal.,2023).Thedegradationisnotconfinedtoonegeneratorfamily.Itappearsinlatentdiffusion,
rectifiedflow,andautoregressivegeneration,whiledirect-generationFADrangesfrom7.89to13.50.
Taken together, these results are consistent with prior domination as a property of the decode-then-
generate recipe: the decoded condition remains informative, but the pretrained generator produces audio
governed largely by its own prior.
Exemplar AnchoringExemplar anchoring restores identification to the level of the retrieval reference
forlatent-diffusiongenerators.RAG-Audioreaches0.43withAudioLDMand0.40withTangoFlux,com-
pared with 0.14 for their directly conditioned counterparts and 0.10 chance. These values approximately
match both the0.40 retrieval baseline andthe 0.43 decoder-level identification score,corresponding to
anapproximately 3×improvementoverdirectgeneration.ThesameinterventionreducesAudioLDM
FAD from 13.50 to 1.25, an approximately order-of-magnitude change, and TangoFlux FAD from 7.89
to2.36(Table1);thesamereductionsholdforFDandKL(Table5).Thiscomparisonshouldnotberead
as RAG-Audio outperforming retrieval. Retrieval directly returns real training audio and is therefore the
10

RAG-Audio: Faithful Brain-to-Audio Reconstruction Preprint
Figure 7:Per-genre reconstructions: stimulus, RAG-Audio, and direct generation (log-mel spectrograms). Cyan
linesare stimulusonsets(shared perrow);the cornerboxis CLAPsimilaritytothe stimulus(green/orange/red).
Directgenerationisrealisticbutoff-gridandlow-similarity;RAG-Audiorecoversonsetandharmonicstructure
(on-grid, high) without copying any retrieved clip. Structural, not sample-accurate, closeness.
strongest non-generative reference, with an FAD of 0.89. RAG-Audio does not surpass this value, nor do
weclaimthatitdoes.Itsadvantageisinsteadthatitmatchesretrieval-levelidentificationwhileproducing
an edited sample rather than returning the retrieved clip verbatim. Figure 6 reflects this distinction:
11

RAG-Audio: Faithful Brain-to-Audio Reconstruction Preprint
novelty increases from approximately 0.06 for retrieval to 0.18 for RAG-Audio while identification
remains at the retrieval level (Table 6). A genre-level breakdown of the residual errors (Appendix G,
Table. 8) shows they remain musically coherent, concentrating among acoustically similar GTZAN
genres (Tzanetakis and Cook, 2002).
Theanchoringstrengthcontrolsthisbalance.AsshowninFig.4(fullsweepinTable7),identification
decreasessmoothlyfromtheretrievalregimetowardthe0.10chancelevelasthestrengthapproaches
1, while novelty increases as more exemplar structure is removed. The range 𝑠≈0.25–0.40provides
the most useful operating region in our experiments, retaining stimulus-related structure without
reducing generation to direct retrieval. The anchoring advantage is also stableacross memory-bank sizes
(Appendix F, Table 8).
Mechanism ControlThe autoregressive negative control indicates that exemplar availability alone is
insufficient. On MusicGen, adding the retrieved exemplar changes identification only from 0.18 to 0.20,
in contrast to the approximately 3×gains observed for AudioLDM and TangoFlux in Fig. 5. MusicGen
hasnocontinuouslatenttrajectorythatadmitsintermediate-timeinitialization;theexemplarcanenter
only through melody conditioning, which remains governed by the autoregressive token prior.
FAD nevertheless improves modestly, from 8.06 to 6.21. This pattern suggests that the exemplar can
nudgeacousticrealismwithoutrestoringstimulusidentity.Thecross-generatorcomparisontherefore
pointsto intermediate-timetrajectory initialization,ratherthanthe merepresenceof aretrieved example,
as the mechanism responsible for the faithfulness gains.
Qualitative reconstructions.Figure 7 makes prior domination and its mitigation directly visible.
Direct brain-conditioned generation produces spectrograms with plausible spectrotemporal texture, yet
their energy drifts off the stimulus onset grid (cyan) and their CLAP similarity to the stimulus stays
lowacrossallsixexamples(red/orangeboxes).Thefailureisthuseasytomissfromrealismorcasual
listening alone precisely the regime we call prior domination. Exemplar anchoring recovers the structure
thepriordiscards:RAG-Audioreconstructionsalignwiththeonsetgridandreproducetheharmonic
banding of the stimulus, and the CLAP-similarity boxes rise correspondingly (green), roughly doubling
overdirectgenerationacrosstheshowngenres.Atthesametime,theRAGpanelsdifferinfinedetail
fromboththestimulusandanysingleretrievedclip,consistentwiththenoveltyresultsinTable6and
confirming that anchoring edits rather than copies.
7 Discussion
Our results support a simple account of when and why brain-to-audio generation fails. A pretrained
generator encodes a strong prior over natural audio, whereas the brain-derived condition is weak
and noisy; combined through direct conditioning, the prior dominates and the output drifts toward
a fluent but stimulus-agnostic sample. Anchoring the sampling trajectory on a retrieved real-audio
exemplar tethers the output to stimulus-related structure, converting prior domination into a controllable
faithfulness-novelty trade-off governed by the anchoring strength.
This mitigation is deliberately generator-specific, and the specificity is informative rather than
incidental. Because exemplar anchoring acts by initializing a continuous latent trajectory, it applies
to latent-diffusion and rectified-flow generators but not to autoregressive models, which expose no
suchtrajectory.TheMusicGennegativecontrolmakesthisconcrete:supplyingtheidenticalexemplar
raises 10-way identification only marginally, from 0.18to0.20, whereas the same exemplar restores
12

RAG-Audio: Faithful Brain-to-Audio Reconstruction Preprint
AudioLDM and TangoFlux to retrieval-level accuracy ( 0.43and0.40). Trajectory initialization, not the
mere availability of a retrieved clip, is therefore the operative mechanism.
We are precise about what RAG-Audio does and does not achieve. It does not surpass plain retrieval
on FAD, and it cannot: retrieval returns real training audio, which is FAD-optimal by construction
(0.89).RAG-Audio’scontributionisinsteadtomatchretrieval-levelidentificationwhileemittinganewly
generated sample rather than replaying a stored clip, cutting FAD by roughly an order of magnitude
relative to direct generation (e.g., AudioLDM 13.5→1.25 ). Its niche is exactly the setting that retrieval
cannot serve: where novel audio is required and a verbatim training clip is unacceptable, yet faithfulness
to the heard stimulus must still be preserved.
8 Limitations
Several limitations bound our claims. The study covers a single dataset and a single modality: music
decodedfromfMRIsotransfertospeechorenvironmentalsoundremainsuntested.Thememorybankis
boundedbythetrainingstimuli;RAG-Audioanchorsto,andthereforecannotreconstructcontentoutside,
that support. Exemplar anchoring requires an initializable latent trajectory and thus does not extend
toautoregressivegenerators.Our 𝑁-wayidentificationmeasuressemanticagreementinCLAPspace
ratherthan sample-accuratewaveformrecovery, sothe spectrogramsshould notbe readas evidenceof
exactreconstruction.Finally,theanchoringstrength 𝑠isahyperparameter:wereportaworkingrange
(𝑠≈0.25–0.40)ratherthanasingleoptimum,andthebestoperatingpointmaydifferacrossgenerators
and datasets.
9 Conclusion
Weidentifiedpriordominationasageneralfailuremodeofthedecode-then-generaterecipeforbrain-
to-audio reconstruction: a decoded embedding that is itself identifiable ( 0.43at10-way, against 0.10
chance) can nonetheless collapse to near-chance reconstructions ( 0.14–0.18) once a strong generator
isapplied.Weintroducedexemplaranchoring,initializingthegenerator’ssamplingtrajectoryfroma
retrieved real-audio clip, via SDEdit-style noising for diffusion models and the analogous rectified-flow
interpolation for flow models and showed that it restores retrieval-level identification while keeping
generation genuinely generative and reducing FAD by roughly an order of magnitude. An autoregressive
negative control localizes the effect to latent-trajectory initialization rather than retrieval alone. Beyond
the method,our single-harnesscomparison turnsa failure modepreviouslyreported onlyqualitatively
inbrain-to-imageworkintoaquantitative,generator-generalcharacterizationonelikelytosharpenas
pretrained generators continue to improve.
Acknowledgements
This work was supported by the European Union’s Horizon Europe research and innovation programme
under the Marie Skłodowska-Curie grant agreement No. 101205348 (CASPER). We acknowledge
the EuroHPC Joint Undertaking for awarding this project access to the EuroHPC supercomputer
LEONARDO, hosted by CINECA (Italy) and the LEONARDO consortium, through the EuroHPC
AI Factories "AI for Science and Collaborative EU Projects" Access call (proposal No. EHPC-AIF-
2026SC01-041). We further acknowledge the CINECA award under the ISCRA initiative (Class C
project IsCd5_CASPER-A), for the availability of high performance computing resources and support.
13

RAG-Audio: Faithful Brain-to-Audio Reconstruction Preprint
Views and opinions expressed are however those of the author(s) only and do not necessarily reflect
those of the European Union or the European Research Executive Agency. Neither the European Union
nor the granting authority can be held responsible for them.
References
Emily J Allen, Ghislain St-Yves, Yihan Wu, Jesse L Breedlove, Jacob S Prince, Logan T Dowdle,
Matthias Nau, Brad Caron, Franco Pestilli, Ian Charest, et al. A massive 7t fmri dataset to bridge
cognitive neuroscience and artificial intelligence.Nature neuroscience, 25(1):116–126, 2022.
Gopala K Anumanchipalli, Josh Chartier, and Edward F Chang. Speech synthesis from neural decoding
of spoken sentences.Nature, 568(7753):493–498, 2019.
Brian B Avants, Nicholas J Tustison, Gang Song, Philip A Cook, Arno Klein, and James C Gee. A
reproducibleevaluationofantssimilaritymetricperformanceinbrainimageregistration.Neuroimage,
54(3):2033–2044, 2011.
LudovicBellier,AnaïsLlorens,DéborahMarciano,AysegulGunduz,GerwinSchalk,PeterBrunner,and
RobertTKnight. Musiccanbereconstructedfromhumanauditorycortexactivityusingnonlinear
decoding models.PLoS biology, 21(8):e3002176, 2023.
AndreasBlattmann,RobinRombach,KaanOktay,JonasMüller,andBjörnOmmer. Semi-parametric
neural image synthesis.arXiv preprint arXiv:2204.11824, 2022.
SebastianBorgeaud,ArthurMensch,JordanHoffmann,TrevorCai,ElizaRutherford,KatieMillican,
GeorgeBmVanDenDriessche,Jean-BaptisteLespiau,BogdanDamoc,AidanClark,etal. Improving
language models by retrieving from trillions of tokens. InInternational conference on machine
learning, pages 2206–2240. PMLR, 2022.
Zijiao Chen, Jiaxin Qing, Tiange Xiang, Wan Lin Yue, and Juan Helen Zhou. Seeing beyond the
brain: Conditional diffusion model with sparse masked modeling for vision decoding. In2023
IEEE/CVFConferenceonComputerVisionandPatternRecognition(CVPR),pages22710–22720.
IEEE Computer Society, 2023.
Jooyoung Choi, Sungwon Kim, Yonghyun Jeong, Youngjune Gwon, and Sungroh Yoon. ILVR:
Conditioning method for denoising diffusion probabilistic models. InIEEE/CVF International
Conference on Computer Vision (ICCV), 2021. arXiv:2108.02938.
MatteoCiferri,MatteoFerrante,andNicolaToschi. Reconstructingmusicperceptionfrombrainactivity
using a prior guided diffusion model.Scientific Reports, 15(1):42108, 2025.
Jade Copet, Felix Kreuk, Itai Gat, Tal Remez, David Kant, Gabriel Synnaeve, Yossi Adi, and Alexandre
Défossez. Simple and controllable music generation.Advances in neural information processing
systems, 36:47704–47720, 2023.
Alexandre Défossez, Charlotte Caucheteux, Jérémy Rapin, Ori Kabeli, and Jean-Rémi King. Decoding
speechperceptionfromnon-invasivebrainrecordings.NatureMachineIntelligence,5(10):1097–1107,
2023.
14

RAG-Audio: Faithful Brain-to-Audio Reconstruction Preprint
Timo I Denk, Yu Takagi, Takuya Matsuyama, Andrea Agostinelli, Tomoya Nakai, Christian Frank, and
Shinji Nishimoto. Brain2music: Reconstructing music from human brain activity.arXiv e-prints,
pages arXiv–2307, 2023.
Rahul S Desikan, Florent Ségonne, Bruce Fischl, Brian T Quinn, Bradford C Dickerson, Deborah
Blacker, Randy L Buckner, Anders M Dale, R Paul Maguire, Bradley T Hyman, et al. An automated
labelingsystemforsubdividingthehumancerebralcortexonmriscansintogyralbasedregionsof
interest.Neuroimage, 31(3):968–980, 2006.
PrafullaDhariwal andAlexanderNichol. Diffusion modelsbeat gansonimage synthesis.Advancesin
neural information processing systems, 34:8780–8794, 2021.
Matteo Ferrante, Matteo Ciferri, and Nicola Toschi. R&b–rhythm and brain: Cross-subject decoding of
music from human brain activity.arXiv preprint arXiv:2406.15537, 2024.
ShawnHershey,SourishChaudhuri,DanielPWEllis,JortFGemmeke,ArenJansen,RChanningMoore,
Manoj Plakal, Devin Platt, Rif A Saurous, Bryan Seybold, et al. Cnn architectures for large-scale
audio classification. In2017 ieee international conference on acoustics, speech and signal processing
(icassp), pages 131–135. IEEE, 2017.
JonathanHoandTimSalimans. Classifier-freediffusionguidance.arXivpreprintarXiv:2207.12598,
2022.
JonathanHo,AjayJain,andPieterAbbeel. Denoisingdiffusionprobabilisticmodels.Advancesinneural
information processing systems, 33:6840–6851, 2020.
QingqingHuang,ArenJansen,JoonseokLee,RaviGanti,JudithYueLi,andDanielPWEllis. Mulan:A
joint embedding of music audio and natural language.arXiv preprint arXiv:2208.12415, 2022.
Chia-Yu Hung, Navonil Majumder, Zhifeng Kong, Ambuj Mehrish, Amir Ali Bagherzadeh, Chuan
Li, Rafael Valle, Bryan Catanzaro, and Soujanya Poria. Tangoflux: Super fast and faithful text
to audio generation with flow matching and clap-ranked preference optimization.arXiv preprint
arXiv:2412.21037, 2024.
Jingyang Huo, Yikai Wang, Xuelin Qian, Yun Wang, Chong Li, Jianfeng Feng, and Yanwei Fu.
Neuropictor: Refiningfmri-to-image reconstructionvia multi-individual pretrainingand multi-level
modulation.arXiv preprint arXiv:2403.18211, 2024.
AlexanderJEKell,DanielLKYamins,EricaNShook,SamVNorman-Haignere,andJoshHMcDermott.
Atask-optimizedneuralnetworkreplicateshumanauditorybehavior,predictsbrainresponses,and
reveals a cortical processing hierarchy.Neuron, 98(3):630–644, 2018.
Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, and Mike Lewis. Generalization
through memorization: Nearest neighbor language models.arXiv preprint arXiv:1911.00172, 2019.
K Kilgour et al. Frechet audio distance: A reference-free metric for evaluating music enhancem.
InterSpeech, 2019.
Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, and Mark D Plumbley. Panns:
Large-scale pretrained audio neural networks for audio pattern recognition.IEEE/ACM Transactions
on Audio, Speech, and Language Processing, 28:2880–2894, 2020.
15

RAG-Audio: Faithful Brain-to-Audio Reconstruction Preprint
VladimirKulikov,MatanKleiner,InbarHuberman-Spiegelglas,andTomerMichaeli. Flowedit:Inversion-
free text-based editing using pre-trained flow models. InProceedings of the IEEE/CVF International
Conference on Computer Vision, pages 19721–19730, 2025.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation
forknowledge-intensivenlptasks.Advancesinneuralinformationprocessingsystems,33:9459–9474,
2020.
YaronLipman, Ricky TQ Chen, HeliBen-Hamu, Maximilian Nickel, and Matthew Le. Flowmatching
for generative modeling. InThe eleventh international conference on learning representations.
Che Liu, Changde Du, Xiaoyu Chen, and Huiguang He. Reverse the auditory processing pathway:
Coarse-to-fine audio reconstruction from fmri.arXiv preprint arXiv:2405.18726, 2024.
Haohe Liu, Zehua Chen, Yi Yuan, Xinhao Mei, Xubo Liu, Danilo Mandic, Wenwu Wang, and Mark D
Plumbley. Audioldm: Text-to-audio generation with latent diffusion models. InInternational
Conference on Machine Learning, pages 21450–21474. PMLR, 2023a.
XingchaoLiu,ChengyueGong,andQiangLiu. Flowstraight andfast:Learningto generateandtransfer
data with rectified flow. InInternational conference on learning representations (ICLR), 2023b.
Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool.
Repaint: Inpainting using denoising diffusion probabilistic models. InProceedings of the IEEE/CVF
conference on computer vision and pattern recognition, pages 11461–11471, 2022.
Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon.
Sdedit:Guidedimagesynthesisandeditingwithstochasticdifferentialequations. InInternational
Conference on Learning Representations.
Sean L Metzger, Kaylo T Littlejohn, Alexander B Silva, David A Moses, Margaret P Seaton, Ran Wang,
Maximilian E Dougherty, Jessie R Liu, Peter Wu, Michael A Berger, et al. A high-performance
neuroprosthesis for speech decoding and avatar control.Nature, 620(7976):1037–1046, 2023.
Tomoya Nakai, Naoko Koide-Majima, and Shinji Nishimoto. Music genre neuroimaging dataset.Data
in Brief, 40:107675, 2022.
ShinjiNishimoto,AnTVu,ThomasNaselaris,YuvalBenjamini,BinYu,andJackLGallant. Recon-
structingvisualexperiencesfrombrainactivityevokedbynaturalmovies.Currentbiology,21(19):
1641–1646, 2011.
Sam Norman-Haignere, Nancy G Kanwisher, and Josh H McDermott. Distinct cortical pathways for
music and speech revealed by hypothesis-free voxel decomposition.neuron, 88(6):1281–1296, 2015.
Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive
coding.arXiv preprint arXiv:1807.03748, 2018.
Jong-Yun Park, Mitsuaki Tsukamoto, Misato Tanaka, and Yukiyasu Kamitani. Sound reconstruction
fromhumanbrainactivityviaagenerativemodelwithbrain-likeauditoryfeatures.arXivpreprint
arXiv:2306.11629, 2023.
16

RAG-Audio: Faithful Brain-to-Audio Reconstruction Preprint
Emilian Postolache, Natalia Polouliakh, Hiroaki Kitano, Akima Connelly, Emanuele Rodolà, Luca
Cosmo,andTaketoAkama. Naturalisticmusicdecodingfromeegdatavialatentdiffusionmodels.
InICASSP2025-2025IEEEInternationalConferenceonAcoustics,SpeechandSignalProcessing
(ICASSP), pages 1–5. IEEE, 2025.
RobinRombach,AndreasBlattmann,DominikLorenz,PatrickEsser,andBjörnOmmer. High-resolution
imagesynthesiswithlatentdiffusionmodels. InProceedingsoftheIEEE/CVFconferenceoncomputer
vision and pattern recognition, pages 10684–10695, 2022.
Litu Rout, Yujia Chen, Nataniel Ruiz, Constantine Caramanis, Sanjay Shakkottai, and Wen-Sheng Chu.
Semantic image inversion and editing using rectified stochastic differential equations.arXiv preprint
arXiv:2410.10792, 2024.
Paul Scotti, Atmadeep Banerjee, Jimmie Goode, Stepan Shabalin, Alex Nguyen, Aidan Dempster,
Nathalie Verlinde, Elad Yundler, David Weisberg, Kenneth Norman, et al. Reconstructing the mind’s
eye: fmri-to-imagewithcontrastive learning anddiffusionpriors.AdvancesinNeural Information
Processing Systems, 36:24705–24728, 2023.
Paul S Scotti, Mihir Tripathy, Cesare Kadir Torrico Villanueva, Reese Kneeland, Tong Chen, Ashutosh
Narang,CharanSanthirasegaran,JonathanXu,ThomasNaselaris,KennethANorman,etal. Mindeye2:
shared-subjectmodelsenablefmri-to-imagewith1hourofdata.InProceedingsofthe41stInternational
Conference on Machine Learning, pages 44038–44059, 2024.
Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models.arXiv preprint
arXiv:2010.02502, 2020a.
Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben
Poole. Score-based generative modeling through stochastic differential equations.arXiv preprint
arXiv:2011.13456, 2020b.
YuTakagiandShinjiNishimoto. High-resolutionimagereconstructionwithlatentdiffusionmodelsfrom
humanbrainactivity. InProceedingsoftheIEEE/CVFconferenceoncomputervisionandpattern
recognition, pages 14453–14463, 2023.
Jerry Tang, Amanda LeBel, Shailee Jain, and Alexander G Huth. Semantic reconstruction of continuous
language from non-invasive brain recordings.Nature Neuroscience, 26(5):858–866, 2023.
Greta Tuckute, Jenelle Feather, Dana Boebinger, and Josh H McDermott. Many but not all deep neural
network audio models capture brain responses and exhibit correspondence between model stages and
brain regions.Plos Biology, 21(12):e3002366, 2023.
George Tzanetakis and Perry Cook. Musical genre classification of audio signals.IEEE Transactions on
speech and audio processing, 10(5):293–302, 2002.
YusongWu,KeChen,TianyuZhang,YuchenHui,TaylorBerg-Kirkpatrick,andShlomoDubnov. Large-
scalecontrastivelanguage-audiopretrainingwithfeaturefusionandkeyword-to-captionaugmentation.
InICASSP2023-2023IEEEInternationalConferenceonAcoustics,SpeechandSignalProcessing
(ICASSP), pages 1–5. IEEE, 2023.
YiYuan,HaoheLiu,XuboLiu,QiushiHuang,MarkDPlumbley,andWenwuWang.Retrieval-augmented
text-to-audio generation. InICASSP 2024-2024 IEEE International Conference on Acoustics, Speech
and Signal Processing (ICASSP), pages 581–585. IEEE, 2024.
17

RAG-Audio: Faithful Brain-to-Audio Reconstruction Preprint
A Code and Reproducibility
For reproducibility, we release the fullcodebase, including preprocessing, the fMRI-to-CLAP decoder,
retrieval, exemplar anchoring, and evaluation, in a public repository:
TBA
Therepositorydocumentstheenvironment,checkpoints,andconfigurationfilesneededtoreproduce
every table and figure in this document.
B Use of Large Language Models
Large language models were used only to improve grammar and clarity in author-written text.
C Implementation Details
Decoder.Thecontrastivedecoder 𝑓𝜃isamultilayerperceptronmappingthe 2000selectedvoxelsto
the512-dimensionalCLAPspace.Itistrainedfor 400epochswiththeInfoNCEobjective(Oordetal.,
2018) at temperature 𝜏=0.07, using the Adam optimizer with weight decay; full layer widths, learning
rate,andbatchsizearegiveninourreleasedconfiguration.Voxelsareselectedbytherelevancescore
of Eq. 5 (𝐾=2000 ), with a haemodynamic lag of ℓ=3TRs and a temporal window of 𝑤=8TRs.
Motion correction is rigid-body, performed in native space with ANTsPy (Avants et al., 2011).
Generators.All three generators are frozen and used at their public checkpoints: AudioLDM
(cvssp/audioldm-s-full v2 ) (Liu et al., 2023a), TangoFlux ( declare-lab/TangoFlux ) (Hung
et al., 2024), and MusicGen ( facebook/musicgen-melody ) (Copet et al., 2023). For anchored
generation we sweep the strength grid 𝑠∈{0.15,0.25,0.35,0.5,0.65,0.8,1.0} ; generation step counts
and guidance scales are listed in the released configuration. Identification uses the LAION CLAP
checkpoint laion/clap-htsat-unfused (Wuet al.,2023),whichisdistinctfrom theCLAPvariant
used as the decoder target.
D Decoder Baselines
The contrastive decoder is compared against ridge, linear, and MLP regressors trained to predict the
CLAP target directly. As summarized in Figure 8, the contrastive decoder attains the best 10-way
identificationdespitethelowestdimension-wisecorrelationwiththetargetembedding(Pearson 𝑟=0.20,
versus 0.68–0.71for the regression decoders). This apparent paradox reflects the training objective:
InfoNCEoptimizestherelativecosinegeometrythat 𝑁-wayidentificationactuallymeasures,whereas
ridge/linear/MLPminimizedimension-wiseerror,whichneednotpreservenearest-neighborrankings.
Because retrieval and identification both operate on cosine similarity, the contrastive criterion is the
better match to the downstream task.
18

RAG-Audio: Faithful Brain-to-Audio Reconstruction Preprint
Figure 8:Decoder 10-way identification (5-subj mean).
Table 2:Identificationat 𝑁=2,5,10 foreverygenerator×arm(direct,RAG,retrieval; 𝑛=300,chance 1/𝑁).RAG
matches the retrieval bound for the latent-trajectory generators but not MusicGen. †: only 10-way was aggregated
for AudioLDM (Table 1).
Generator Arm 2-way 5-way 10-way
AudioLDMdirect† †0.14
RAG† †0.43
retrieval† †0.40
TangoFluxdirect 0.515 0.249 0.140
RAG 0.785 0.555 0.403
retrieval 0.792 0.554 0.406
MusicGendirect 0.590 0.310 0.180
RAG 0.615 0.336 0.201
retrieval 0.796 0.553 0.402
E Per-Subject Results
Thedecoderisreliableacrossallfivesubjects.Table3reportsper-subjectidentificationforthecontrastive
decoder ( 60test clips each): every subject exceeds chance at all 𝑁, with a mean 10-way accuracy of
0.434(chance 0.10) and a best subject (sub-003) at 0.532. Table 4 reports the peak voxel relevance
(max|corr| with the CLAP target) per subject, ranging from 0.425(sub-001) to 0.599(sub-003); the
orderingtracksidentification accuracy, indicatingthat subjects withstrongerauditory-cortex coupling
aredecodedmorereliably.Bilateralsuperior-temporal-gyruslocalizationisconsistentacrosssubjects
(Figure 3), and the single-subject map for sub-001 is shown in the main text.
19

RAG-Audio: Faithful Brain-to-Audio Reconstruction Preprint
Table 3:Per-subjectidentificationofthecontrastivedecoder( 60testclipseach;chance 1/𝑁).Allfivesubjects
exceed chance at every𝑁, confirming the decoder is reliable across subjects.
Subject 2-way 5-way 10-way 50-way Pearson
sub-001 0.782 0.555 0.393 0.121 0.177
sub-002 0.820 0.591 0.421 0.145 0.189
sub-003 0.836 0.651 0.532 0.206 0.234
sub-004 0.790 0.560 0.437 0.173 0.174
sub-005 0.794 0.546 0.388 0.154 0.233
mean 0.804 0.581 0.434 0.159 0.201
Table 4:Per-subjectpeakvoxelrelevance(max |corr|withtheCLAPtarget),indexingdecoderstrengthinauditory
cortex for each subject.
Subject sub-001 sub-002 sub-003 sub-004 sub-005
peak|corr|0.425 0.496 0.599 0.454 0.485
Table 5:Audio realism ( 𝑛=300; lower is better): FAD (VGGish) with FD and KL (PANNs Cnn14). RAG reduces
all three for AudioLDM and TangoFlux toward the real-audio retrieval reference, while MusicGen changes little.
Generator Arm FAD FD KL
AudioLDMdirect 13.49 79.40 3.08
RAG 1.25 17.23 1.07
retrieval 0.89 16.40 1.05
TangoFluxdirect 7.89 59.11 2.24
RAG 2.36 21.49 1.01
retrieval 0.89 16.40 1.05
MusicGendirect 8.06 52.99 1.85
RAG 6.21 59.32 1.82
retrieval 0.89 17.55 1.08
Table 6:Novelty (1−costo the nearest training clip) and pairwise diversity for the TangoFlux arms. RAG is far
more novel than verbatim retrieval (≈0) while staying below unconstrained direct generation.
Arm novelty diversity
retrieval 0.06 0.42
RAG (ours) 0.18 0.39
direct (prior) 0.33 0.48
F Memory-Bank Size Ablation
G Genre Confusion
Tocharacterizetheresidualerrors,eachgeneratedclipislabelledbythegenreofitsnearestground-truth
neighbourinCLAPspace(leave-one-out).Figure9showstheresultingconfusionmatrixfortheRAG
arm (TangoFlux): top-1 genre accuracy is 0.33, or3.3×the0.10chance level. Well-separated genres
are recovered most often (classical 0.63, jazz 0.53, hip hop/pop 0.43), while the errors concentrate
20

RAG-Audio: Faithful Brain-to-Audio Reconstruction Preprint
amongtheacousticallyoverlappingGTZAN(TzanetakisandCook,2002)genres(metal,rock,blues)
that are confusable even for audio-only classifiers. The confusions are musically coherent (e.g.,
disco↔pop↔reggae, rock↔disco), indicating that reconstruction errors preserve broad timbral and
rhythmic structure rather than being random.
Figure 9:Genre confusion for the RAG arm (TangoFlux).
H SDEdit Strength Sweep
Table7detailsthedependenceof 10-wayidentificationontheanchoringstrength 𝑠thatunderliesFigure4.
For AudioLDM,identification ishighest at lowstrength ( 0.49at𝑠=0.2) anddecreases monotonically
towardthe 0.10chancelevelas 𝑠→1,crossingtheretrievalboundwithintheworkingrange.TangoFlux
was evaluated at its operating point 𝑠=0.4, where it reaches 0.40. Realism is stable across the anchored
range:AudioLDMFADis 1.14,1.12,and 1.25at𝑠=0.15,0.2,and 0.25respectively,sofaithfulness
can be tuned over this interval without degrading audio quality. We therefore adopt 𝑠≈0.25–0.40as the
default operating region.
Table 7:Ten-way identification versus anchoring strength 𝑠(AudioLDM sweep; TangoFlux at its operating point).
Accuracyishighestatlow 𝑠anddecaystothe 0.10chancelevelas 𝑠→1;retr.=retrievalbound,direct=prior-only.
Generator𝑠=0.20.25 0.3 0.4 0.5 retr. direct
AudioLDM0.490.43 0.39 0.30 0.14 0.40 0.14
TangoFlux — — — 0.40 — 0.40 0.14
WetestedwhetherRAG-Audio’sadvantagegrowsastheretrievalmemoryshrinks,hypothesizing
21

RAG-Audio: Faithful Brain-to-Audio Reconstruction Preprint
gracefuldegradation.Thedatadidnotsupportthis.Table8reportAudioLDMRAG( 𝑠=0.2)against
retrievalFADatbanksizes 30,120,and 480.Atreducedbanksizestheestimatesarenoisy( 𝑛=60pairs
versus 300at the full bank), and RAG tracks retrieval within that noise rather than separating from it as
the bank shrinks. We report this as an honest negative result: the benefit of anchoring does not increase
under retrieval scarcity in this dataset.
Table 8:Memory-banksizeablation:AudioLDMRAG( 𝑠=0.2)vs.retrievalFADatbanksizes 30/120/480 (𝑛=60
pairs, noisy). RAG tracks retrieval within noise and does not degrade gracefully as the bank shrinks.
bank 30 bank 120 full (480)
RAG FAD 1.43 1.88 1.12
retrieval FAD 1.90 1.28 0.89
I Dataset and Reproducibility
WeusetheBrain2MusicfMRIdataset(Denketal.,2023),releasedasOpenNeuro ds003720 (Nakai
etal.,2022),obtainedviaDataLad( datalad get onthe *_bold.nii files);theauditorystimuliare
drawnfromthetenGTZANgenres(TzanetakisandCook,2002).Wefollowtheofficialsubject-wise
split of 480training and 60test clips per subject. Preprocessing applies ANTsPy rigid motion correction
in native space (Avants et al., 2011), a lag of 3TRs and window of 8TRs, and top- 2000voxel selection
byEq. 5.Auditory-cortexoverlapiscomputedagainst theHarvard-Oxfordatlas (Desikanetal.,2006).
All runs use a single GPU per job; we release seeds, code, and configurations.
22