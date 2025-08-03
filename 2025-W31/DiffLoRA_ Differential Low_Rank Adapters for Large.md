# DiffLoRA: Differential Low-Rank Adapters for Large Language Models

**Authors**: Alexandre Misrahi, Nadezhda Chirkova, Maxime Louis, Vassilina Nikoulina

**Published**: 2025-07-31 14:24:59

**PDF URL**: [http://arxiv.org/pdf/2507.23588v1](http://arxiv.org/pdf/2507.23588v1)

## Abstract
Differential Transformer has recently been proposed to improve performance in
Transformer models by canceling out noise through a denoiser attention
mechanism. In this work, we introduce DiffLoRA, a parameter-efficient
adaptation of the differential attention mechanism, with low-rank adapters on
both positive and negative attention terms. This approach retains the
efficiency of LoRA while aiming to benefit from the performance gains of
differential attention. We evaluate DiffLoRA across a broad range of NLP tasks,
including general benchmarks, many-shot in-context learning, RAG, and
long-context tests. We observe that, although DiffLoRA falls short of other
parameter-efficient fine-tuning methods in most evaluation tasks, it shows
interesting results in certain domains (+11 pts on LoRA for HumanEval). We
analyze the attention patterns post-finetuning to identify the reasons for this
behavior.

## Full Text


<!-- PDF content starts -->

DiffLoRA: Differential Low-Rank Adapters for Large Language Models
Alexandre Misrahi1*Nadezhda Chirkova2Maxime Louis2Vassilina Nikoulina2
1EPFL2NA VER LABS Europe
alexandre.misrahi@epfl.ch, vassilina.nikoulina@naverlabs.com
Abstract
Differential Transformer has recently been pro-
posed to improve performance in Transformer
models by canceling out noise through a de-
noiser attention mechanism. In this work,
we introduce DiffLoRA, a parameter-efficient
adaptation of the differential attention mecha-
nism, with low-rank adapters on both positive
and negative attention terms. This approach
retains the efficiency of LoRA while aiming to
benefit from the performance gains of differen-
tial attention. We evaluate DiffLoRA across
a broad range of NLP tasks, including gen-
eral benchmarks, many-shot in-context learn-
ing, RAG, and long-context tests. We observe
that, although DiffLoRA falls short of other
parameter-efficient fine-tuning methods in most
evaluation tasks, it shows interesting results
in certain domains (+11 pts on LoRA for Hu-
manEval). We analyze the attention patterns
post-finetuning to identify the reasons for this
behavior.
1 Introduction
Large language models (LLMs) have achieved re-
markable success across diverse NLP tasks, but
adapting these massive models to new domains or
tasks remains challenging and costly. Full fine-
tuning of an LLM for each application is often
infeasible due to the large number of parameters.
This drives the need for efficient and robust LLM
adaptation techniques that can customize model
behavior for a variety of tasks. Multiple parameter-
efficient fine-tuning methods have emerged to ad-
dress this challenge, the most prominent approach
being LoRA (Hu et al., 2021), which injects small
trainable weight matrices into a pre-trained model
instead of updating all weights.
In parallel, recent architectural innovations like
the Differential Transformer (Ye et al., 2024) have
tackled the well-known issue of attention sinks
*Work done during internship at NA VER LABS Europe(Xiao et al., 2024). Differential Transformer intro-
duces a differential attention mechanism (DiffAttn)
that amplifies attention to important context while
canceling out noise. This strategy demonstrated
significant performance improvement in context-
critical tasks such as Retrieval-Augmented Gen-
eration (RAG) or In-Context Learning (ICL), as
well as remarkable domain robustness. However, a
current limitation of this method is that it requires
to train a model from scratch.
In this work, we explore DiffLoRA1, a tech-
nique that integrates LoRA and DiffAttn to adapt
pre-trained LLMs using low-rank adapters. As in
(Grattafiori et al., 2024), LoRA adapters are in-
corporated at each layer of the model, enabling
it to learn the denoising weights associated with
DiffAttn. The goal of DiffLoRA is to adapt a
pre-trained model in a parameter-efficient man-
ner, while aiming to match the performance im-
provements demonstrated by the Differential Trans-
former, and potentially outperform corresponding
baselines in context-heavy tasks.
2 DiffLoRA
In this section, we provide a formal description
for our method. DiffLoRA uses a similar attention
function as Differential Transformer:
DiffAttn (X) =
smQ1KT
1√
d
−λ·smQ2KT
2√
d
V
where dis the model’s hidden size used in trans-
former layers, and smis the softmax function. In
the setting of a pre-trained LLM, we obtain Q1,K1
from pre-trained WQ1,WK1and aim to parameter-
efficiently train the denoising terms WQ2,WK2
such that computation time and resources are simi-
lar to LoRA. To do so, we train low-rank adapters
(Hu et al., 2021) BQ2, BK2∈RN×r, AQ2, AK2∈
1We release our code at https://github.com/
alexmsrh/difflora
1arXiv:2507.23588v1  [cs.CL]  31 Jul 2025

Rr×dsuch that
Q2=X(BQ2AQ2)
K2=X(BK2AK2)
We also add adapters on the positive term to in-
crease the expressiveness of the adapters:
Q1=X(WQ1+BQ1AQ1)
K1=X(WK1+BK1AK1)
where the new weights {A, B}{Q,K}{1,2}are ini-
tialized and trained as in LoRA. This is done for
the attention mechanism at each hidden layer of the
model.
3 Experiments
We train all models with a single epoch on Tulu-
2 (Ivison et al., 2023)2instruction tuning dataset.
We perform an additional experiment with Tulu-3
(Lambert et al., 2024) to assess the impact of larger
training dataset sizes. We rely on the open-instruct3
framework for finetuning. The hyperparameters
used for training are given in Appendix Table 2.
We proceed by describing the evaluation settings.
3.1 General Evaluation
We first evaluate the performances in terms of core
LLM abilities, to investigate whether the fine-tuned
model preserves its initial capabilities. We select
a subset of datasets representing different types of
knowledge encoded into LLMs: (1) Knowledge
recall (TruthfulQA, PopQA, ARC-challenge), (2)
reasoning (DROP, BBH), (3) math (GSM8k), (4)
coding (HumanEval). We use the OLMES frame-
work4designed for evaluation reproducibility in
LLMs. We rely on predefined evaluation settings/-
metrics for the above mentioned tasks.
3.2 Context-Sensitive Evaluation
For In-Context Learning (ICL) and Needle-in-the-
Haystack (NIH) tasks we rely on evaluation scheme
from HELMET (Yen et al., 2025).
In-Context-Learning. The TREC tasks (Li and
Roth, 2002) consist in classifying question type
among 6 and 50 labels, respectively. The Clinic150
task (Larson et al., 2019) and the Banking77 task
(Casanueva et al., 2020) consist in classifying ques-
tion intent among 151 and 77 classes, respectively.
2huggingface.co/datasets/allenai/tulu-v2-sft-mixture
3github.com/allenai/open-instruct
4github.com/allenai/olmesThe task aims to evaluate the capability of Dif-
fLoRA models to adapt to new tasks in a zero-shot
fashion. As in (Yen et al., 2025), we report accu-
racy on the test sets.
Needle-in-Haystack. We evaluate Needle-in-
Haystack (NIH) performance across several set-
tings: Multi-Key (MK) tasks the model to retrieve
the correct key in the context with multiple noisy
ones, and Multi-Value (MV) tasks the model to
retrieve all the values associated with a certain key.
RAG-QA We also evaluate our models in RAG
Q&A tasks, to assess their capability to exploit
context and generate sound text. We follow RAG
settings proposed in (Rau et al., 2024), using
BERGEN5framework. More details on RAG set-
tings are available in the Appendix. We evaluate
on both general QA benchmarks such as KILT-
NQ (Petroni et al., 2021) and PopQA (Mallen
et al., 2023), as well as more specific domain
benchmarks such as biomedical (BioASQ (Nen-
tidis et al., 2023)), tech-support (TechQA (Castelli
et al., 2019)) and finance (FiQA). This evaluates
the ability of DiffLoRA models to effectively an-
swer questions by using context retrieved from a
datastore.
3.3 Baselines
We use Llama-3.2-1B-Instruct model6as starting
point of our experiments. We compare the perfor-
mance of DiffLoRA to this model to assess the
impact of introducing denoiser adapters. In order
to decouple the effect of finetuning from the effect
of DiffAttn, we also perform LoRA finetuning on
the same tuning datasets. We set Full LoRA rank in
a way to match the number of trainable parameters
of DiffLoRA models (more details at Appendix
Tab. 2).
3.4 DiffLoRA Variants
We hypothesize that some adaptation might be nec-
essary in the positive term of attention in order to
better adapt to the introduction of the negative side.
To ensure that it is comparable with LoRA and the
full denoiser setting in terms of number of parame-
ters, we set the adapter rank for this variant to r/2,
where ris the rank for the setting with adapters only
on the negative term. In our experiments we set
r= 64 . In (Ye et al., 2024) an extra normalization
5github.com/naver/bergen
6meta-llama/Llama-3.2-1B-Instruct
2

TruthfulQAPopQA BoolQ
HumanEvalDROPGSM8KBBH
ARC-CAvgLlama-3.2-1B-Instruct
LoRA
DiffLoRA-64
DiffLoRA-64 (fix =0.1)
DiffLoRA-32
DiffLoRA-32 + GN
DiffLoRA-32 + Tulu30.46 0.14 0.68 0.64 0.29 0.40 0.37 0.43 0.43
0.42 0.15 0.76 0.53 0.33 0.36 0.35 0.43 0.42
0.49 0.14 0.73 0.64 0.26 0.35 0.36 0.45 0.43
0.44 0.14 0.72 0.61 0.26 0.36 0.36 0.45 0.42
0.47 0.14 0.71 0.61 0.26 0.35 0.37 0.44 0.42
0.49 0.00 0.39 0.00 0.01 0.01 0.02 0.24 0.15
0.49 0.13 0.69 0.62 0.26 0.35 0.36 0.45 0.42Figure 1: Evaluation of general LLM capabilities before and after finetuning. DiffLoRA-32: both right and left term
of diff attention contain learnable parameters, DiffLoRA-64: only right term is learnable
1000 2000 3000 4000 5000 600030405060Accuracy (%)
TREC fine
1000 2000 3000 4000 5000 600070758085
TREC coarse
2000 4000 6000
# Samples60708090Accuracy (%)
Clinic150
1000 2000 3000 4000 5000 6000
# Samples6065707580
Banking77Llama-3.2-1B-Instruct
FullLora, r8DiffLora, =0.1, right side, r64
DiffLora, learn , both sides, r32
Figure 2: Evaluation on Many-shot In-Context Learning
(Group Norm) applied to each head independently
to stabilize scale across heads, and a correspond-
ing scaling factor of (1−λinit)to stabilize this
normalization in the gradient. We therefore add
a model with Group Norm (GN) for comparison
(more details in Appendix 2). Following (Ye et al.,
2024), in our experiments we learn the parameter λ,
however we generally observe more stable results
by freezing λto a small fixed value ( 0.1).
4 Results
Fig. 1 reports the results of different DiffLoRA
variants, as well as baseline results for the origi-
nal model and LoRA finetuning. First, we note
that most of the models stay more or less on par
with the original model. We note that model’s
performances do vary according to the task (+11
pts in HumanEval, -7pts in DROP), but generally
stay within the same range as an original model.
The only exception is the model with Group Nor-
50000 100000
Context Length405060708090Accuracy (%)
MK = 2
50000 100000
Context Length010203040506070
MK = 3
50000 100000
Context Length405060708090
MVLlama-3.2-1B-Instruct
FullLora, r8
DiffLora, learn , both sides, r32
DiffLora, =0.1, right side, r64
DiffLora, =0.1, both sides, r32
Figure 3: Needle-in-Haystack tests with variants Mul-
tiKey (MK, one key to retrieve among multiple) and
MultiValue (MV , retrieve all values corresponding to
the given key)
malization. We believe that in case of pretrained
model such stabilization of gradients is less critical.
Moreover, it might hurt previously learnt attention
patterns and therefore degrade the results. In order
to assess whether DiffLoRA deals better with the
context we perform extra evaluations.
Many-Shot In-Context Learning The ICL re-
sults (Fig. 2) show that the DiffLoRA models
Table 1: RAG evaluation, with top-5 retrieved docu-
ments, evaluated with LLM-as-a-judge.
BioASQ PopQA TechQA
Llama-3.2-1B-Instruct 0.678 0.494 0.532
FullLoRA 0.728 0.528 0.556
DiiffLoRA-64 0.629 0.451 0.39
DiffLoRA-64- λ= 0.1 0.638 0.495 0.407
DiffLoRA-32 0.585 0.479 0.344
DiffLoRA-32 + GN 0.025 0.041 0.059
DiffLoRA-32 + Tulu3 0.594 0.466 0.339
3

BOSCONTEX
T 1MAGIC NUMBERCONTEX
T 2QUER
Y00.20.40.6
BOSCONTEX
T 1MAGIC NUMBERCONTEX
T 2QUER
YBOSCONTEX
T 1MAGIC NUMBERCONTEX
T 2QUER
YBOSCONTEX
T 1MAGIC NUMBERCONTEX
T 2QUER
YLlama-3.2-1B LoRA DLoRA-32 DLoRA, Tulu-3Figure 4: Change in attention pattern distribution in different models. For DiffLoRA variants we plot attention
mass for main component (green) and denoiser component (yellow). Note that attention mass is normalized by the
number of tokens in each part of the sequence. The negative attention is shown after it is scaled by λ. DiffLoRA
corresponds to the variant with learnable λand LoRa parameters in both terms.
perform similarly as the initial model, however
they are outperformed by LoRA. When increasing
the context length with more sample demonstra-
tions, DiffLoRA seems to struggle even more in
TREC-fine and Banking77. This might be due
to the nature of instruction tuned data, and the
max_sequence_length = 4096 applied during
finetuning. LoRA is less impacted, likely because
it diverges less from the initial model.
Needle-in-Haystack tests Needle-in-Haystack
tests reveal different hierarchies among models.
The initial model seems to outperform finetuned
models in all tasks. However, the hierarchy be-
tween LoRA and DiffLoRA models is not so clear:
in MK=2 LoRA significantly outperforms all Dif-
fLoRA variants (see Appendix C for a degenerate
example), while in MV task all DiffLoRA variants
largely outperform LoRA.
RAG-QA In RAG evaluation (Tab. 1 or full table
in Appendix D) the DiffLoRA significantly under-
performs compared to LoRA. Compared to the ini-
tial model, DiffLoRA performs better on general
domain benchmarks (KILT-NQ, PopQA), while,
surprisingly, DiffLoRA degrades even more on less
general domain tasks (BioASQ, TechQA, FiQA).
5 Discussion
Our experiments reveal that LoRA outperforms
DiffLoRA in most tasks. However, DiffLoRA out-
performs LoRA in some tasks such as Code ques-
tions and multiple-key retrieval. We performed
manual inspection of the results to better under-
stand models behavior after tuning. We observethat the generation capability of LLM gets broken
in DiffLoRA (examples in Appendix C). Such de-
generation could explain drop in performance on
RAG tasks. Most of the core LLM evaluation tasks
are performed in MCQA fashion, and therefore do
not explicitly evaluate generation capability.
Attention Mass. A significant characteristic of
DiffTransformer is the allocation of attention mass
on the relevant parts of the context, which effec-
tively suppresses the attention sinks (Xiao et al.,
2024). Fig. 4 compares the change in attention
patterns across different models. We note that Dif-
fLoRA slightly changes attention pattern compared
to initial model (Llama-3.2-1B-Instruct), by denois-
ing context around Magic Number, and decreasing
attention mass on BOS token. However, the over-
all pattern is pretty similar to the one obtained by
the model finetuned with LoRA. Therefore, such
behaviour could also be attributed to the data distri-
bution on which models were finetuned. We note
that increasing the number training data (Tulu-3 vs
Tulu-2) leads to stronger denoising, but we do not
observe strong pattern change, compared to the one
reported by (Ye et al., 2024). This suggests that
we would need much more data in order to learn a
different attention mechanism.
6 Conclusion
We introduced DiffLoRA, a parameter-efficient
method that incorporates differential attention into
pre-trained LLMs using low-rank adapters. Initial
results demonstrate some encouraging patterns but
more investigation is required to make such model
work as expected.
4

References
Iñigo Casanueva, Tadas Tem ˇcinas, Daniela Gerz,
Matthew Henderson, and Ivan Vuli ´c. 2020. Efficient
intent detection with dual sentence encoders. In Pro-
ceedings of the 2nd Workshop on Natural Language
Processing for Conversational AI , pages 38–45, On-
line. Association for Computational Linguistics.
Vittorio Castelli, Rishav Chakravarti, Saswati Dana, An-
thony Ferritto, Radu Florian, Martin Franz, Dinesh
Garg, Dinesh Khandelwal, Scott McCarley, Mike
McCawley, Mohamed Nasr, Lin Pan, Cezar Pen-
dus, John Pitrelli, Saurabh Pujar, Salim Roukos, An-
drzej Sakrajda, Avirup Sil, Rosario Uceda-Sosa, Todd
Ward, and Rong Zhang. 2019. The techqa dataset.
Preprint , arXiv:1911.02984.
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad Al-
Dahle, Aiesha Letman, Akhil Mathur, Alan Schel-
ten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh
Goyal, Anthony Hartshorn, Aobo Yang, Archi Mi-
tra, Archie Sravankumar, Artem Korenev, Arthur
Hinsvark, Arun Rao, Aston Zhang, Aurelien Ro-
driguez, Austen Gregerson, Ava Spataru, Baptiste
Roziere, Bethany Biron, Binh Tang, Bobbie Chern,
Charlotte Caucheteux, Chaya Nayak, Chloe Bi,
Chris Marra, Chris McConnell, Christian Keller,
Christophe Touret, Chunyang Wu, Corinne Wong,
Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Al-
lonsius, Daniel Song, Danielle Pintz, Danny Livshits,
Danny Wyatt, David Esiobu, Dhruv Choudhary,
Dhruv Mahajan, Diego Garcia-Olano, Diego Perino,
Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy,
Elina Lobanova, Emily Dinan, Eric Michael Smith,
Filip Radenovic, Francisco Guzmán, Frank Zhang,
Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis An-
derson, Govind Thattai, Graeme Nail, Gregoire Mi-
alon, Guan Pang, Guillem Cucurell, Hailey Nguyen,
Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan
Zarov, Imanol Arrieta Ibarra, Isabel Kloumann, Is-
han Misra, Ivan Evtimov, Jack Zhang, Jade Copet,
Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park,
Jay Mahadeokar, Jeet Shah, Jelmer van der Linde,
Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu,
Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang,
Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park,
Joseph Rocca, Joshua Johnstun, Joshua Saxe, Jun-
teng Jia, Kalyan Vasuden Alwala, Karthik Prasad,
Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth
Heafield, Kevin Stone, Khalid El-Arini, Krithika Iyer,
Kshitiz Malik, Kuenley Chiu, Kunal Bhalla, Kushal
Lakhotia, Lauren Rantala-Yeary, Laurens van der
Maaten, Lawrence Chen, Liang Tan, Liz Jenkins,
Louis Martin, Lovish Madaan, Lubo Malo, Lukas
Blecher, Lukas Landzaat, Luke de Oliveira, Madeline
Muzzi, Mahesh Pasupuleti, Mannat Singh, Manohar
Paluri, Marcin Kardas, Maria Tsimpoukelli, Mathew
Oldham, Mathieu Rita, Maya Pavlova, Melanie Kam-
badur, Mike Lewis, Min Si, Mitesh Kumar Singh,
Mona Hassan, Naman Goyal, Narjes Torabi, Niko-
lay Bashlykov, Nikolay Bogoychev, Niladri Chatterji,
Ning Zhang, Olivier Duchenne, Onur Çelebi, PatrickAlrassy, Pengchuan Zhang, Pengwei Li, Petar Va-
sic, Peter Weng, Prajjwal Bhargava, Pratik Dubal,
Praveen Krishnan, Punit Singh Koura, Puxin Xu,
Qing He, Qingxiao Dong, Ragavan Srinivasan, Raj
Ganapathy, Ramon Calderer, Ricardo Silveira Cabral,
Robert Stojnic, Roberta Raileanu, Rohan Maheswari,
Rohit Girdhar, Rohit Patel, Romain Sauvestre, Ron-
nie Polidoro, Roshan Sumbaly, Ross Taylor, Ruan
Silva, Rui Hou, Rui Wang, Saghar Hosseini, Sa-
hana Chennabasappa, Sanjay Singh, Sean Bell, Seo-
hyun Sonia Kim, Sergey Edunov, Shaoliang Nie, Sha-
ran Narang, Sharath Raparthy, Sheng Shen, Shengye
Wan, Shruti Bhosale, Shun Zhang, Simon Van-
denhende, Soumya Batra, Spencer Whitman, Sten
Sootla, Stephane Collot, Suchin Gururangan, Syd-
ney Borodinsky, Tamar Herman, Tara Fowler, Tarek
Sheasha, Thomas Georgiou, Thomas Scialom, Tobias
Speckbacher, Todor Mihaylov, Tong Xiao, Ujjwal
Karn, Vedanuj Goswami, Vibhor Gupta, Vignesh
Ramanathan, Viktor Kerkez, Vincent Gonguet, Vir-
ginie Do, Vish V ogeti, Vítor Albiero, Vladan Petro-
vic, Weiwei Chu, Wenhan Xiong, Wenyin Fu, Whit-
ney Meers, Xavier Martinet, Xiaodong Wang, Xi-
aofang Wang, Xiaoqing Ellen Tan, Xide Xia, Xin-
feng Xie, Xuchao Jia, Xuewei Wang, Yaelle Gold-
schlag, Yashesh Gaur, Yasmine Babaei, Yi Wen,
Yiwen Song, Yuchen Zhang, Yue Li, Yuning Mao,
Zacharie Delpierre Coudert, Zheng Yan, Zhengxing
Chen, Zoe Papakipos, Aaditya Singh, Aayushi Sri-
vastava, Abha Jain, Adam Kelsey, Adam Shajnfeld,
Adithya Gangidi, Adolfo Victoria, Ahuva Goldstand,
Ajay Menon, Ajay Sharma, Alex Boesenberg, Alexei
Baevski, Allie Feinstein, Amanda Kallet, Amit San-
gani, Amos Teo, Anam Yunus, Andrei Lupu, An-
dres Alvarado, Andrew Caples, Andrew Gu, Andrew
Ho, Andrew Poulton, Andrew Ryan, Ankit Ramchan-
dani, Annie Dong, Annie Franco, Anuj Goyal, Apara-
jita Saraf, Arkabandhu Chowdhury, Ashley Gabriel,
Ashwin Bharambe, Assaf Eisenman, Azadeh Yaz-
dan, Beau James, Ben Maurer, Benjamin Leonhardi,
Bernie Huang, Beth Loyd, Beto De Paola, Bhargavi
Paranjape, Bing Liu, Bo Wu, Boyu Ni, Braden Han-
cock, Bram Wasti, Brandon Spence, Brani Stojkovic,
Brian Gamido, Britt Montalvo, Carl Parker, Carly
Burton, Catalina Mejia, Ce Liu, Changhan Wang,
Changkyu Kim, Chao Zhou, Chester Hu, Ching-
Hsiang Chu, Chris Cai, Chris Tindal, Christoph Fe-
ichtenhofer, Cynthia Gao, Damon Civin, Dana Beaty,
Daniel Kreymer, Daniel Li, David Adkins, David
Xu, Davide Testuggine, Delia David, Devi Parikh,
Diana Liskovich, Didem Foss, Dingkang Wang, Duc
Le, Dustin Holland, Edward Dowling, Eissa Jamil,
Elaine Montgomery, Eleonora Presani, Emily Hahn,
Emily Wood, Eric-Tuan Le, Erik Brinkman, Este-
ban Arcaute, Evan Dunbar, Evan Smothers, Fei Sun,
Felix Kreuk, Feng Tian, Filippos Kokkinos, Firat
Ozgenel, Francesco Caggioni, Frank Kanayet, Frank
Seide, Gabriela Medina Florez, Gabriella Schwarz,
Gada Badeer, Georgia Swee, Gil Halpern, Grant
Herman, Grigory Sizov, Guangyi, Zhang, Guna
Lakshminarayanan, Hakan Inan, Hamid Shojanaz-
eri, Han Zou, Hannah Wang, Hanwen Zha, Haroun
Habeeb, Harrison Rudolph, Helen Suk, Henry As-
pegren, Hunter Goldman, Hongyuan Zhan, Ibrahim
5

Damlaj, Igor Molybog, Igor Tufanov, Ilias Leontiadis,
Irina-Elena Veliche, Itai Gat, Jake Weissman, James
Geboski, James Kohli, Janice Lam, Japhet Asher,
Jean-Baptiste Gaya, Jeff Marcus, Jeff Tang, Jen-
nifer Chan, Jenny Zhen, Jeremy Reizenstein, Jeremy
Teboul, Jessica Zhong, Jian Jin, Jingyi Yang, Joe
Cummings, Jon Carvill, Jon Shepard, Jonathan Mc-
Phie, Jonathan Torres, Josh Ginsburg, Junjie Wang,
Kai Wu, Kam Hou U, Karan Saxena, Kartikay Khan-
delwal, Katayoun Zand, Kathy Matosich, Kaushik
Veeraraghavan, Kelly Michelena, Keqian Li, Ki-
ran Jagadeesh, Kun Huang, Kunal Chawla, Kyle
Huang, Lailin Chen, Lakshya Garg, Lavender A,
Leandro Silva, Lee Bell, Lei Zhang, Liangpeng
Guo, Licheng Yu, Liron Moshkovich, Luca Wehrst-
edt, Madian Khabsa, Manav Avalani, Manish Bhatt,
Martynas Mankus, Matan Hasson, Matthew Lennie,
Matthias Reso, Maxim Groshev, Maxim Naumov,
Maya Lathi, Meghan Keneally, Miao Liu, Michael L.
Seltzer, Michal Valko, Michelle Restrepo, Mihir Pa-
tel, Mik Vyatskov, Mikayel Samvelyan, Mike Clark,
Mike Macey, Mike Wang, Miquel Jubert Hermoso,
Mo Metanat, Mohammad Rastegari, Munish Bansal,
Nandhini Santhanam, Natascha Parks, Natasha
White, Navyata Bawa, Nayan Singhal, Nick Egebo,
Nicolas Usunier, Nikhil Mehta, Nikolay Pavlovich
Laptev, Ning Dong, Norman Cheng, Oleg Chernoguz,
Olivia Hart, Omkar Salpekar, Ozlem Kalinli, Parkin
Kent, Parth Parekh, Paul Saab, Pavan Balaji, Pe-
dro Rittner, Philip Bontrager, Pierre Roux, Piotr
Dollar, Polina Zvyagina, Prashant Ratanchandani,
Pritish Yuvraj, Qian Liang, Rachad Alao, Rachel
Rodriguez, Rafi Ayub, Raghotham Murthy, Raghu
Nayani, Rahul Mitra, Rangaprabhu Parthasarathy,
Raymond Li, Rebekkah Hogan, Robin Battey, Rocky
Wang, Russ Howes, Ruty Rinott, Sachin Mehta,
Sachin Siby, Sai Jayesh Bondu, Samyak Datta, Sara
Chugh, Sara Hunt, Sargun Dhillon, Sasha Sidorov,
Satadru Pan, Saurabh Mahajan, Saurabh Verma,
Seiji Yamamoto, Sharadh Ramaswamy, Shaun Lind-
say, Shaun Lindsay, Sheng Feng, Shenghao Lin,
Shengxin Cindy Zha, Shishir Patil, Shiva Shankar,
Shuqiang Zhang, Shuqiang Zhang, Sinong Wang,
Sneha Agarwal, Soji Sajuyigbe, Soumith Chintala,
Stephanie Max, Stephen Chen, Steve Kehoe, Steve
Satterfield, Sudarshan Govindaprasad, Sumit Gupta,
Summer Deng, Sungmin Cho, Sunny Virk, Suraj
Subramanian, Sy Choudhury, Sydney Goldman, Tal
Remez, Tamar Glaser, Tamara Best, Thilo Koehler,
Thomas Robinson, Tianhe Li, Tianjun Zhang, Tim
Matthews, Timothy Chou, Tzook Shaked, Varun
V ontimitta, Victoria Ajayi, Victoria Montanez, Vijai
Mohan, Vinay Satish Kumar, Vishal Mangla, Vlad
Ionescu, Vlad Poenaru, Vlad Tiberiu Mihailescu,
Vladimir Ivanov, Wei Li, Wenchen Wang, Wen-
wen Jiang, Wes Bouaziz, Will Constable, Xiaocheng
Tang, Xiaojian Wu, Xiaolan Wang, Xilun Wu, Xinbo
Gao, Yaniv Kleinman, Yanjun Chen, Ye Hu, Ye Jia,
Ye Qi, Yenda Li, Yilin Zhang, Ying Zhang, Yossi Adi,
Youngjin Nam, Yu, Wang, Yu Zhao, Yuchen Hao,
Yundi Qian, Yunlu Li, Yuzi He, Zach Rait, Zachary
DeVito, Zef Rosnbrick, Zhaoduo Wen, Zhenyu Yang,
Zhiwei Zhao, and Zhiyu Ma. 2024. The llama 3 herd
of models. Preprint , arXiv:2407.21783.Pengcheng He, Jianfeng Gao, and Weizhu Chen. 2021.
Debertav3: Improving deberta using electra-style pre-
training with gradient-disentangled embedding shar-
ing.Preprint , arXiv:2111.09543.
Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan
Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and
Weizhu Chen. 2021. Lora: Low-rank adaptation of
large language models. Preprint , arXiv:2106.09685.
Hamish Ivison, Yizhong Wang, Valentina Pyatkin,
Nathan Lambert, Matthew Peters, Pradeep Dasigi,
Joel Jang, David Wadden, Noah A. Smith, Iz Belt-
agy, and Hannaneh Hajishirzi. 2023. Camels in a
changing climate: Enhancing lm adaptation with tulu
2.Preprint , arXiv:2311.10702.
Nathan Lambert, Jacob Morrison, Valentina Pyatkin,
Shengyi Huang, Hamish Ivison, Faeze Brahman,
Lester James V . Miranda, Alisa Liu, Nouha Dziri,
Shane Lyu, Yuling Gu, Saumya Malik, Victoria
Graf, Jena D. Hwang, Jiangjiang Yang, Ronan Le
Bras, Oyvind Tafjord, Chris Wilhelm, Luca Soldaini,
Noah A. Smith, Yizhong Wang, Pradeep Dasigi, and
Hannaneh Hajishirzi. 2024. Tülu 3: Pushing frontiers
in open language model post-training.
Stefan Larson, Anish Mahendran, Joseph J. Peper,
Christopher Clarke, Andrew Lee, Parker Hill,
Jonathan K. Kummerfeld, Kevin Leach, Michael A.
Laurenzano, Lingjia Tang, and Jason Mars. 2019. An
evaluation dataset for intent classification and out-of-
scope prediction. In Proceedings of the 2019 Confer-
ence on Empirical Methods in Natural Language Pro-
cessing and the 9th International Joint Conference
on Natural Language Processing (EMNLP-IJCNLP) ,
pages 1311–1316, Hong Kong, China. Association
for Computational Linguistics.
Carlos Lassance, Hervé Déjean, Thibault Formal, and
Stéphane Clinchant. 2024. Splade-v3: New baselines
for splade. arXiv preprint arXiv:2403.06789 .
Xin Li and Dan Roth. 2002. Learning question clas-
sifiers. In COLING 2002: The 19th International
Conference on Computational Linguistics .
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das,
Daniel Khashabi, and Hannaneh Hajishirzi. 2023.
When not to trust language models: Investigating
effectiveness of parametric and non-parametric mem-
ories. Preprint , arXiv:2212.10511.
Anastasios Nentidis, Georgios Katsimpras, Anasta-
sia Krithara, Salvador Lima López, Eulália Farré-
Maduell, Luis Gasco, Martin Krallinger, and Geor-
gios Paliouras. 2023. Overview of BioASQ 2023: The
Eleventh BioASQ Challenge on Large-Scale Biomedi-
cal Semantic Indexing and Question Answering , page
227–250. Springer Nature Switzerland.
Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick
Lewis, Majid Yazdani, Nicola De Cao, James Thorne,
Yacine Jernite, Vladimir Karpukhin, Jean Maillard,
Vassilis Plachouras, Tim Rocktäschel, and Sebastian
Riedel. 2021. Kilt: a benchmark for knowledge in-
tensive language tasks. Preprint , arXiv:2009.02252.
6

David Rau, Hervé Déjean, Nadezhda Chirkova, Thibault
Formal, Shuai Wang, Vassilina Nikoulina, and
Stéphane Clinchant. 2024. Bergen: A benchmarking
library for retrieval-augmented generation. Preprint ,
arXiv:2407.01102.
Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song
Han, and Mike Lewis. 2024. Efficient streaming
language models with attention sinks. Preprint ,
arXiv:2309.17453.
Tianzhu Ye, Li Dong, Yuqing Xia, Yutao Sun, Yi Zhu,
Gao Huang, and Furu Wei. 2024. Differential trans-
former. Preprint , arXiv:2410.05258.
Howard Yen, Tianyu Gao, Minmin Hou, Ke Ding,
Daniel Fleischer, Peter Izsak, Moshe Wasserblat, and
Danqi Chen. 2025. Helmet: How to evaluate long-
context language models effectively and thoroughly.
InInternational Conference on Learning Representa-
tions (ICLR) .
A RAG settings
Following (Rau et al., 2024) Given a query, we
useSPLADE-v3 (Lassance et al., 2024) retriever
to identify a first set of relevant documents from
Wikipedia collection. These documents are further
reranked using DeBERTa-v3 (He et al., 2021), a
cross-encoder computing relevance score for each
document relative to the query. For generation,
we use instruction-tuned Llama-3.2-1B(Grattafiori
et al., 2024). To evaluate the quality of responses,
we rely on an evaluation computed by a LLM-as-
ajudge with the SOLAR-10.7B model7as back-
bone. (Rau et al., 2024) find that this metric has
high correlation with GPT4.
B Hyperparameters
See Table 2. Notice that we set the LoRA rank to
8 in order to match the total number of parameters
in the model, since LoRA also adds weights to
the feed-forward layers in addition to the attention
layers, as well as the value and output matrices
inside the attention.
7huggingface/upstage/SOLAR-10.7B-Instruct-v1.0Parameter Value
Learning Rate 1e-4
Full LoRA rank 8
DiffLoRa both terms rank=32, alpha=64
DiffLoRa, right term only rank=64, alpha=128
Batch size 64
max_input_length 4096
Table 2: Hyperparameters used in training. We em-
pirically identify a good learning rate of 1e-4 for both
LoRA and DiffLoRA.
C Examples
Example NIH-MK
What is the magic uuid for
4e6ccf9a-fbc2-41ba-8655-486660d4417c
LoRA: (correct) The
special magic uuid for
4e6ccf9a-fbc2-41ba-8655-486660d4417c
mentioned in the provided text is:
3f45a9e6-1a1d-4e5d-8e5d-3f5a9e6a6f5a.
DiffLoRA-right, learn λ: (degen-
erate) The special magic uuid for
4e6ccf9a-fbc2-41ba-8655-486660d4417c
mentioned in the provided text is:
3f977a9-0e8a-4f2a-8f5a-0a0a0a[...]
Example RAG-BioASQ
Is Prasinezumab effective for
Parkinson’s Disease?
LoRA: (somewhat correct) Yes,
Prasinezumab is being studied for
its effect on Parkinson’s disease.
DiffLoRA-right, learn λ: (degenerate)
Step 1: The question is not a valid
question. The question is not a
valid question [...]
D Other Results
7

Figure 5: Disttribution of attention mass depending on the position of learnable parameters in Differentaion attention
(right and left terms vs right term only), and the choic of λ(fixed or learnable). We note that distribution are quite
similar to the original models. When λis learnable and only parameters on the denoiser are learnt the change is
slightly more pronounced.
Table 3: RAG evaluation, with top-5 retrieved documents, evaluated with LLM-as-a-judge.
BioASQ FiQA KILT-NQ PopQA TechQA Avg
Llama-3.2-1B-Instruct 0.678 0.483 0.594 0.494 0.532 0.5562
FullLoRA 0.728 0.527 0.666 0.528 0.556 0.601
DiffLoRA-64 0.629 0.52 0.611 0.451 0.39 0.5202
DiffLoRA-64- λ= 0.1 0.638 0.511 0.619 0.495 0.407 0.534
DiffLoRA-32 0.585 0.511 0.621 0.479 0.344 0.508
DiffLoRA-32 + GN 0.025 0.097 0.031 0.041 0.059 0.0506
DiffLoRA-32 + Tulu3 0.594 0.413 0.584 0.466 0.339 0.4792
8