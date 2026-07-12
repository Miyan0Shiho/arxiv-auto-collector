# Privilege and confidentiality in generative AI workflows

**Authors**: Václav Janeček, Thomas Melham

**Published**: 2026-07-06 14:38:58

**PDF URL**: [https://arxiv.org/pdf/2607.05479v1](https://arxiv.org/pdf/2607.05479v1)

## Abstract
Generative AI (GenAI) systems store and process client data in three distinct ways: in the model's parameters through training and memorisation, in the context window during a live session, and in knowledge databases for retrieval-augmented generation (RAG). Each mode creates different and often counter-intuitive risks to confidentiality and legal professional privilege, and each calls for specific governance responses. Drawing on the first English and American decisions to address privilege and generative AI, UK and Munir v Secretary of State for the Home Department and United States v Heppner, on the orthodox privilege authorities against which those decisions must be read, and on recent computer science research, we explain the three modes of data storage and processing in terms accessible to practitioners and analyse the legal consequences of each. We then situate the analysis within the regulatory framework governing solicitors in England and Wales and within the ordinary principles of professional negligence, arguing that the standard of effective information governance (and with it the benchmark against which negligence and misconduct will be measured) is changing. Although we write primarily for SRA-regulated practitioners, our data-governance analysis is framed to extend to any jurisdiction in which the protection of privilege or professional secrecy depends on demonstrable confidentiality. The ultimate aim of this article is to help legal services professionals understand salient data leakage risks in GenAI systems and thereby facilitate a more responsible deployment of GenAI on client data and other sensitive material.

## Full Text


<!-- PDF content starts -->

 Janeček and Melham, Privilege and confidentiality in generative AI workflows  (SMUR, 22 June 2026, v10.0 ) 
1 Privilege and confidentiality in generative AI workflows  
Václav Janeček, University of Bristol Law School   
Thomas Melham, University of Oxford, Department of Computer Science  
Acknowledgment. This article arose out of Dr Janeček’s research at the IGSG -CNR in Flore nce 
(Leverhulme International Fellowship) and the authors’ interdisciplinary teaching for the Oxford 
LawTech Education Programme.  
Abstract.  Generative AI (GenAI) systems store and process client data in three distinct ways: in 
the model’s parameters through training and memorisation, in the context window during a live 
session, and in knowledge databases for retrieval -augmented generation (RAG) . Each mode 
creates different and often counter -intuitive risks to confidentiality and legal professional 
privilege, and each calls for specific  governance responses. Drawing on the first English and 
American decisions to address privilege and generative AI, UK and Munir v Secretary of State 
for the Home Department  and United States v Heppner , on the orthodox privilege authorities 
against which those decisions must be read, and on recent computer science research, we explain 
the three modes of data storage and processing in terms accessible to practitioners and analyse 
the legal consequences of each. We then situate the analysis within the regulatory framework 
governing solicitors in England and Wales and within the ordinary principles of professional 
negligence, arguing that the standard of effective information governance (and with it the 
benchmark against which negligence and misconduct will be measured) is changing. Although 
we write primarily for SRA -regulated practitioners, our data -governance analysis is framed to 
extend to any jurisdiction in which the protection of privilege or professional secrecy depends on 
demonstrable confidentiality.  The ultimate aim of this article is to help legal services 
professionals understand  salient data leakage risks in GenAI systems  and thereby facilitate a 
more responsible deployment of GenAI on client data and other sensitive material . 
Introduction  
The familiar phrases —‘rubbish in, rubbish out’  and ‘bias in, bias out’ —capture a basic truth: the 
capabilities of AI systems, and of large language models (LLMs) in particular, are closely tied to 
the quality of the data from  which they are built and to which they are given access. But t he 
relationship between data and generative AI  runs deeper than output quality, and it is this deeper 
relationship that should concern the legal profession. Wherever client data goes, the lawyer’s 
professional obligations go with it . And in generative AI systems, data moves in ways that are 
often invisible by design.  
In this article, we examine three distinct ways in which data —including confidential client data 
and personal data —is stored and processed in generative AI systems: (1) model memorisation; 
(2) the context window; and (3) retrieval -augmented generation (RAG). In relation to each, we 
identify data privacy and confidentiality risks that are easy to miss even for technically informed  
users and  explain what practitioners can do about them in day -to-day work —with a view to 
maintain ing data confidentiality and protect ing legal privilege that may pertain to the data .  

 Janeček and Melham, Privilege and confidentiality in generative AI workflows  (SMUR, 22 June 2026, v10.0 ) 
2 The headline message is twofold. First, the risks are manageable , but only if one understands the 
difference between the three modes of data processing. A lawyer who cannot distinguish 
between data that is ‘baked into’ a model, data that transits a provider’s infrastructure during a 
session, and data that persists in a vector database, cannot make sound judgements about 
confidentiality, privilege or data protection compliance. Second, because generative AI systems 
pose new and unintuitive data leakage risks, information governance protocols need to be 
adjusted accordingly: the standard of what counts as effective data governance —and, by 
extension, what a reasonably competent practitioner must do to protect privileged and 
confidential material —is changing.  
The legal context sharpened considerably between late 2025 and early 2026. In November 2025 
the Upper Tribunal warned  that uploading confidential client documents into a consumer -grade 
AI tool ‘is to place this information on the internet in the public domain, and thus to breach client 
confidentiality and waive legal privilege’.  1 This guidance had been elevated into its formal 
headnote, though strictly it was an obiter  comment , as discussed  below. Three months later, in 
United States v Heppner , the Southern District of New York held —as ‘a question of first 
impression nationwide’ —that a criminal defendant’s exchanges with a consumer -grade AI 
chatbot were protected neither by attorney –client privilege nor by the work product doctrine.2 
And in April 2026 the Chancellor of the High Court, Sir Colin Birss, devoted a speech to legal 
professional privilege in the age of AI, stressing that AI does not change the legal test for 
privilege but that confidentiality —its precondition —cannot be assumed when public AI systems 
are used.3 The doctrinal framework is familiar; what is unfamiliar is the technology to which it 
now must  be applied.  The purpose of this article is to make that technology visible.  Some issues 
have already been analysed in emerging practitioner -orientated and academic writings ,4 but this 
article goes beyond those studies by focusing on the  technology understanding and  AI literacy 
needs  that, if unmet,  are likely to end up being the source of professional negligence in this area 
going forward.   
A note on terminology is needed at the outset, because loose labels have already caused judicial 
confusion. We distinguish  between  consumer -grade  tools —services used on standard public 
terms, under which the provider may typically retain inputs, use them for model training (subject 
to opt -outs), and disclose them in defined circumstances —and enterprise -grade  deployments, in 
which inputs are processed under a negotiated contractual framework that typically excludes 
training use, restricts retention, and imposes confidentiality obligations on the provider. The 
 
1 UK and R (on the application of Munir) v Secretary of State for the Home Department (AI hallucinations; 
supervision; Hamid)  [2026] UKUT 81 (IAC) at [60] (hereafter Munir ). 
2 United States v Heppner  No 1:25 -cr-00503 -JSR (SDNY, 17 February 2026) (Rakoff J).  
3 Sir Colin Birss, ‘Legal Professional Privilege in the Age of AI’ (Speech to the City of London Law Society, 22 
April 2026), available at https://perma.cc/FX2Z -GL3Q . 
4 Eg, T Whittaker and S Bourton ‘AI and Legal Privilege’ (Burges Salmon, October 2025), available at 
https://perma.cc/M77Q -HMU9; P Fields ‘When You Upload, What Have You Disclosed? AI, Confidentiality and 
the Embedded Document’ (Barrister Magazine, 17 June 2026), available at https://perma.cc/VPR8 -4TM5; J Perlin  
‘Client Confidentiality and Generative AI’ (2027) 40 Harv JL & Tech (forthcoming) , available at 
https://ssrn.com/abstract=6227838 . 

 Janeček and Melham, Privilege and confidentiality in generative AI workflows  (SMUR, 22 June 2026, v10.0 ) 
3 distinction is defined by the data -handling terms and the technical controls behind them, not by 
the product’s brand, price, or the openness of its source code. As will be seen, the failure to draw 
the line in this way  is the principal defect in the early case law , which instead employs labels 
such as ‘open  source’  or ‘closed  source’ AI tools . 
The article proceeds as follows. Section 2 addresses memorisation; section 3 the context 
window, including disclosure to model providers and its consequences for privilege; section 4 
RAG pipelines; and section 5 the professional stakes, namely the regulatory framework, the 
negligence analysis, and the extent to which the argument travels to other jurisdictions. Figure 1 
provides an overview of the three modes and their persistence characteristics.  
 
Figure 1: Three modes of data persistence in generative AI systems  
Memorisation: The model’s permanent memory  
How data becomes part of the model  
Memorisation is the most permanent of the three modes of data processing considered here, and 
the most familiar. When an LLM is trained, the training algorithm  processes hundreds of billions 
of words —books, websites, code, academic papers, legal documents. Statistical patterns in this 
material are encoded (‘baked’) into the model’s parameters: hundreds of billions of numerical 
values that constitute the model’s long -term memory . Technically speaking , model parameters 
are not computer memory in the traditional sense (a storage  space for writing/reading 
information ) as the information is ‘written’ into model parameters during mode l training, while it 
is ‘read’ during inference  sessions . Crucially, what gets baked in is not only general linguistic 
patterns. Some specific, verbatim content can be  encoded , distributed in non -trivial ways across 
those parameters . And this can be extracted from the model with the right prompting techniques. 
Computer science researchers have repeatedly demonstrated that LLMs can ‘ memorise ’ parts of 


 Janeček and Melham, Privilege and confidentiality in generative AI workflows  (SMUR, 22 June 2026, v10.0 ) 
4 their training data in this way and can be induced to emit them verbatim, including personally 
identifiable information.5 That, very roughly, is model memorisation.  
A common assumption among law professionals is that a generative AI tool works like a search 
engine: query in, result out, and nothing more. But i n certain circumstances, what a user types in 
can become part of the model’s permanent memory.  This mi ght happen when  what the user types 
in is stored  as verbatim text. And this can then be used as further training data in a subsequent 
training run for the model . Once the stored data is used for model training, the user cannot undo 
this: t here is no delete button on a trained AI model .6 As a result , the memorised sequence may 
be disclosed to other users of that model. This is broadly the technical reason for the argument 
that feeding data into consumer AI tools is like putting those data in the public domain . 
The European Data Protection Board has recognised the data protection significance of this 
point: in Opinion 28/2024 it concluded that AI models trained on personal data cannot 
automatically be regarded as anonymous; whether personal data can be extracted from a model 
must be assessed case by case, having regard (among other things) to the likelihood of 
regurgitation of training data.7 In other words, the regulator s treat the model itself as potentially 
containing personal data  and this  view maps onto the confidentiality analysis that follows.  
The consumer tool problem: Samsung  
The best -known real -world example comes from a different industry. In early 2023, Samsung 
Electronics lifted an internal ban on ChatGPT; within weeks, three separate incidents had 
occurred in which engineers reportedly pasted confidential material —including proprietary 
source code and internal meeting notes —into the tool.8 Under OpenAI’s terms applicable to 
consumer products at the time, that data could be used to train OpenAI’s models.9 The engineers 
presumably did not think of themselves as training someone else’s AI, but the information they 
put in was eligible for training use and therefore capable of being memorised  and produced 
verbatim by a model accessible to the public.  
The obvious lesson  that confidential data should not be pasted into a consumer -grade AI tool  is 
correct so far as it goes. Indeed, for consumer -grade tools the risk is at least relatively 
transparent: the terms of service generally make clear that input data may be used for training , 
unless the user opts out.  
 
5 N Carlini and others ‘Extracting Training Data from Large Language Models’ in Proceedings of the 30th USENIX 
Security Symposium  (2021) ; J Hayes and others, ‘Measuring Memorization in Language Models via Probabilistic 
Extraction’ in Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for 
Computational Linguistics: Human Language Technologies (Vol 1)  (2025).  
6 Eg, S Liu and others ‘Rethinking machine unlearning for large language models’ (2025) 7 Nat Mach Intell  181. 
7 European Data Protection Board ‘Opinion 28/2024 on certain data protection aspects related to the processing of 
personal data in the context of AI models’ (adopted 17 December 2024), esp paras 29 –58. 
8 ‘Concerns become reality… Samsung Electronics “misuse” emerges one after another as soon as the ChatGPT ban 
is lifted’ The Economist  (Korea edition) (30 March 2023) , available at https://perma.cc/3UL4 -9CUV . 
9 OpenAI ‘Privacy Policy’, openai.com/policies/privacy -policy. As of June 2026, c onsumer -tier inputs may be used 
for model training subject to opt -out; API and enterprise -tier inputs are excluded from training by default.  

 Janeček and Melham, Privilege and confidentiality in generative AI workflows  (SMUR, 22 June 2026, v10.0 ) 
5 Enterprise tools, fine -tuning and irreversibility  
The harder question concerns enterprise -grade tools, where the provider contractually commits 
not to use customer data to train models available to other users.10 In such a setting, many legal 
organisations may still want their data to be used to fine -tune11 a private instance of an AI model, 
for example to control running cost, boost performance, or attune the model to their preferences. 
Here the memorisation risk is more subtle. It is less about whether the data enters a publicly 
accessible model —under such terms, it should not —and more about irreversibility.   
Once a model is fine -tuned on specific data  (for example  private  or client -specific data ), that 
information cannot be selectively removed. Although data subjects have a right to erasure under 
the UK GDPR and the EU GDPR, honouring that right in respect of a trained model is 
technically very difficult, if not practically impossible: ‘machine unlearning’ remains an 
unsolved research problem at production scale.12 And one must not lose sight of the prior 
question: a valid legal basis would be needed for using client data (including personal data) to 
train an AI model in the first place.  
Two consequences deserve emphasis because they factually affect the confidentiality question  
and resist more traditional data governance frameworks. First, a privately fine -tuned model is 
itself a new repository of confidential information: a model fine -tuned on the matter files of 
various clients irreversibly commingles those  clients’ confidential information, and whereas 
conventional information barriers assume that documents can be segregated, a fine -tuned model 
cannot technically instantiate such segregation . Secondly, the assessment is temporal: a model 
fine-tuned today will eventually be judged against the extraction techniques of tomorrow, should 
it remain in service  (or in backups ) for years.  
This bears on a question many in -house teams and firms are actively considering: should they 
fine-tune a private model on their own precedents, matter files and house style? In most cases, 
not yet, because the risk –reward balance is unfavourable: the main benefits sought from fine -
tuning can be substantially achieved through in -context learning instead, ie, by giving the model 
what it needs at the point of use rather than baking it permanently into the model  and risking the 
collapsing of some information barriers . How that works, and what risks it carries, is the subject 
of the next section.  
The memorisation risk therefore reduces to two practical rules: first, know the tool’s training 
policy before using it for any client or otherwise sensitive work; secondly, hold off fine -tuning 
on client or personal data until the erasure problem has a workable technical solution. These 
 
10 See eg OpenAI, openai.com/enterprise -privacy/ (ChatGPT Enterprise commitments); Anthropic ‘Privacy Policy’, 
anthropic.com/ legal/ privacy  and data processing practices disclosed via Anthropic Trust Center, 
https://trust.anthropic.com/ . 
11 Fine-tuning is the process of further training a pre -trained model on a smaller, task - or domain -specific dataset to 
adapt model parameters toward a particular objective while retaining the general capabilities acquired during pre -
training.  
12 UK GDPR, art 17; Regulation (EU) 2016/679, art 17; G -O Barbulescu and P Triantafillou ‘To Each (Textual 
Sequence) Its Own: Improving Memorized -Data Unlearning in Large Language Models’ in Proceedings of the 41st 
International Conference on Machine Learning  (2024), PMLR :235:3003 -3023.  

 Janeček and Melham, Privilege and confidentiality in generative AI workflows  (SMUR, 22 June 2026, v10.0 ) 
6 rules are consistent with  (indeed, they supply the technical rationale behind ) the position now 
taken by the professional bodies, namely the Law Society’s guidance that no confidential data 
should be put into free online generative AI services, and the SRA’s identification of training -
data transfer as a particular threat to confidentiality and privilege.13 Firms that prohibit 
consumer -grade tools for client work, and prohibit client data from being used for AI model 
training (whether the resulting model is public or private), are applying the same logic.  
The context window: In -session working memory  
In-context learning is not training  
If memorisation is the model’s permanent memory, baked in at training time, the context window 
is its short -term working memory, operating at inference time —during the live session.  
Let us begin this section with a short point about terminology, because it routinely confuses 
lawyers reading vendor documentation: the ability of an LLM -based system to make use of 
material supplied in the session is often called ‘in -context learning’, but it is not machine 
learning in the training sense, and the model’s parameters do not change.  In-context learning 
simply means giving the model what it needs for a task within the conversation itself —a contract 
for review, background facts for a summary,  or a policy document for analysis. The model takes 
in what it has been given, works with it for that session, and produces an output  based on this 
temporarily ‘learned’ context . No information is permanently baked into the model, and when 
the session ends , the model itself retains nothing.  
Such in -context ‘learning’  is a powerful  function that is very useful for lawyers, and in many 
ways much safer than fine -tuning, because the data does not persist in the model. But while the 
model itself retains nothing , it does not mean the context window is risk -free. As will be shown, 
at least t hree categories of risk arise: disclosure to the model provider; shared and persistent 
context; and prompt injection. Before turning to those, one practical limitation of context 
windows deserves attention, because it goes to the reliability of AI -assisted work product rather 
than to confidentiality: the long -input problem.  
Context rot and the long -input problem  
Context windows are measured in tokens (a token is roughly three -quarters of a word). Modern 
windows can be  large —typically of the order of a hundred thousand tokens, ie, the length of a  
short  book, or the documentary material for a very simple piece of litigation —but they are not 
infinite.14 What happens when the uploaded material is bigger than the window? Some tools 
refuse the upload and report  the file is too large. That is the transparent  response. Others process 
the input in batches  or compress , thereby imposing  some kind of artificial dividing line or even 
content reformulation onto the uploaded material. Yet o thers silently truncate  the input : they 
 
13 The Law Society ‘Generative AI - the essentials’ (guidance, updated September 2025) , available at 
https://perma.cc/9UMX -9RGQ ; SRA ‘Risk Outlook report: the use of artificial intelligence in the legal market’ (20 
November 2023) , available at https://perma.cc/C5XS -ZCN5 . 
14 The figures are indicative for typical professional deployments in 2026; advertised maxima vary by model and 
tier, and effective usable context is generally smaller than the advertised limit.  

 Janeček and Melham, Privilege and confidentiality in generative AI workflows  (SMUR, 22 June 2026, v10.0 ) 
7 process the first part of the material and quietly ignore the rest. The model then gives a perfectly 
confident -seeming  answer based on an incomplete or skewed picture, with no flag that it never 
processed  the whole file  or that it did some modification to the uploaded  material . Figure 2 
illustrates the failure mode.  
 
Figure 2: Silent truncation of long inputs  
The consequences are real: a contract review that misses the final quarter of the document, or a 
risk summary that fails to flag a clause buried near the end  is justifiably a source of concern on 
legal tasks which thrive on long -form document s. Industry e xperts  call this family of failures 
‘context rot’: the quiet degradation of a model’s performance and  reliability in processing  its 
context window as inputs grow longer or the window fills up.15 The phenomenon is broader than 
truncation alone. Models seem to attend less reliably to material in the middle of long inputs than 
at the beginning or end  (positional bias in long contexts) ;16 reasoning quality degrades as input 
volume grows even where the relevant information is present; and where a system compresses or 
summarises documents to fit the window, the compression may strip exactly the language  on 
which legal meaning depends.17 The output looks as polished as ever , but without verification 
one cannot trust that it is based on the whole document.  
 
15 H Wu ‘Silent but Deadly - Context Rot Problems in Legal’ (Syntheia, 27 March 2026), available at 
https://perma.cc/TY64 -EUBF ; K Hong and others ‘ Context Rot: How Increasing Input Tokens Impacts LLM 
Performance ’ (Chroma Technical Report, July 14, 2025), available at https://perma.cc/43YF -DP88.  
16 N F Liu and others ‘Lost in the Middle: How Language Models Use Long Contexts’ (2024) 12 Transactions of 
the Association for Computational Linguistics  157. 
17 ibid (positional degradation) ; M Elaraby and D Litman ‘ARC: Argument Representation and Coverage Analysis 
for Zero -Shot Long Document Summarization with Instruction Following LLMs’ in Proceedings of the 19th 
Conference of the European Chapter of the Association for Computational Linguistics  (Vol 1)  (2026) 
(demonstration of the issues on legal data : CANLII ); H Jiang and others ‘LLMLingua: Compressing Prompts for 
Accelerated Inference of Large Language Models’ in Proceedings of the 2023 Conference on Empirical Methods in 
Natural Language Processing (2023) . 


 Janeček and Melham, Privilege and confidentiality in generative AI workflows  (SMUR, 22 June 2026, v10.0 ) 
8 The danger for legal practice is that this failure mode is invisible to traditional supervisory 
review: the error does not resemble anything a supervising partner has been trained to catch  and 
yet this error can compromise the quality of the output . Moreover, if the system silently pre-
processes the input material to fit it within the context window , the pre -processing itself may 
pose risk s to data confidentiality, depending on what technical methods are used to achieve the 
compression goal.  
The practical responses are straightforward: when working with long documents, test the tool 
with a reference to something near the end of the document; for very large document sets, 
RAG —discussed in section 4 —may be  a safer architecture than squeezing everything into one 
context window ; and for enterprise tools it is best to ask directly for details  about how the AI tool 
in question  tackles the context rot challenge . Technologists are working on a general solution to 
this problem , but it is far from being resolved right now .18 
Disclosure to the model provider: The privilege question  
We can now turn to the confidentiality risks proper. Everything put into the context window is 
visible to the model and shapes its outputs during the session. When the session ends, the 
window is cleared. That is the theory. In practice, data from the context window can travel 
further than the user may expect.  
Every major consumer -grade AI tool routes  user input through the provider’s infrastructure. So 
do the leading legal AI platforms: at the time of writing, prominent legal tools such as Harvey 
and Legora sub -process data through the major model providers —Harvey through OpenAI and 
others, Legora through Azure -hosted OpenAI models  and other s.19 The leading vendors have, to 
their credit, built their offerings around precisely the contractual architecture we describe in this 
article: Harvey contractually guarantees that customer inputs, outputs and uploaded documents 
are not used to train the underlying models, requires zero data retention from its upstream model 
providers, and offers regional data residency, audit logs and customer -set retention policies; 
Legora  makes equivalent no -training and no -retention commitments, holds an ISO 42001 AI -
governance certification, and provides configurable retention with deletion of all data at contract 
end.20  
But unless the organisation runs its own sealed -off instance of a model —for example via local 
hosting and deployment of an AI model  on its exclusively controlled  computers —the data still 
 
18 A Modarressi and others ‘NoLiMa: Long -Context Evaluation Beyond Literal Matching ’ in Proceedings of the 
42nd International Conference on Machine Learning  (2025) PMLR :267:44554 –44570 ; J Su and others ‘RoFormer: 
Enhanced Transformer with Rotary Position Embedding’ (2024) 568 Neurocomputing  127063; D Aumiller and 
others ‘Structural Text Segmentation of Legal Documents’ in Proceedings of the Eighteenth International 
Conference on Artificial Intelligence and Law  (2021) 2 . 
19 See, eg, Microsoft ‘How Legora is transforming the legal workspace using Azure OpenAI Service’ (Microsoft 
customer stories , 27 May 2025 ), available at https://perma.cc/5X6Y -2SFC ; Legora  ‘Trust Center’ 
(security.legora.com/); Harvey ‘Security’ and ‘Trust Center ’ (harvey.ai/security  and trust.harvey.ai/ ). 
20 Harvey , ‘Your Data, Your Control: How Harvey Manages Customer Data’ ( 20 March 2026 ), available at 
https://perma.cc/J9FE -BQR9 ; Legora ‘Security and compliance’ (legora.com/security). Both platforms publish SOC 
2, ISO 27001  and ISO 42001  attestations , for example.  

 Janeček and Melham, Privilege and confidentiality in generative AI workflows  (SMUR, 22 June 2026, v10.0 ) 
9 has to go somewhere to be processed; the prompt ultimately passes through the AI model on a 
provider’s systems, and the protection depends on these commitments holding, and being 
verifiable, down the chain.  
The legal question is how the law characterises this transit  of data to the model provider . Is it a 
disclosure to a third party? Does it destroy confidentiality? Does it waive privilege?  
It is worth pausing on why this question is not simply a re -run of the cloud computing debates of 
the 2010s. Privileged material has always passed through third -party infrastructure —postal 
services, email servers, hosted document management systems —and the profession has rightly 
proceeded on the basis that confidential transit through a service provider bound by 
confidentiality does not destroy privilege. Three features distinguish generative AI transit. First, 
under consumer -grade terms the provider may acquire use rights over the content —most 
importantly, the right to train models on it —which no postal or hosting analogy parallels. 
Training converts transient transit into potential permanent incorporation (section 2). Second, the 
provider’s systems do not merely store or carry the material but process  it: the content shapes 
outputs, logs, safety -review queues and caches in ways that multiply the points of persistence. 
Thirdly, the resulting artefacts —chat histories, cached responses, embeddings —are themselves 
new documents, generated partly or entirely outside the lawyer’s systems and outside the client’s 
control. Whether these differences are legally decisive will depend on the terms in each case; but 
they explain why the question deserves fresh analysis rather than straightforward extension of 
the cloud -computing debates . 
The doctrinal baseline  
Confidential communications between lawyers and clients attract special protection —legal 
professional privilege in England and Wales, attorney -client privilege in the United States, 
professional secrecy in civilian systems. The labels and doctrinal details vary, but the underlying 
principle is close to universal. In England and Wales, legal professional privilege is a 
fundamental right grounded, on one influential account, in the rule of law itself, and once 
established it is absolute: it cannot be overridden by some countervailing public interest in 
disclosure.21 Legal advice privilege protects confidential lawyer -client communications made for 
the purpose of giving or obtaining legal advice, understood broadly as the continuum of 
communications in the relevant legal context;22 litigation privilege protects confidential 
communications with third parties whose dominant purpose is the conduct of litigation in 
reasonable contemplation.23 Both limbs extend to in -house lawyers as a matter of English law,24 
though not in EU competition investigations —a point to which we return.  
 
21 Three Rivers District Council v Governor and Company of the Bank of England (No 6)  [2005] 1 AC 610 (HL) at 
[34], per Lord Scott, crediting Professor Zuckerman’s rule -of-law rationale. On the absolute character of the 
privilege once established, see R v Derby Magistrates’ Court, ex p B  [1996] 1 AC 487 (HL).  
22 Balabel v Air India  [1988] Ch 317 (CA) , [1988] 2 WLR 1036 . 
23 Waugh v British Railways Board  [1980] AC 521 (HL); and see R (on the application of Jet2.com Ltd) v Civil 
Aviation Authority [2020] EWCA Civ 35, [2020] 2 QB 1027 (consolidating the dominant purpose jurisprudence ); 
Serious Fraud Office v Eurasian Natural Resources Corpn Ltd  [2018] EWCA Civ 2006, [2019] 1 WLR 791 (on 
reasonable contemplation of litigation ). 

 Janeček and Melham, Privilege and confidentiality in generative AI workflows  (SMUR, 22 June 2026, v10.0 ) 
10 Four features of the English doctrine frame the AI analysis. First, the courts have declined to 
extend privilege beyond the lawyer –client relationship by reference to the function rather than 
the status of the adviser: that is the lesson of R (Prudential) v Special Commissioner of Income 
Tax, in which the Supreme Court refused to extend legal advice privilege to tax advice given by 
accountants.25 Any argument that interactions with an AI system giving ‘legal advice’ should 
themselves attract privilege would have to confront Prudential  head on.  
Second, confidentiality and privilege are related but structurally different protections, and the 
difference carries most of the analytical weight in the AI context. Confidentiality is a 
precondition of privilege: as Lord Scott put it in Three Rivers (No 6) , unless the communication 
for which privilege is sought is a confidential one, ‘there can be no question of legal advice 
privilege arising’; the confidential character of the communication ‘is not by itself enough to 
enable privilege to be claimed but is an essential requirement’.26  
All privileged communications are therefore confidential . But the converse does not hold, and 
the two protections operate differently. Privilege is a shield against compulsion : an exception to 
the general obligation to produce documents or answer questions —and almost absolute, 
admitting no balancing against competing interests.27 Confiden tiality , by contrast, is a right to 
restrain voluntary  use or divulgence: broader in coverage, but heavily qualified —it concedes  to 
compulsory process and may concede  to the public interest.28 As Millett J observed in Price 
Waterhouse v BCCI , privilege is normally an answer to compulsory disclosure, whereas 
‘confidentiality alone affords protection only against voluntary disclosure without the consent of 
the person to whom the duty of confidentiality is owed’.29 
Within that architecture, two doctrinally distinct routes can defeat the privilege: loss of 
confidentiality , which destroys the precondition (once information is genuinely in the public 
domain, there is nothing left for privilege to bite on), and waiver , ie, conduct of the privilege -
holder treating the material in a manner inconsistent with maintaining the privilege. The two are 
sometimes  conflated —including, as we will see, in the first GenAI-related  decision —and the 
conflation matters because their consequences differ: loss of confidentiality is total, while waiver 
can be partial and purpose -limited.  
 
24 Alfred Crompton Amusement Machines Ltd v Customs and Excise Commissioners (No 2)  [1972] 2 QB 102 (CA) 
at 129, per Lord Denning MR; affd [1974] AC 405 (HL).  
25 R (Prudential plc) v Special Commissioner of Income Tax  [2013] UKSC 1, [2013] 2 AC 185.  
26 Three Rivers (No 6) , supra  n 21 at [24]  per Lord Scott.  
27 PCCW -HKT Telephone Ltd v Aitken  (2009) FACV No 27 of 2008 (HKCFA) at [59], per Lord Hoffmann NPJ, 
quoted in Passmore, Privilege  (5th edn) para 1 -415. 
28 Attorney General v Jamaican Bar Association  (PC), per Lords Briggs and Hamblen, quoted in Passmore, ibid para 
1-416, citing R v Derby Magistrates’ Court, ex p B  [1996] AC 487.  
29 Price Waterhouse v BCCI Holdings (Luxembourg) SA  [1992] BCLC 583, per Millett J, quoted in Passmore, ibid 
para 1 -420. 

 Janeček and Melham, Privilege and confidentiality in generative AI workflows  (SMUR, 22 June 2026, v10.0 ) 
11 Third, and crucially for AI systems, limited disclosure to a third party under obligations of 
confidence does not, without more, destroy either confidentiality or privilege. Privileged material 
is routinely placed in the hands of translators, e -disclosure providers and cloud -hosting providers 
without anyone suggesting that the privilege is thereby lost . The Privy Council has confirmed 
that even deliberate disclosure for a limited purpose, on terms preserving confidentiality, does 
not amount to a general waiver;30 and communications of the substance of privileged advice to 
third parties, where confidentiality is preserved, likewise remain protected.31 This line of 
authority is, in our view, a strong  support  for the proposition that transit of privileged material 
through an enterprise -grade AI provider’s infrastructure should not in itself imperil privilege. 
The question in every case is whether the terms and the technical reality genuinely preserve 
confidentiality.  
Fourth, even where privileged material escapes, the law does not always treat the protection as 
irretrievably ‘gone’. It is the rights arising from the relationship of confidence —not privilege 
itself —that entitle a client to restrain third -party interference with privileged material,32 and in 
litigation the court’s permission is required before inadvertently inspected privileged material 
may be deployed.33 The remedial position is imperfect  because an injunction cannot truly un -ring 
a bell, still less claw back data that has entered a model’s training corpus . But the absolutist 
language  of some early commentary (‘the privilege is gone’) overstates the doctrinal position in 
at least some data leakage  scenarios.  
It helps to separate three scenarios that are often run together. In the first, a litigant consults a 
public AI tool directly for what is functionally legal advice, with no lawyer involved.  The 
Chancellor of the High Court has described typing a consumer -law question into the public 
version of ChatGPT and receiving a structured analysis ending with a request for further 
instructions .34 Such an  interaction may look exactly like an exchange between client and adviser , 
but no privilege attaches, because the ‘adviser’ is not a legal professional, and Prudential  stands 
in the way of any functional extension. In the second, a client uses an AI tool on their own 
initiative alongside  an existing retainer and feeds the results to their lawyer; that is Heppner , 
discussed below, where material unprotected when created did not acquire protection by being 
routed to counsel. In the third, the lawyer uses an AI tool as part of producing advice for the 
client; here the analysis is the most forgiving —provided the system is secure —because a lawyer 
consulting computational resources is not obviously different from a lawyer consulting a 
textbook or an online database. The three scenarios attract different analyses  and conflating them 
may produce both overconfidence and overcaution.  
 
30 B v Auckland District Law Society  [2003] UKPC 38, [2003] 2 AC 736 at [68] –[71].  
31 USP Strategies plc v London General Holdings Ltd  [2004] EWHC 373 (Ch) at [9] –[21], per Mann J.  
32 Passmore, Privilege  (5th edn) paras 1 -417 and 1 -421, discussing Lachaux v Independent Print Ltd  [2015] EWHC 
3677 (QB), affd in [2017] EWCA Civ 1327.  
33 CPR r31.20; Al Fayed v Commissioner of Police of the Metropolis  [2002] EWCA Civ 780 at [16], setting out the 
governing principles.  
34 Supra  n 3, paras 15 –19. 

 Janeček and Melham, Privilege and confidentiality in generative AI workflows  (SMUR, 22 June 2026, v10.0 ) 
12 Finally, Sir Colin Birss ’s recent e xtrajudicial speech  made a nother point of practical importance. 
AI does not change the legal test: legal advice privilege and litigation privilege remain grounded 
in confidentiality and, where relevant, a dominant litigation purpose. Protecting privilege in the 
age of AI is therefore primarily a matter of governance, not technology. And  where it is the 
lawyer  who uses a secure AI system to help provide advice to the client, it is ‘hard to see’ how 
that use could impact the client’s privilege: lawyers have always been entitled to consult 
textbooks and online resources without undermining privilege.35 
Munir: Consumer -grade tools and waiver  
In November 2025 , the Upper Tribunal (Immigration and Asylum Chamber) addressed for the 
first time in England and Wales what happens to privilege when privileged material is uploaded 
to a consumer -grade AI tool. UK and R (on the application of Munir) v Secretary of State for the 
Home Department36 began as two conjoined Hamid  proceedings concerning AI -generated 
hallucinated citations; no privilege dispute was before the tribunal, no party argued the point, and 
the relevant passage cites no authority. The privilege statement is therefore obiter. That status 
should be kept firmly in view, because the tribunal elevated the warning into its formal headnote:  
Uploading confidential documents into an open -source AI tool, such as ChatGPT, is to 
place this information on the internet in the public domain, and thus to breach client 
confidentiality and waive legal privilege, and any such conduct might itself warrant 
referral to the regulatory body and should, in any event, be referred to the Information 
Commissioner’s Office.37 
On this view the privilege is not weakened or qualified by the upload; it is gone. The tribunal 
contrasted what it called ‘open -source ’ tools, such as the free version of ChatGPT, with ‘closed 
source AI tools which do not place information in the public domain, such as Microsoft Copilot’, 
which it said are available for tasks such as summarising ‘without these risks’.38 
The tribunal’s analysis is, with respect, doubly imprecise. The terminology is technically 
misinformed: ChatGPT is not ‘open -source’ in any accepted sense —its weights and code are 
proprietary —and the relevant distinction is the consumer -grade/enterprise -grade line drawn in 
section 1, not openness of source. More importantly, the legal characterisation conflates the two 
doctrinal routes distinguished above. Material pasted into a consumer -grade tool is not literally 
placed ‘on the internet in the public domain’; it is disclosed to an identified provider on terms 
that permit retention, training use and onward disclosure. The better analysis is not loss of 
confidentiality through publication, but that disclosure on such terms is inconsistent with 
maintaining confidentiality as against the provider, and so may destroy the reasonable 
expectation  of confidentiality on which privilege depends —or, where the client authorises the 
upload, may amount to waiver. On the limited -disclosure authorities, by contrast, an upload to an 
 
35 Ibid, esp paras 20 -26. 
36 Supra  n 1. 
37 Ibid, headnote para 4 , see also [20] and [60].  
38 Ibid at [21].  

 Janeček and Melham, Privilege and confidentiality in generative AI workflows  (SMUR, 22 June 2026, v10.0 ) 
13 enterprise -grade deployment under confidentiality obligations sits on the same side of the line as 
the e -disclosure provider or the cloud host, and should not in our view , in itself , defeat 
privilege.39 
Setting the imprecision aside, the tribunal’s underlying sentiment is sound: enterprise -grade 
systems operating within a secure contractual framework pose far fewer risks. Even then the 
answer is incomplete, because the data is typically still transmitted to an upstream model 
provider for processing, and the protection depends on the sub -processing terms holding all the 
way down the chain  of sub -processors . In any event, t he law is not fully settled; the relevant 
passage in Munir  is unreasoned obiter guidance from a tribunal; and there is  thus ample scope for 
the position to develop.  
Commentators have likewise observed that the tribunal’s blunt public -versus -private line is too 
absolute.40 On this view, which we share, the real question is not the label a system carries but 
whether the user can reasonably expect confidentiality given the contractual and technical reality 
of how the data is handled; if a platform genuinely prevents external access to inputs, retention, 
and retraining on those inputs, using it should not in itself waive privilege.  
Heppner: The terms of use decide  
Around the same time, the Southern District of New York reached a convergent result by a 
different route. In United States v Heppner ,41 the defendant, already under federal investigation, 
used the free consumer -grade tier of Anthropic’s Claude chatbot —on his own initiative, not at 
his lawyers’ direction —to generate documents outlining his defence strategy, which he then 
shared with his lawyers. After his arrest, the FBI seized some 31 such documents, and Heppner 
asserted privilege over them.  
Judge Rakoff held that the documents were protected neither by attorney -client privilege nor by 
the work product doctrine. Three strands of the reasoning matter here. First, the communications 
were not between client and attorney: ‘Because Claude is not an attorney,  … that alone disposes 
of Heppner’s claim of privilege’  (at p 5) . Second —and most instructive for present purposes —
there was no reasonable expectation of confidentiality, because the consumer terms the 
defendant had accepted permitted the provider to collect inputs and outputs, use them for 
training, and disclose them to third parties , including governmental authorities. Third, work -
product protection failed because counsel had neither prepared nor directed the preparation of the 
documents; material that is not protected when created does not become protected merely by 
later reaching a lawyer.42 
 
39 B v Auckland , supra  n 30. 
40 See eg G Johnston  ‘Legal privilege and AI under English law’ ( Juralio, 8 March 2026), available at  
https://perma.cc/V4FF -ARUN; T Whittaker and B Kolodziej ‘AI and privilege’ (Burges Salmon, 13 April 2026), 
available at  https://perma.cc/8A55 -U4V4 . 
41 Supra  n 2. 
42 Ibid at pp 4 -10 of the slip opinion.  

 Janeček and Melham, Privilege and confidentiality in generative AI workflows  (SMUR, 22 June 2026, v10.0 ) 
14 The analytical lesson echoes the critique of Munir : the ‘open’ versus ‘closed’ label is beside the 
point; what mattered were the specific terms the user had accepted, which defeated any 
reasonable expectation of confidentiality. American commentary has already questioned the 
decision’s categorical edge, pointing out that courts do not treat the use of Gmail or cloud 
storage as destroying confidentiality even though those providers can access user content.43 But 
the practical message on both sides of the Atlantic is the same: when client -related material is put 
into an AI system, the contractual and technical data -handling reality —not the marketing label —
determines whether confidentiality, and with it privilege, survives.  
Disclosure, discoverability and self -reporting  
What, then, does an enterprise -grade deployment with strict privacy controls require in practice? 
At minimum: a data processing agreement identifying the provider as processor; contractual 
exclusion of customer inputs and outputs from model training; zero or strictly time -limited 
retention of session data, including server -side logs; no human review of inputs save in defined 
abuse scenarios; encryption in transit and at rest; and clarity about where, geographically and 
organisationally, the data is processed —including whether the legal AI vendor sub -processes 
through an upstream model provider and on what terms. These questions should be asked, and 
the answers documented, before any client material enters the system: if a privilege challenge 
ever comes, the firm will need to evidence that its expectation of confidentiality was reasonable 
at the time of use.  
One exposure survives even perfect contractual terms, though, and the privilege/confidence 
architecture set out above explains why it matters. A provider’s contractual duty of confidence 
protects only against voluntary disclosure; it is no answer to compulsory process directed at the 
provider —a regulator’s statutory information demand, a foreign court’s production order, or a 
litigation preservation order over user session data in proceedings to which the law firm is a 
stranger.44 In such a scenario the only protection capable of resisting compulsion is the client’s 
privilege itself. That is a further, and perhaps the strongest, reason for ensuring that privilege 
survives data transit  through the GenAI tool : if confidentiality has been preserved throughout, 
the client has something to assert against the compelled provider; if not, contractual confidence 
alone is unlikely to hold the line.45 
The legal press has begun making  the same point against vendor reassurance. Commentary in 
Legal Futures argues that a firm cannot discharge its confidentiality obligations on vendor 
assurances alone: even where training is contractually excluded, sub -processor chains and 
internal review teams remain potential disclosure routes, inputs may be logged for 
 
43 E X Guo ‘ United States v Heppner ’ (Harvard Law Review Blog, 23 March 2026) , available at 
https://perma.cc/CML7 -S4N9 . 
44 Price Waterhouse v BCCI , supra  n 29. On the limited statutory carve -outs protecting privilege against regulators’ 
information -gathering powers, see Passmore, ibid para 1 -413. 
45 cf R (McKenzie) v Director of the Serious Fraud Office  [2016] EWHC 102 (Admin) , [2016] 1 WLR 1308 , 
discussed in Passmore, ibid para 1 -418: where material is taken under coercive powers, the safeguards required of 
the holder differ from those of a fiduciary —underscoring that the client’s strongest protection in third -party hands is 
the subsistence of the privilege itself.  

 Janeček and Melham, Privilege and confidentiality in generative AI workflows  (SMUR, 22 June 2026, v10.0 ) 
15 troubleshooting, cross -border processing creates exposure to foreign compulsory powers, and 
derivative information may persist in forms that are difficult to audit.46 We agree: the vendor 
commitments described above are real progress, but they are the beginning of due diligence, not 
its conclusion.  
Two further points complete the picture. The first arises independently of any waiver question: 
prompts entered into an AI system and the outputs it generates are ‘documents’ for disclosure 
purposes, since a document is anything in which information of any description is recorded.47 
Prompts and outputs produced by a lawyer using a secure, closed system in the course of giving 
advice may well be protected as privileged working papers, consistently with the Chancellor’s 
analysis above; prompts written by a client or by a non -lawyer in the business will usually enjoy 
no such protection, and a litigant’s exchanges with a chatbot about a live dispute may make 
uncomfortable reading on inspection. Document retention policies, litigation holds and 
disclosure protocols should accordingly address AI session data explicitly.  
The second is a regulatory point with a practical moral. In one of the matters considered in 
Munir , a legal professional whose firm had used a generative AI tool to summarise client 
correspondence and Home Office decision letters recognised the upload as a breach, self -
reported to his regulators, and was not referred further by the tribunal —the tribunal making clear 
that a referral would otherwise have been made.48 The lesson is not that the conduct was 
acceptable; the tribunal was clear that it was not. The lesson is that prompt self -reporting 
materially influenced the outcome for the individual. Information governance protocols should 
accordingly cover not only prevention but also incident response, including reporting obligations 
to the regulator and, where personal data is involved, to the Information Commissioner’s Office.  
Shared and persistent context  
The third context window risk arises from the ability of some GenAI  tools to allow context to 
persist across sessions and to be shared with colleagues —project -based workspaces in the 
leading assistants are current examples.49 This is useful  in that  the model can retain  matter 
background, hold a running brief, and apply consistent drafting guidelines across a team working 
in the same shared workspace. But if confidential client data enters the workspace, it might be  
accessible to everyone in it.  
The unintuitive part is what  is shared. It is not only direct documents, but the data held in the 
shared context —including snippets of individual users’ chats with the AI conducted within that 
workspace. A lawyer who enters into an exchange over  a sensitive aspect of a matter with an AI  
assistant inside a shared project may be disclosing that exchange to every workspace member, 
including members who join later.  
 
46 E Dimitrova ‘AI and client confidentiality: the next regulatory faultline’ (Legal Futures, 27 February 2026) , 
available at https://perma.cc/P5UK -DECJ . 
47 CPR r31.4; and see Practice Direction 57AD, para 2.5 for proceedings in the Business and Property Courts.  
48 Supra  n 1 at [20] and [21].  
49 Eg the AI ‘project’  environments  or shared AI ‘w orkspaces’ which retain project files, instructions and chat -
derived context across sessions and share them with project members.  

 Janeček and Melham, Privilege and confidentiality in generative AI workflows  (SMUR, 22 June 2026, v10.0 ) 
16 The conflict -of-interest and confidentiality implications are immediate, and English law sets the 
bar high. In Bolkiah v KPMG  the House of Lords held that where a firm holds a former client’s 
confidential information and proposes to act in a matter adverse to that client, it must show that 
there is no real risk of the information coming into the possession of those acting on the other 
matter; ad hoc arrangements improvised within a single department will not do —an effective 
information barrier must be ‘an established part of the organisational structure of the firm’.50  
A shared AI workspace is the antithesis of an established structural barrier: it is a deliberately 
constructed channel through which matter -related context flows to everyone with membership. 
Barriers  carefully engineered at the level of file permissions can be bypassed entirely by a shared 
context window that sits outside them, and on Bolkiah  principles it would be for the firm to 
demonstrate that no real risk of leakage through that channel exists. Anyone contributing to a 
shared workspace needs to be able to answer one question: who else can see the context I am 
working in? Before such collaboration features are used on live matters, the organisation’s 
governance framework should answer it for them . And  workspace membership should be 
provisioned from the same source of truth as matter -level access controls.  
Prompt injection  
The fourt h context window risk is prompt injection: malicious content embedded in material the 
AI is invoked  to analyse, containing hidden instructions designed to hijack the model’s 
processing . The example most relevant to legal practice: a counterparty sends a contract for 
review. Embedded invisibly in the file —in white text, invisibly  small  font, document metadata, 
or a form not intelligible to humans at all —is an instruction directing  the AI to summarise the 
agreement favourably or to ignore a particular clause. The danger is that the  model processes all 
the input indiscriminately and generates output accordingly.  
The UK National Cyber Security Centre has warned that prompt injection should not be 
analogised to familiar vulnerabilities , such as SQL injection, and may be worse, because the 
underlying weakness cannot be fully patched: an LLM draws no inherent distinction between 
‘instructions’ and ‘data’ —under the hood there is only ever the prediction of the next token —so 
injection can never be completely mitigated in the way SQL injection can be through 
parameterised queries.51 The NCSC’s advice is to treat an LLM as an ‘inherently confusable 
deputy’ and to design systems so that the impact of a successful injection is contained, rather 
than hoping to prevent injection altogether.  
For practitioners the implications are twofold. Where documents come from external parties, AI -
assisted analysis of them should be treated with appropriate scepticism and outputs verified 
against the source; better still, the data should be cleaned or converted to a plain format before 
entering the system. And where an AI tool is connected to other systems —email, document 
management, drafting tools —the question for the vendor is not whether it blocks prompt 
 
50 Bolkiah v KPMG  [1999] 2 AC 222 (HL) at 239, per Lord Millett.  
51 D Chismon ‘Prompt injection is not SQL injection (it may be worse)’ (National Cyber Security Centre blog, 8 
December 2025), available at https://perma.cc/9EG2 -YC3D . 

 Janeček and Melham, Privilege and confidentiality in generative AI workflows  (SMUR, 22 June 2026, v10.0 ) 
17 injection (no vendor can completely) but what the model can do if an injection succeeds, and 
what deterministic safeguards limit it.  
Retrieval -augmented generation  
The architecture  
RAG is increasingly the architecture behind legal AI tools, and for good reason. Pre -trained 
LLMs have a knowledge cut -off date and have never been exposed to  the firm’s internal 
precedent bank or the client’s files; the model therefore needs to be given relevant data at 
inference time. But, as section 3 explained, the context window cannot hold everything at once.  
RAG solves this with a separate, searchable knowledge store. Documents —cases, legislation, 
internal precedents, client files —are split into chunks of a few hundred words each, converted 
into numerical representations called embeddings, and stored in a vector database. When the user 
provides an input , the system retrieves the chunks most semantically similar (or otherwise 
relevant) to the query into the context window, and the model generates its response  drawing on 
both its trained knowledge and the retrieved chunks. Figure 3 shows the pipeline and marks the 
points at which the risks discussed below arise.  
 
Figure 3: A RAG pipeline and its principal risk points  
It is a useful architecture  that reduces hallucination, sidesteps the long -input problem, and avoids 
baking client data into model parameters. But it introduces a second place where client data 
persists —typically  outside the document management system —and carries with it  five specific 
risks.  


 Janeček and Melham, Privilege and confidentiality in generative AI workflows  (SMUR, 22 June 2026, v10.0 ) 
18 Embedding persistence  
When a document is embedded and stored in a vector database, the data persists until someone 
explicitly deletes it. This is the first risk. When a matter closes, a client relationship ends, or a 
document falls outside the retention policy, the embedded version does not by default  go with it. 
The vector database is, in substance, a copy of the client file in another format, sitting outside the 
systems to which retention schedules are usually applied. This is why well -governed legal teams 
set explicit retention policies for the lifecycle of RAG knowledge bases —in our experience, 
periods of the order of 30 days are typical for matter -specific stores —and why it matters whether  
the AI tool in question has customer -configurable retention settings . In the leading tools the 
control typically exists, but it operates only if someone configures it.  
That said, e ven a well -drafted retention policy brings complications of its own: the team loses 
access to the knowledge base on expiry, so project timelines should account for the scheduled 
expiry of any RAG store the team depends on. The point matters even more  because generative 
AI tools increasingly create RAG knowledge bases automatically , eg, from large documents 
uploaded into a project workspace or from chat history . RAG is sometimes used as a technique 
to control the number  of tokens in the context window: the system only pulls in the d ocument  
chunks it needs , when it needs them . Data can thus enter a persistent retrieval store without 
anyone having consciously decided to put it there.  
Embedding inversion: The database is the documents  
Everything so far concerns the user’s own conduct , but there are risks that do not depend on it at 
all. It is tempting to assume that embeddings —long lists of numbers —are a safely abstracted, 
effectively anonymised representation of the underlying text. Recent research shows otherwise: 
text embeddings reveal almost as much as the text itself. Morris and colleagues demonstrated 
that iterative ‘inversion’ methods can recover 92 per cent of short (32 -token) text passages 
exactly from their embeddings, and can extract verbatim personal information, such as patient 
names, from embeddings of clinical notes.52 Two qualifications matter for the risk assessment: 
the published attacks assume the attacker has access to the embedding vectors and query access 
to (or knowledge of) the embedding model used to create them, and reconstruction fidelity 
degrades for longer passages. But neither assumption is exotic —most production systems use a 
handful of widely available embedding models —and the trajectory of the research is towards 
stronger, more general inversion. A bad actor who gains access to a vector database, even 
without the underlying documents, may therefore be able to reconstruct meaningful content from 
the embeddings alone.  
From a confidentiality standpoint, the conclusion is simple: the vector database should be treated 
as the underlying documents would be , ie, as part of the organisation’s system of record, subject 
to the same access controls, encryption standards , and retention schedules as the document 
management system.  
 
52 J X Morris and others ‘Text Embeddings Reveal (Almost) As Much As Text’ in Proceedings of the 2023 
Conference on Empirical Methods in Natural Language Processing  (2023) ; D Seputis and others ‘Rethinking the 
Privacy of Text Embeddings: A Reproducibility Study of “Text Embeddings Reveal (Almost) As Much As Text”’ in 
Proceedings of the Nineteenth ACM Conference on Recommender Systems (RecSys '25)  (2025) . 

 Janeček and Melham, Privilege and confidentiality in generative AI workflows  (SMUR, 22 June 2026, v10.0 ) 
19 Semantic caching  
The third risk is semantic caching . In our experience, this risk is something only a few 
practitioners have heard of. Caching is a general computing technique: when a task is expensive 
or slow, the system quietly saves the result somewhere quick to access , so that it need not redo 
the work next time. This happens inside AI systems too, without any obvious notification: RAG 
systems cache chunk embeddings and sometimes entire query responses.  In fact, caching also 
affects data in the context window —the input prompts or responses may be cached during a n 
open session with the AI system in order to minimise computation cost  and this can, in a project 
with a shared context window or project files, backfire by leaking one user’s prompt or answer to 
another user . Caching takes several forms, but one particularly risky from a confidentiality 
standpoint is semantic caching.  
In a RAG system with semantic caching, if two users ask ‘similar enough’ questions, the system 
can return the same cached answer rather than running the full retrieval process twice. But 
‘similar enough’ in the eyes of a caching algorithm can cut across client and matter boundaries: 
one fee -earner’s query about a force majeure clause could, in a poorly configured system, return 
a cached response generated using a different matter’s privileged documents. That is data 
contamination with direct professional conduct implications. The practical rule is simple: 
semantic caching should be disabled by default —or strictly partitioned per matter  or per user  
(though this may have significant cost implications) —on any RAG pipeline used for client work, 
and the question ‘is response caching shared across users or matters?’ belongs in every vendor 
due diligence questionnaire.  
Access control  
The fourth risk is access control. On document management systems, documents carry 
permissions: a member of one team cannot read files belonging to a different team, and 
information barriers are enforced at file level. The question is whether the RAG system inherits 
those permissions —or whether the retrieval database cuts across them.  
In theory, chunks in a RAG pipeline should inherit the access permissions of their parent files. In 
practice, general -purpose RAG stacks typically do not do this by default : there is no standard 
mechanism for reading access rights from an uploaded file into a third -party AI tool’s retrieval 
logic. A document that only three people in the organisation may open can become, once 
embedded, retrievable in answer to anyone’s query. The legal -specific platforms are addressing 
the gap  though. For example,  Harvey’s partnership with iManage is designed to let the AI read 
documents in place, governed by the document management system’s existing access controls, 
and Legora  offers document -management integrations and permission mechanisms at matter -
level on the same logic.53 The direction of travel is right, but the firm must verify —not assume —
that permissions actually propagate into retrieval in the configuration deployed.  
 
53 ‘Harvey Announces Tech Partnership With iManage’ (Artificial Lawyer, 5 June 2025) , available at 
https://perma.cc/7S4B -RBDF ; Legora ‘Security and compliance’, supra  n 20; on matter -level permissioning in 
client -facing workspaces, ‘Legora Launches Client Collaboration Portal’ (Artificial Lawyer, 7 November 2025) , 
available at https://perma.cc/3AJ4 -EFVU . 

 Janeček and Melham, Privilege and confidentiality in generative AI workflows  (SMUR, 22 June 2026, v10.0 ) 
20 The safest approach is one RAG pipeline per matter  or per user , populated only with documents 
the relevant  people have  the right to use and share. It is more work to set up, but it maps onto 
professional obligations, limits the blast radius of any leak, and keeps retrieval focused on what 
is actually relevant.  
Retrieval mismatch  
The fifth risk concerns accuracy, and it affects practitioners however carefully the other four are 
managed. Researchers have recently identified and quantified a failure mode they call 
Document -Level Retrieval Mismatch (DRM): a RAG system searching a large legal database 
retrieves content from an entirely incorrect source document.54 Not the wrong paragraph —the 
wrong document. The failure is especially common in legal settings precisely because legal 
documents are structurally similar: contracts follow templates, statutes follow drafting 
conventions, judgments follow standard structures. The retrieval system ’s robustness is 
compromised  by the resemblance, and the model then generates a confident, well -written answer 
based on the wrong source. The user would not know without checking.  
The same research suggests mitigations at the system design level (enriching chunks with 
document -level summaries markedly reduces DRM), but for practitioners the architectural lesson 
converges with the access control point: the more structurally similar the documents in a 
knowledge base, the higher the risk of retrieval from the wrong one, so per -matter RAG is not 
only safer from a privilege standpoint but more reliable on quality grounds too. And on outputs 
that matter, the retrieval sources should be verified —opened and read —not merely observed to 
exist.  
The professional stakes  
The risks canvassed above are not merely technical curiosities; they engage the core duties of the 
profession. We address, in turn, the regulatory framework, the professional negligence analysis, 
the cost objection, and the extent to which our argument travels beyond England and Wales.  
The regulatory framework: Solicitors and barristers  
No SRA rule mentions generative AI, but the existing framework already reaches every risk 
described above. The duty of confidentiality is the anchor. Paragraph 6.3 of the SRA Code of 
Conduct for Solicitors requires the solicitor to ‘keep the affairs of current and former clients 
confidential unless disclosure is required or permitted by law or the client consents’; the same 
obligation binds firms, which must in addition have in place effective governance structures, 
arrangements, systems and controls ensuring compliance.55 The duty is not exhausted by 
refraining from deliberate disclosure: as Bolkiah  makes clear, the lawyer must take effective 
steps against any real risk of confidential information escaping or being misused.56 Every mode 
 
54 M Reuter and others ‘Towards Reliable Retrieval in RAG Systems for Large Legal Datasets’ in Proceedings of 
the Natural Legal Language Processing Workshop 2025  (2025).  
55 SRA Code of Conduct for Solicitors, RELs and RFLs, para 6.3; SRA Code of Conduct for Firms, paras 2.1 and 
6.3. 
56 Supra  n 50 at 235 –39. 

 Janeček and Melham, Privilege and confidentiality in generative AI workflows  (SMUR, 22 June 2026, v10.0 ) 
21 of data flow discussed in this article —training, context transit, shared workspaces, vector stores, 
caches —is a channel through which that duty can be breached without any document ever being 
‘sent’ to anyone in the conventional sense. Pasting a client letter into a consumer -grade chatbot 
is, on the regulator’s own analysis, such a breach: the SRA’s Risk Outlook report identifies as 
‘particular threats’ to confidentiality and privilege a staff member ‘using an online AI, such as 
ChatGPT, to answer a question on a client’s case’, confidential data ‘being revealed when it is 
transferred to an AI provider for training’, and —strikingly, given section 4  of this article —‘the 
output from an AI system replicating confidential details from one case in its response to 
another’.57 The Law Society’s guidance is to the same effect and blunter: where a firm uses ‘a 
free, online generative AI service’ with no operational relationship with the vendor beyond use, 
‘do not put any confidential data into the tool’.58 
Confidentiality is not the only relevant head. Competence and supervision (paras 3.2 –3.5 of the 
Code for Solicitors) require the practitioner to understand the tools through which the service is 
delivered and to supervise work effected through them —the lesson the tribunal drew explicitly in 
Munir , where the failure was supervisory before it was technological.59 The duty not to mislead 
the court or others (para 1.4) is engaged by unverified AI output, as Ayinde  and the Law 
Society’s September 2025 update emphasise.60 Client communication obligations bear on 
whether and how clients should be told that generative AI is used on their matters; the Law 
Society advises that expectations be discussed and clearly communicated.61 And the reporting 
obligations close the loop: a serious breach —and the upload of privileged client material to a 
consumer -grade tool may well be one —engages the duty to report promptly to the SRA (paras 
7.7–7.8 of the Code for Solicitors), and, where personal data is involved, the controller’s duty to 
notify the Information Commissioner within 72 hours unless the breach is unlikely to result in 
risk to individuals, with notification to affected data subjects where the risk is high.62 The self -
reporting thread in Munir , discussed above, shows the practical value of taking those obligations 
seriously.  
The Bar’s regulator has addressed the same ground in greater technical detail. The Bar Standards 
Board’s guidance on AI, current from May 2026, analyses the use of AI tools as outsourcing  
under rC86 of the BSB Handbook —the barrister remains responsible for the work and must 
ensure the provider’s terms and technical capability can maintain the confidentiality obligations 
of Core Duty 6 —and states in terms that ‘it is unlikely that free -of-charge, publicly available AI 
systems fulfil these conditions’.63 The guidance places general -purpose tools without adequate 
 
57 Supra n 13, section on confidentiality and privacy.  
58 Supra  n 13, para 3.3.  
59 Supra  n 1, headnote para 2 and [56] –[58].  
60 R (Ayinde) v London Borough of Haringey; Al -Haroun v Qatar National Bank QPSC  [2025] EWHC 1383 
(Admin) , [2025] 1 WLR 5147 ; The Law Society, supra  n 13. 
61 Supra  n 58, paras 1.4 and 3.1.  
62 SRA Code of Conduct for Solicitors, RELs and RFLs, paras 7.7 –7.8; UK GDPR, arts 33  and 34. 
63 Bar Standards Board ‘Guidance on the use of Artificial Intelligence and Other Technologies’ (current from 18 
May 2026) , available at https://perma.cc/89P6 -GNCQ , section 4 and rC86.3 commentary.  

 Janeček and Melham, Privilege and confidentiality in generative AI workflows  (SMUR, 22 June 2026, v10.0 ) 
22 data protections, and agentic AI features, in the highest risk band; it expects record -keeping to 
extend to AI audit logs and prompt deletion policies; and —corroborating our shared -context 
analysis —it warns that a chambers -level AI assistant may allow other barristers in chambers to 
‘view or infer sensitive client information’, ‘particularly if other barristers in your chambers are 
representing an opposing party’.64 Notably, the guidance also expects counsel to consider 
whether a client’s or instructing solicitor’s own AI use has compromised privilege and to advise 
accordingly65—regulatory confirmation of the client -warning duty developed below. Both 
branches of the profession, in short, are now on notice from their regulators, as are judges from 
theirs.66 
Two further regulatory observations  are relevant here . First, accountability is non -delegable: as 
the SRA puts it, the firm remains ‘responsible and accountable for the outputs’ of the AI it 
uses;67 vendor assurances allocate commercial risk, not professional responsibility. Secondly, 
abstention is not a safe harbour either: the same Risk Outlook observes that ‘the risk to firms 
might not come from adopting AI, but from failing to do so’.68 The regulatory expectation, in 
other words, is calibrated adoption under effective governance , and this is precisely the position 
that our three -mode analysis  in this article  is designed to enable.  
Privilege governance  
As Munir  and Heppner  illustrate, confidentiality  provides  the structural support : where the data -
handling terms destroy any reasonable expectation of confidentiality, privilege fails or is waived, 
and it is the client who bears the loss. Conversely, as the Chancellor of the High Court has 
indicated, a lawyer’s use of a demonstrably secure system to assist in giving advice should not 
imperil the client’s privilege,69 and on the limited -disclosure authorities an enterprise -grade 
deployment under genuine confidentiality obligations should fall on the safe side of the line. The 
difference between those two outcomes is governance —and, if a challenge ever comes, evidence 
of governance: the firm that can produce its data processing agreement, its zero -retention 
configuration, and its vendor due diligence record is in a categorically stronger position to 
defend the reasonableness of its expectation of confidentiality than the firm that can produce 
only an invoice.  
 
64 Ibid, sections 1 and 4.  
65 Ibid, sections 1 (‘Core Duty 7’) and 6.  
66 Courts and Tribunals Judiciary ‘Artificial Intelligence (AI) Guidance for Judicial Office Holders’ (31 October 
2025) , available at https://perma.cc/5NEM -U57H . 
67 Supra  n 13, section on accountability.  
68 Ibid, section on opportunities from AI.  
69 Supra  n 3, para 26.  

 Janeček and Melham, Privilege and confidentiality in generative AI workflows  (SMUR, 22 June 2026, v10.0 ) 
23 The professional negligence analysis  
The standard of care expected of a solicitor is that of the reasonably competent practitioner 
professing the relevant skill,70 and it evolves with the tools and knowledge available to the 
profession.  
Compliance with common practice will not necessarily suffice because where a practice widely 
followed in the profession carries a foreseeable and readily avoidable risk, a court may condemn 
the practice itself.71 A firm whose AI deployment mirrors its peers’ is therefore not thereby safe. 
Additionally , the standard is temporal: it is judged by the knowledge reasonably available at the 
time of the alleged breach, not with hindsight. That cuts both ways. Conduct excusable in 2023, 
when these failure modes were obscure, may be inexcusable in 2026, now that the regulator, the 
Law Society, the NCSC and the judiciary have all published on them; conversely, a firm cannot 
be blamed  for failing to anticipate an extraction technique unknown at the time of deployment —
though it can be blamed  for failing to reassess  a deployment as the threat landscape moves, 
which is why periodic review  is essential . 
A lawyer who deploys generative AI without understanding the data flows described above risks 
falling below the standard: data re tention, caching, model memorisation, context sharing, context 
rot, silent truncation and retrieval mismatch and so on are now documented, foreseeable failure 
modes, and inputting of client data under a risky setting or unverified reliance on confident -
looking output is precisely the kind of risk that documented foreseeability makes culpable, as the 
growing line of hallucinated -citation cases shows.72 
It should not be assumed that every AI -related mishap sounds in damages. On current scope -of-
duty analysis, the question is whether the loss claimed falls within the risk that the relevant duty 
existed to guard against.73 Where a solicitor’s retainer includes the safekeeping of the client’s 
confidential and privileged information —as it almost always will —the loss of privilege over 
strategy documents through processing in an inadequately vetted system falls squarely within 
that risk. Causation will often be straightforward (but for the upload, the material would have 
remained privileged), and the loss —inspection of strategy documents by an opponent, or by a 
prosecutor, as in Heppner —is concrete and was avoidable at modest cost. Quantification is 
harder, because the value of lost privilege manifests in litigation outcomes, settlement positions 
and regulatory exposure; but difficulty of quantification has never been a bar to liability, and the 
disciplinary consequences will often arrive before any negligence claim does. The accuracy 
failure modes feed a more familiar pattern, namely negligent advice or drafting based on an 
incomplete reading of the documents —the question being whether the lawyer’s verification 
practice met the standard, not whether the tool erred.  
 
70 Bolam v Friern Hospital Management Committee  [1957] 1 WLR 582; for solicitors, Midland Bank Trust Co Ltd v 
Hett, Stubbs & Kemp  [1979] Ch 384 , [1978] 3 WLR 167 . 
71 Edward Wong Finance Co Ltd v Johnson Stokes & Master  [1984] AC 296 (PC).  
72 R (Ayinde) , supra n 60; Munir , supra  n 1. 
73 Manchester Building Society v Grant Thornton UK LLP  [2021] UKSC 20, [2022] AC 783; Meadows  v Khan 
[2021] UKSC 21, [2022] AC 852.  

 Janeček and Melham, Privilege and confidentiality in generative AI workflows  (SMUR, 22 June 2026, v10.0 ) 
24 There is also a client -facing dimension that is easy to overlook. Heppner  was not a case of 
lawyer error: it was the client’s  self-directed use of a consumer -grade chatbot that generated the 
unprotected material. A client cannot be supervised into compliance, but a client can be warned. 
Retainer letters and engagement communications should now routinely advise clients not to 
discuss the matter with, or upload matter documents to, consumer -grade AI tools —in the same 
way that clients have long been warned about discussing litigation on social media or using a 
spouse’s email account. Given how natural it has become to ‘ask the chatbot’, a failure to give 
that warning may itself come to look like a gap in competent client care —and for barristers, the 
BSB guidance now expressly contemplates advising clients where their own AI use may have 
compromised privilege.74 
The cost objection  
A fair objection to these prescriptions is that they are not free: the burden falls hardest on small 
firms, legal aid practices and thinly resourced in -house teams, while the access -to-justice benefits 
of cheap consumer -grade tools accrue precisely to those who cannot afford bespoke 
deployments —a point the Chancellor of the High Court acknowledged in noting the pro -access 
role of AI -assisted litigants in person.75 Three responses  to that objection are in order . First, the 
core recommendations of this article —know the data -handling terms, keep client data out of 
consumer -grade tools, partition by matter, verify outputs —cost attention rather than capital, and 
the Law Society’s SME -focused guidance seem s to be  built on the same premises.76 Secondly, 
the real comparison is between governance costs and breach costs: a single privilege waiver, 
regulatory referral or notifiable data breach will consume more partner time than a procurement 
questionnaire. Thirdly, the standard of care accommodates proportionate, documented risk 
decisions; what it does not accommodate is ignorance of where the data goes.  
Beyond England and Wales  
This paper doctrinally focuses on  English  law professionals , but the structural  claims are 
deliberately jurisdiction -neutral, because they attach to the technology rather than the law: 
privilege and professional secrecy regimes protect what is confidential, and generative AI 
systems challenge confidentiality through the same three modes everywhere.  
The extendibility  is visible in the early cases already : Heppner  reached the Munir  sentiment by 
orthodox American means (reasonable expectation of confidentiality defeated by the terms of 
use);77 the American Bar Association’s first ethics opinion on generative AI requires lawyers to 
evaluate the risk that client information will be disclosed to or accessed by others, recommends 
informed consent before client confidences enter self -learning tools, and warns that boilerplate 
 
74 Supra  n 63, section 6; for direct access clients, rC123.  
75 Supra  n 3, para 16.  
76 The Law Society ‘Generative AI - lifecycle considerations for SMEs’ (2024), accompanying the guidance supra  n 
13. 
77 Supra  n 2. 

 Janeček and Melham, Privilege and confidentiality in generative AI workflows  (SMUR, 22 June 2026, v10.0 ) 
25 consent in engagement letters will not do;78 and the EDPB’s position that trained models are not 
automatically anonymous supplies the same analytical move  for the EU’s data protection 
regime.79  
Of course, there are limits to the extendibility  of the analysis  too. In EU competition 
investigations, for example, Akzo Nobel  denies privilege to in -house lawyer communications 
altogether, so in -house teams operating in that context must treat the AI analysis as an overlay on 
an already unprotected baseline.80 And in civilian professional secrecy systems, where the duty is 
owed by the lawyer and backed by criminal law rather than held by the client as a litigation 
privilege, the ‘waiver’ analysis transfers poorly —but the data -governance analysis transfers in 
full, since a secrecy obligation is breached by the leak itself regardless of any waiver doctrine. 
The portable core is this: wherever protection depends on demonstrable confidentiality, the three 
modes of data persistence determine what must be governed.  
What does adjusted governance look like in concrete terms? For memorisation: know each tool’s 
training policy, and do not fine -tune on client or personal data until erasure is technically 
solvable. For the context window: use enterprise -grade tools with strict privacy controls; check 
for context rot on long documents; disable or scope prompt caching; govern shared workspaces 
so that membership tracks matter access rights; and treat externally sourced documents as 
prompt injection risks. For RAG: disable or partition semantic caching; manage embedding 
persistence under explicit retention schedules; ensure access controls follow documents into the 
vector store; build per -matter pipelines; and verify retrieval sources on outputs that matter. And 
across all three: train staff to recognise these failure modes, and maintain an incident -response 
protocol that includes self -reporting —which, as Munir  shows, the tribunals will weigh.  
Conclusion  
Three modes, three sets of risks, one overarching principle: wherever the client’s data goes, the 
lawyer’s professional obligations go with it. An AI system, no matter how impressive it is in its 
performance,  does not know  by itself if  it is processing privileged material , and a responsible 
data governance framework needs to account for that.  
Our conclusions can thus be stated as follows:  
(a) the doctrinal tests for privilege have not changed, and are unlikely to; what has changed is 
the factual substrate to which they apply, and the knowledge a competent practitioner must 
have to apply them;  
(b) putting client material into a consumer -grade tool —one whose terms permit retention, 
training use and onward disclosure —is a breach of confidentiality and puts privilege at 
 
78 American Bar Association, Standing Committee on Ethics and Professional Responsibility, Formal Opinion 512 
‘Generative Artificial Intelligence Tools’ (29 July 2024), discussing Model Rules 1.1, 1.4 and 1.6.  
79 Supra  n 7. 
80 Akzo Nobel Chemicals Ltd v European Commission  Case C -550/07 P [2010] ECR I -8301 , [2011] 2 AC 338. 

 Janeček and Melham, Privilege and confidentiality in generative AI workflows  (SMUR, 22 June 2026, v10.0 ) 
26 serious risk, whichever doctrinal route (loss of confidentiality, waiver, or destruction of the 
reasonable expectation) a court ultimately prefers;  
(c) on the limited -disclosure authorities, a properly configured enterprise -grade deployment —
no training use, zero or minimal retention, confidentiality obligations holding down the sub -
processing chain, and documentation to prove it —should not in itself imperil privilege, and 
the contrary suggestion in the early case law rests on obiter guidance and imprecise 
terminology;  
(d) the persistence layers that lawyers do not see —fine-tuned parameters, shared workspaces, 
vector stores, caches —are where the unintuitive breaches will happen, and they must be 
governed with the same rigour as the document management system; and  
(e) because these failure modes are now documented and publicised by the regulator, the 
professional bodies and the senior judiciary, the standard of reasonable care has already 
begun  (and will no doubt continue)  to absorb them: yesterday’s reasonable precautions may 
be tomorrow’s negligence.  
The underlying challenge, in short, is not the artificial intelligence but the data: moving through 
systems that are not designed for transparency, in ways that remain counter -intuitive until they 
are made visible. Lawyers who understand the difference between a model’s parameters, its 
context window, and its retrieval stores can manage these risks  much better  and maintain the 
confidentiality of material processed by GenAI systems, wherever they practise. That 
understanding is increasingly part of what technological competence in 2026  looks like —legal 
professionals who know  which aspects of  the AI systems’ design affect data confidentiality and 
who can therefore check how the AI tool in question address es them .  