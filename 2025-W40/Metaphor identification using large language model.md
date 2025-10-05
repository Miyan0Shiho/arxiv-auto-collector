# Metaphor identification using large language models: A comparison of RAG, prompt engineering, and fine-tuning

**Authors**: Matteo Fuoli, Weihang Huang, Jeannette Littlemore, Sarah Turner, Ellen Wilding

**Published**: 2025-09-29 14:50:18

**PDF URL**: [http://arxiv.org/pdf/2509.24866v2](http://arxiv.org/pdf/2509.24866v2)

## Abstract
Metaphor is a pervasive feature of discourse and a powerful lens for
examining cognition, emotion, and ideology. Large-scale analysis, however, has
been constrained by the need for manual annotation due to the context-sensitive
nature of metaphor. This study investigates the potential of large language
models (LLMs) to automate metaphor identification in full texts. We compare
three methods: (i) retrieval-augmented generation (RAG), where the model is
provided with a codebook and instructed to annotate texts based on its rules
and examples; (ii) prompt engineering, where we design task-specific verbal
instructions; and (iii) fine-tuning, where the model is trained on hand-coded
texts to optimize performance. Within prompt engineering, we test zero-shot,
few-shot, and chain-of-thought strategies. Our results show that
state-of-the-art closed-source LLMs can achieve high accuracy, with fine-tuning
yielding a median F1 score of 0.79. A comparison of human and LLM outputs
reveals that most discrepancies are systematic, reflecting well-known grey
areas and conceptual challenges in metaphor theory. We propose that LLMs can be
used to at least partly automate metaphor identification and can serve as a
testbed for developing and refining metaphor identification protocols and the
theory that underpins them.

## Full Text


<!-- PDF content starts -->

METAPHOR IDENTIFICATION USING LARGE LANGUAGE MODELS:
ACOMPARISON OF RAG,PROMPT ENGINEERING,AND
FINE-TUNING
A PREPRINT
Matteo Fuoli
Department of Linguistics and Communication
University of Birmingham
m.fuoli@bham.ac.ukWeihang Huang
Department of Linguistics and Communication
University of Birmingham
wxh207@student.bham.ac.uk
Jeannette Littlemore
Department of Linguistics and Communication
University of Birmingham
j.m.littlemore@bham.ac.uk
Sarah Turner
Centre for Arts, Memory & Communities
Coventry University
Sarah.Turner@coventry.ac.ukEllen Wilding
Department of Linguistics and Communication
University of Birmingham
esw217@student.bham.ac.uk
October 2, 2025
ABSTRACT
Metaphor is a pervasive feature of discourse and a powerful lens for examining cognition, emotion,
and ideology. Large-scale analysis, however, has been constrained by the need for manual annotation
due to the context-sensitive nature of metaphor. This study investigates the potential of large language
models (LLMs) to automate metaphor identification in full texts. We compare three methods: (i)
retrieval-augmented generation (RAG), where the model is provided with a codebook and instructed to
annotate texts based on its rules and examples; (ii) prompt engineering, where we design task-specific
verbal instructions; and (iii) fine-tuning, where the model is trained on hand-coded texts to optimize
performance. Within prompt engineering, we test zero-shot, few-shot, and chain-of-thought strategies.
Our results show that state-of-the-art closed-source LLMs can achieve high accuracy, with fine-tuning
yielding a median F1 score of 0.79. A comparison of human and LLM outputs reveals that most
discrepancies are systematic, reflecting well-known grey areas and conceptual challenges in metaphor
theory. We propose that LLMs can be used to at least partly automate metaphor identification and can
serve as a testbed for developing and refining metaphor identification protocols and the theory that
underpins them.
Keywordsmetaphor identification ·large language models ·metaphor detection ·linguistic annotation ·artificial
intelligence
1 Introduction
A large body of research has shown metaphor to be a pervasive feature of discourse and a fundamental tool for
communication and interaction. By drawing on more concrete, familiar, easily understood ideas, speakers can articulatearXiv:2509.24866v2  [cs.CL]  1 Oct 2025

Metaphor identification using large language modelsA PREPRINT
their subjective experiences in ways that are more accessible to others. Metaphor analysis thus provides a valuable
lens for exploring important research questions with profound individual and social implications, such as patients’
experiences of cancer (Semino et al., 2017) or public perceptions of immigration and national identity (Musolff, 2016).
However, metaphor analysis is an extremely time-consuming process because there are currently no reliable methods
for automating the crucial yet challenging task of identifying metaphorical expressions in text. The need for manual
text annotation, a labour-intensive and error-prone process, inherently limits the scalability of metaphor analysis and, by
extension, the degree to which findings can be generalized.
Recent studies in natural language processing (NLP) have shown that large language models (LLMs) – advanced AI
systems trained on massive text corpora – can be used to identify metaphors automatically with promising levels of
accuracy (Puraivan et al., 2024; Tian et al., 2024; Yang et al., 2024; Hicke and Kristensen-McLachlan, 2024). However,
these studies define the task narrowly, focusing on classifying single, often pre-selected words as either literal or
metaphorical within decontextualized sentences. This narrow operationalization not only yields outputs of limited value
to linguists, but also raises concerns about validity, as metaphors often go beyond single words and can only be properly
interpreted by considering the surrounding text. Moreover, existing research explores only a limited set of strategies for
deploying LLMs in this task, leaving much of the methodological landscape unexplored.
In this study, we assess the performance of LLMs in the task of identifying and annotating metaphor in full texts.
We conduct a series of experiments to compare three distinct methods: (i) ‘retrieval-augmented generation’ (RAG),
where we provide the model with our codebook as a knowledge resource and instruct it to annotate our corpus based
on the rules, principles, and examples included in it; (ii) ‘prompt engineering’, where we design verbal instructions
for the model to perform the task; and (iii) ‘fine-tuning’, where we train the model on a sample of hand-coded texts
to optimize its performance. Within the prompt engineering strand, we experiment with three common prompting
strategies: ‘zero-shot’, ‘few-shot’, and ‘chain-of-thought’. We also examine the effects of varying the number and
type of examples included in the prompt to guide the model in the identification task. We conduct these experiments
using 10 of the most advanced closed- and open-source LLMs available at the time of writing. We apply these LLMs
to a corpus of film reviews hand-coded using a ‘phraseological’ approach that captures both single- and multi-word
metaphorical expressions (Fuoli et al., 2022).
This study advances both metaphor studies and NLP by providing the first evaluation of LLMs for full-text metaphor
identification and annotation, alongside a comparison of three core methodological approaches. If LLMs can perform
this task accurately, they could transform metaphor research by providing an accessible tool for efficiently annotating
large text corpora, enabling researchers to scale up and make their analyses more generalizable, and redirect resources
from low-level annotation tasks to higher-level interpretation and theory-building. LLMs offer a key advantage over
other computational methods: they can be guided using natural language prompts, thus empowering researchers without
advanced programming expertise to automate one of the most tedious and time-consuming parts of their work. In
addition to this methodological contribution, our study offers new theoretical insights into metaphor by analysing the
LLM annotations and comparing them with those of human coders. We show that this process can help reveal areas
where current metaphor identification protocols – and the theory that underpins them – may be refined.
2 Background
This section provides the essential theoretical and methodological background for this study. We begin by defining
the analytical task of metaphor identification and reviewing existing approaches, before outlining our own. Next, we
summarize previous work on LLM-assisted metaphor identification.
2.1 Identifying metaphors in discourse
Metaphor can be defined in different ways according to whether one is interested in approaching it at the linguistic
level, at the conceptual level, or at the level of discourse (Steen, 2009). However, in practice, it is impossible to perceive
a clear separation between these different levels. In this study, therefore, we are interested in providing an automated
identification procedure that can be used regardless of the level. In order to do this, we need to start with a definition of
metaphor that works on all three levels. We therefore define metaphor as a tool for reasoning, or for communication,
that involves comparing one entity (the ‘topic’) to another entity (the ‘vehicle’), in order to express an opinion about
that entity, to communicate one’s feelings about it, and/or to render it more easily understood. We acknowledge the
presence of underlying systematic (Cameron, 2010a) and/or conceptual (Lakoff and Johnson, 1980) mappings that
underpin a good number of surface-level “linguistic” metaphors, although this is not a focus of the current study.
A long-standing problem faced by metaphor researchers has been the need to establish a rigorous, efficient, and
replicable procedure to identify metaphors in discourse. Various solutions have been proposed to this problem, which
2

Metaphor identification using large language modelsA PREPRINT
has led to the development of a number of different approaches. These approaches have advantages and disadvantages.
Approaches such as the MIP (Group, 2007) or the MIPVU procedure (Steen et al., 2010) involve taking every lexical
unit in the text and establishing whether it is being used metaphorically. This involves checking to see whether its
contextual meaning can be understood through comparison with a more basic entity. This is a rigorous approach
involving the use of dictionaries to establish the different senses of the lexical unit, and as such its results have been
shown to be highly replicable. The problem with this approach is that it is very time-consuming, and it is therefore
difficult to apply to large corpora. Furthermore, identifying meaningful metaphorical items often requires an additional
level of coding, which considers how individual lexical items flagged as potentially metaphorical combine to form a
single metaphorical meaning unit within a phrase or longer stretch of text.
Other researchers have adapted their metaphor identification methodology based on their research questions, which has
led to a greater or lesser adherence to the MIP(VU). For example, studies such as Littlemore and Turner (2020) have
adopted a more phrase-based approach to metaphor, reminiscent of Cameron’s 2003 ‘Metaphor Identification through
Vehicle Terms’ (MIV). The MIV procedure consists of finding pieces of language in the text that have the potential to be
interpreted metaphorically. Cameron defines ‘vehicle term’ as a word or phrase that “stands out clearly as incongruous
or anomalous in its discourse context” (Cameron, 2010b, p. 4). For Cameron, there are two necessary conditions for the
identification of linguistic metaphor: a contrast in meaning between the vehicle and topic and a transfer or connection
of meaning between them (Cameron, 2003, pp. 59-60). A similar approach to metaphor identification has also been
taken by Johansson Falck and Okonski (2023) in their Procedure for Identifying Metaphorical Scenes (PIMS). Used
specifically for the identification of metaphor in prepositional constructions, this method allows the analyst to identify
metaphorical meaning that extends over phrases or longer stretches of text. It focuses on the scenes evoked by these
phrases and allows the analyst to establish whether the scenes are metaphorical, non-metaphorical, or ambiguous.
The flexibility of this latter set of approaches allows them to be tailored to the characteristics of the data, and to focus
on the phraseological meaning. These methods bring to the fore the ways in which metaphor functions in context across
broader stretches of text. However, this flexibility may engender issues with replicability, as it is not always easy to see
where a metaphor begins and ends. Also, when the metaphor identification procedure has been developed iteratively in
order to meet the needs of the research questions, it may be difficult for a third party to apply the procedure in the same
way.
To illustrate the differences between approaches that focus primarily on individual lexical units (which are usually
words) and approaches that focus more on metaphor at the level of the phrase, let us consider the following example
from Turner and Littlemore (2023) which is taken from an interview with a bereaved parent talking about how she felt
when her child died:
We were just forcibly thrown into this awful universe of just misery and pointlessness.
If an analyst were to apply a procedure such as the MIP(VU), they would only identify lexical units such asthrown,
universe, and possiblyforciblyas metaphorical. While this analysis may be useful if one were interested in measuring
metaphoric density, it does not provide insight into the holistic experience that the parent is describing. A more
scene-based approach to metaphor analysis would take the whole utterance as a single metaphorical instance. This
provides richer insight into the nature of the experience that the parent is describing but does allow for a certain degree
of subjectivity in deciding where the metaphor begins and ends.
For the reasons outlined above, the approach to metaphor identification that we take in this study aligns more with this
latter set of approaches. The procedure employed is described in detail in Fuoli et al. (2022). We began by reading the
entire text to establish a general understanding of the meaning. We then identified meaning units at the level of phrase
following Cameron’s (2003) VIP. For each meaning unit, we established its meaning in context (i.e. its contextual
meaning, taking into account what comes before and after the meaning unit). Having done so, we determined whether
the phrase had a more basic contemporary meaning in other contexts than the one in the given context. For our purposes,
basic meanings tend to be more concrete. If the meaning unit had a more concrete contemporary meaning in other
contexts than the given context, and if this contextual meaning could be understood in comparison with it, it was marked
as metaphorical. In some cases, metaphors were identified at the level of the single word. However, they often extended
beyond single words. This could occur when:
1. The expression was a conventional idiom, e.g.have your cake and eat it.
2. There were hyphenated words which formed a single lexical unit, e.g.tough-as-nails.
3.There was an adjectival entailment of a metaphorically used noun (or an adverbial entailment of a metaphori-
cally used verb) that was internally semantically coherent with the literal sense of the noun or verb. Phrases
that were internally coherent were marked as a single metaphor, even when there was a non-metaphorical
stretch of texts separating them. However, if there were two distinct ideas in the same metaphorical phrase,
3

Metaphor identification using large language modelsA PREPRINT
these were marked as separate metaphors. E.g.It’s pretty much asunken shipof a movie(Ships can sink in
the literal world andsunkenis serving as a premodifier ofshipin this sentence).
4.There was an adjectival entailment of a metaphorically used noun (or an adverbial entailment of a metaphori-
cally used verb) that was internally semantically coherent with the metaphorical sense of the noun or verb, but
which would not occur in literal language, e.g. thenegative baggagethat the reviewers were likely to have.
(baggagein its metaphorical sense here can be positive or negative but literal baggage is not conventionally
described in these terms).
Phrases were coded as metaphor even when they were signalled with tuning devices such aslikeoras. Individual words
were not broken down into their metaphorical components. We followed an overarching principle where we kept the
length of the annotated text spans to a minimum.
Despite the presence of these relatively rigorous, replicable procedures for metaphor identification, and despite the
relatively high inter-rater reliability scores obtained for some of the procedures, including the approach that we use in
the study, it should be acknowledged that a degree of subjectivity is always inherent in metaphor identification. There
will always be grey areas about what constitutes a metaphor or not, and what may be experienced as metaphorical by
one person may be experienced as literal by another. This makes the practice of inter-rater reliability testing all the
more important, as it is through comparing annotations made independently by different analysts that we can assess the
transparency and robustness of our annotation criteria and gain useful insights for refining them. However, performing
such tests is not always feasible due to the significant time investment required from multiple people. To address
this challenge, we propose that LLMs could be leveraged as artificial reliability coders, provided they are sufficiently
accurate in identifying metaphor.
Another key challenge LLMs can help us address is scale. The time-consuming nature of metaphor identification can
limit the scope of projects, impeding or even preventing the analysis of large datasets. For these reasons, if it were
possible to automate the process of metaphor identification through the use of LLMs, this would make it easier to
expand the scope of metaphor studies, allowing metaphor scholars to make more generalisable claims based on their
findings.
2.2 Metaphor identification using LLMs
Automating metaphor identification has long been a central goal in NLP. Traditionally, researchers approached this task
– commonly referred to in the field as ‘metaphor detection’ – by developing supervised machine-learning algorithms that
label words as literal or metaphorical using linguistic features such as ‘word embeddings’ (numerical representations of
word meaning derived from co-occurrence patterns in large text corpora), part-of-speech tags, and word concreteness
ratings (Ge et al., 2023). The performance of these systems is generally evaluated against hand-coded datasets such
as the VU Amsterdam Metaphor Corpus (Steen et al., 2010) or the MOH-X corpus (Mohammad et al., 2016), which
provide annotations of metaphorical language at the level of individual words. These traditional NLP approaches
have achieved solid results, with the best models surpassing 80% accuracy (for a review and comparison of model
performances, see Ge et al., 2023).
In recent years, LLMs have emerged as a transformative technology, reshaping research practices and priorities in NLP
and driving a paradigm shift across the field. Among the many applications LLMs have been deployed for, metaphor
identification has attracted growing interest. Most studies in this area test LLMs on the task of classifying pre-selected
single words in sentences as either literal or metaphorical. For example, Puraivan et al. (2024) evaluate the performance
of OpenAI’s LLMs GPT-4o and GPT-4 Turbo on a corpus of Spanish sentences. They compare two variants of the
prompt used to instruct the LLMs: one providing only task instructions, and another supplementing these instructions
with a brief explainer of metaphor based on Conceptual Metaphor Theory (CMT). They report high average accuracy
scores, peaking at 88.29%. Interestingly, neither the use of the more advanced GPT-4o model nor the inclusion of
theory-based instructions led to performance improvements.
Instead of relying on a single prompt, Tian et al. (2024) introduce an interactive prompting strategy informed by CMT
and the MIP procedure designed to help LLMs make more accurate metaphorical judgments by scaffolding their
“reasoning”. Specifically, the model is guided through a sequence of three questions to identify the source domain
implied by the target word, determine its target domain within the sentence, and evaluate whether these domains differ,
thereby justifying a metaphorical interpretation. This step-by-step approach substantially improves both the accuracy
and interpretability of LLM-based metaphor identification, achieving an F1 score as high as 82.59.
Taking a different approach, Yang et al. (2024) treat GPT-3.5 Turbo not as a direct metaphor classifier, but as a tool
for constructing linguistic resources to support metaphor detection. Specifically, they prompt the LLM to generate
verb-specific lists of common literal subject-verb and verb-object collocations, which serve as reference points for
4

Metaphor identification using large language modelsA PREPRINT
metaphor identification. This process is further enhanced by generalizing the subjects and objects into broader topic
categories using resources like WordNet, LDA topics, and Oxford dictionaries. A verb is classified as metaphorical
when the subject and object in the target sentence fail to match any of the topics associated with its literal collocations.
Their method achieves fairly good results, with the best performing configuration producing an F1 score of 70.1.
While the studies above test LLMs on the narrow task of judging whether a pre-selected word in a sentence is
metaphorical, Liang et al. (2025) use GPT-4 to identify all metaphorical words in a set of sentences from the VU
Amsterdam Metaphor Corpus and a smaller set of online news texts. The task follows the MIPVU procedure and
focuses solely on conventional metaphors. The authors test a wide range of prompt designs and experiment with
varying the number of examples provided to the model (0, 1, 5, and 10). Across all settings, GPT-4’s performance
was consistently poor, with the best results achieved in the 1-shot condition (F1 = 30.4), far below the performance of
previous-generation transformer models such as RoBERTa (F1 77.9–79.7). The authors attribute the weak results to
several interacting factors: the style and complexity of the prompts, the limited context provided by single-sentence
inputs (some as short as a single word), the uneven representativeness of the examples, and broader limitations of
GPT-4. They conclude that prompt-based approaches with LLMs remain unreliable for this task, though future work
exploring optimized prompt design or alternative methods such as fine-tuning and RAG may yield better results.
Along similar lines, Hicke and Kristensen-McLachlan (2024) test GPT-3.5 Turbo, GPT-4, and GPT-4o on the task of
labelling words as either metaphorical or literal in sentences drawn from the TroFi dataset (Birke and Sarkar, 2006)
and from Lakoff and Johnson (1980). Their prompt is adapted almost verbatim from the original MIP procedure,
supplemented by two annotated example sentences. In addition to identifying metaphorical words, the models are
asked to provide a brief description of each word’s more basic meaning, enabling the researchers to assess how well the
LLMs can carry out Step 3 of MIP and to make the outputs more interpretable by exposing the models’ “rationale”.
The results show that all models perform above chance. GPT-4o performed best, correctly identifying all metaphors in
74% of sentences from the Lakoff and Johnson dataset and 65% of the basic meanings. However, all models tended
to over-label words as metaphorical and struggled with function words, multiword expressions, and certain metaphor
types.
While, with the exception of Liang et al. (2025), these studies suggest that LLMs hold promise for accurate metaphor
identification, the way this task is implemented in NLP, both in traditional and LLM-based approaches, differs
significantly from how metaphor scholars practice it. As seen above, instead of training systems to process full texts and
identify all metaphorical expressions contained in them, NLP systems are typically designed to classify individual words
– often specific, pre-selected target words – as either literal or metaphorical within decontextualized sentences. This
approach runs counter to a core principle in metaphor analysis, namely that metaphor is a context-sensitive phenomenon,
shaped by both real-world knowledge and the evolving discursive context of a text (Semino, 2008). The importance of
context is reflected in the very first step of the MIP procedure (often treated as the methodological foundation of NLP
approaches), which explicitly instructs annotators to “read the entire text-discourse to establish a general understanding
of the meaning” (Group, 2007, p. 3). Another important drawback of current NLP approaches is that they operate
strictly at the word level and fail to account for multiword metaphorical expressions, which is problematic for the
reasons discussed in Section 2.1.
The choice to operationalize the task of metaphor detection as the labelling of individual words within individual
decontextualized sentences is likely driven by methodological considerations and technical constraints. This setup
makes benchmarking of results across studies and iterative improvement easier. But it also reflects the fact that most
pre-LLM architectures, within which metaphor detection tasks were originally developed, were constrained by relatively
narrow context windows, which made it technically infeasible to process and annotate entire texts at once (Dong
et al., 2023). However, the task, operationalized in this way, has limited construct and ecological validity, and reduces
metaphor identification to a narrow engineering challenge rather than a tool that can enhance our understanding of
metaphor and facilitate its linguistic analysis across discursive domains.
Our study aims to address these limitations and establish a new framework for assessing the capabilities of LLMs
in the task of metaphor identification that better captures the way metaphor actually works in discourse and that
produces usable output to support the analysis metaphor across a wide range of contexts. Modern LLMs not only
exhibit unprecedented contextual sensitivity but are also able to process texts of considerable length. In the experiments
described below, we therefore test LLMs’ metaphor identification capabilities on complete texts. The output of the
methods we test is an XML tagged corpus, which can serve both as the basis for quantitative analyses of metaphor and
as input for a variety of downstream NLP tasks. To overcome the limitations of word-based approaches and enhance the
validity of analysis, we adopt a phraseological approach to metaphor identification. As discussed above, this approach
is not restricted to individual words but accommodates multiword metaphorical expressions, where warranted by the
discursive context.
5

Metaphor identification using large language modelsA PREPRINT
3 Methods
We evaluated LLMs’ ability to identify metaphors by tasking them with labelling all metaphorical expressions in a
corpus of IMDb film reviews using <Metaphor> and</Metaphor> XML tags. The corpus, summarized in Table 1,
was compiled and manually annotated as part of a previous study by two of the authors of this paper based on the
criteria discussed in Section 2.1, achieving robust inter-coder agreement (average Cohen’s kappa = 0.85; for details on
the annotation and inter-coder agreement procedures, see Fuoli et al., 2022). We compared three LLM-based methods:
RAG, prompt engineering, and fine-tuning. Each method is described in turn below.
Table 1: Corpus details.
Total number of texts94
Total number of sentences2,828
Total number of words59,147
Average text length (in words)629.4
Total number of metaphorical expressions2,599
RAG allows the LLM to retrieve relevant information from an external resource before generating a response (Lewis
et al., 2020). In this way, the model can ground its output in additional, task-specific knowledge rather than relying
solely on the information encoded in its training data. In our implementation, we provided the model with the detailed
codebook we developed for annotating the corpus. The codebook, available in the Supplementary Materials, includes
our working definition of metaphor, a set of detailed annotation guidelines, and many illustrative examples. We
prompted the model to refer to the codebook to decide whether a candidate linguistic expression qualifies as a metaphor.
Thus, the RAG-based approach is intended to simulate the normal procedure a human annotator would follow when
annotating metaphorical expressions in text.
The prompt used for RAG, shown in the Supplementary Materials, consisted of two components: a ‘system’ prompt and
a ‘user’ prompt. Following best-practice guidelines, we used the system prompt to establish the general task framing
and define the model’s default behaviour and “persona” (e.g. OpenAI., 2023). Here, we primed the LLM to act as
an expert in metaphor annotation and instructed it to follow the guidelines in the codebook. The last sentence of the
system prompt aimed to ensure the LLM returns only the coded text instead of adding extra explanations, comments,
or formatting (e.g. “Here is your tagged text:”). This was done to prevent the need for further text processing once
the annotation is complete. The user prompt generally specifies the particular question to be addressed – in this case a
request to identify and tag the metaphors in a text.
The second method we evaluated is prompt engineering. Prompt engineering (or “prompting”) refers to the practice of
designing verbal instructions that guide LLMs to produce relevant and accurate outputs. It is one of the most rapidly
evolving areas of research in NLP and AI, with a wide range of strategies devised and tested in prior research (for
an accessible overview, see Boonstra, 2024). In this study, we adapted three of the most widely used and versatile
prompting strategies: zero-shot, few-shot, and chain-of-thought. In zero-shot prompting, the model is provided only
with a task description and must rely solely on its built-in knowledge to generate responses (Brown et al., 2020). The
prompt we used for this strategy is the same we use for the RAG approach but without the second last sentence in the
system prompt instructing the model to consult the codebook.
A common alternative to zero-shot prompting is few-shot prompting, where the model is exposed to a handful of
task-related examples along with their expected outputs before attempting to complete the task (Brown et al., 2020).
This technique leverages ‘in-context learning’, whereby the model infers patterns from the examples provided and
generalizes from them to new, unseen inputs without updating its underlying parameters (Brown et al., 2020). In our
case, we provided the model with a series of examples of manually annotated sentences extracted from our corpus.
These examples were presented as a sequence of alternating ‘user’ and ‘assistant’ messages: each user message asked
the model to identify metaphors in a given sentence, and each assistant message provided the correct annotation. After
the sequence of examples, the model received a final user message containing a full text to be annotated. The system
prompt was the same as the one used in the zero-shot prompt.
The third prompting strategy we tested is few-shot chain-of-thought prompting (CoT), where the examples provided to
the model are supplemented with an explanation of the reasoning behind the correct answer (Wei et al., 2022). This
prompting strategy aims to improve performance by making the reasoning process underlying the task explicit, thus
helping the model more effectively internalize and replicate the decision-making steps. Following He et al. (2024),
we presented the model with annotated examples followed by an explanation of the coding based on our annotation
6

Metaphor identification using large language modelsA PREPRINT
protocol. When examples contained multiple metaphorical expressions, each expression was presented and explained in
turn. We generated all the explanations accompanying the examples using OpenAI’s GPT 4o. We applied a consistent
template, shown below, to streamline the process and ensure standardization.
The word “[WORD]” has a more basic contemporary meaning: in other contexts, it refers to [BASIC
MEANING]. In this example, it describes [FIGURATIVE MEANING] by comparing them to
something literally [BASIC MEANING], which makes it a metaphorical usage.
All generated explanations were manually reviewed for accuracy. Only two minor adjustments were made, which
highlights GPT 4o’s strong inherent capabilities for metaphorical reasoning.
As in the few-shot prompt, the examples and corresponding explanations were presented as a sequence of alternating
‘user’ and ‘assistant’ messages, followed by a final user message that elicited both the annotations and an explanation
of the coding choices. We anticipated that requesting both would improve performance by prompting the model to
explicitly “reason” about its decisions. In addition, the explanations provide valuable insight into patterns in the LLM’s
coding and may serve as input for future iterations of this experiment. The system prompt was slightly adapted by
rephrasing the final instruction to ensure that the model did not suppress the generation of explanations.
While available empirical evidence suggests that providing models with examples generally leads to improved perfor-
mance (e.g. Brown et al., 2020; Shyr et al., 2024), it is less clear what the optimal number of examples is. For example,
Ding et al. (2023) found that for some tasks, 10-shot prompts did not generate as accurate responses as two-shot prompts.
In the context of metaphor identification, Liang et al. (2025) observed that including a single example enhanced LLM
performance, but adding further examples yielded no additional gains. To explore how the number of examples affects
performance in our task, we varied both the few-shot and CoT prompts by including either four or eight examples.
Another important aspect we considered when selecting the examples was the type of metaphor they included. In
addition to identifying metaphorical expressions, Fuoli et al. (2022) classify them as either ‘conventional’ or ‘creative’.
Conventional metaphors are those that make use of widely attested comparisons without changing them in any way,
whereas creatively used metaphors involve drawing what look like completely new comparisons, or extending existing
comparisons in novel ways (Turner and Littlemore, 2023). The ratio of conventional to creative metaphors in the corpus
is approximately 9:1. To assess whether LLMs are sensitive to this distinction, we created two versions of the few-shot
and CoT prompts, one preserving the original distribution and another balancing the examples to include an equal
number of conventional and creative metaphors.
The complete prompt set is provided in the Supplementary Materials. The wording of all the prompts was iteratively
refined through trial runs on small batches of texts and through ‘meta-prompting’, a common technique in which LLMs
are asked to evaluate and improve the prompt itself.
The third approach we tested is model fine-tuning. Fine-tuning refers to the process of further training a language model
on a task-specific dataset to improve performance (Gururangan et al., 2020). Unlike prompt engineering, fine-tuning
modifies the model itself by adjusting its internal parameters via supervised learning on labelled data directly relevant to
the target task. In our implementation, we fine-tuned the LLMs on 80% of our film review corpus, using the remaining
20% as a held-out test set to evaluate performance.
Table 2 provides an overview of the methods tested and their corresponding variants. We tested these approaches using
the models shown in Table 3. We compare models of different sizes to assess whether size affects performance on
the metaphor identification task. The size of the LLMs developed by OpenAI is not publicly available, but reliable
estimates put them at several orders of magnitude larger than the relatively small open-source models we selected;
GPT 4.1, for instance, is estimated to contain 1.8 trillion parameters (GlobalGPT, nd). The ‘mini’ and ‘nano’ variants
represent the medium- and small-sized versions of their respective base models (GPT 4.1 and o3). Smaller models
have a lower energy and environmental impact and are therefore preferable to larger ones. In addition, the open-source
models selected can all be run locally on modern consumer-grade hardware, affording users greater control and privacy.
Evaluating the performance of open-source models is also important for widening access to this technology and
improving the replicability of LLM-based research. We also include both reasoning and non-reasoning models in our
evaluation, where reasoning refers to the model’s capacity to perform multi-step logical inferences beyond simple
pattern recognition.
The models were accessed via their Application Programming Interfaces (APIs) to enable batch execution of prompts.
For each method and prompt variant tested, we conducted five experimental runs to assess the consistency of the
output. The ‘temperature’ parameter was left at its default setting. In the case of fine-tuning, each run used a different
sample for both the training and test sets to ensure that results were not biased by the specific composition of the data.
Fine-tuning was not performed with the reasoning models because, unlike standard models, they are trained using
multi-stage procedures that combine supervised learning with reinforcement learning, which relies on expert human
7

Metaphor identification using large language modelsA PREPRINT
Table 2: Overview of methods tested and their variants.
Method Strategy Number of examples Creative vs conventional
metaphor ratio
RAG N/A N/A N/A
Prompt engineering Zero-shot 0 N/A
Few-shot 4 Even
4 Original
8 Even
8 Original
Chain-of-thought 4 Even
4 Original
8 Even
8 Original
Fine-tuning N/A N/A N/A
Table 3: Large Language Models used in experiments.
Model Parameter size Accessibility Developer Reasoning capabilities
Llama 3.1 8 billion Open source Meta No
Llama 3.2 3 billion Open source Meta No
Llama 3.2 1 billion Open source Meta No
Deepseek R1 8 billion Open source DeepSeek Yes
GPT 4.1 Not disclosed Closed source OpenAI No
GPT 4.1 mini Not disclosed Closed source OpenAI No
GPT 4.1 nano Not disclosed Closed source OpenAI No
o3 Not disclosed Closed source OpenAI Yes
o3 mini Not disclosed Closed source OpenAI Yes
o4 mini Not disclosed Closed source OpenAI Yes
feedback to develop step-by-step reasoning capabilities. In contrast, non-reasoning models can be fine-tuned using
supervised learning alone (i.e. training the model on labelled examples without additional reinforcement or human
feedback steps), a capability that is fully supported through the standard fine-tuning APIs.
While, as discussed above, LLMs can be used without advanced programming skills, accessing them via their API does
require some basic programming knowledge. To ensure that our workflow is as accessible and replicable as possible, we
have created a set of annotated Jupyter Notebooks that include and explain the Python code used to run our experiments.
These resources are publicly available via the following GitHub repository: https://github.com/Weihang-Huang/
MetaphorIdentification . The repository also contains the manually annotated corpus, the uncoded corpus, the
codebook used for annotation and RAG prompting, and the full set of complete prompts tested in this study.
4 Results
This section presents the results of our experiments. We begin by reporting quantitative data on the LLMs’ performance
across the tested methods and prompting strategies. We then qualitatively analyse the output of the best-performing
method to shed light on the types of inconsistencies observed between humans and LLMs, and to gain insights into how
both LLM performance and the metaphor annotation protocol might be improved. All data and code for the quantitative
analysis are available on our GitHub repository.
8

Metaphor identification using large language modelsA PREPRINT
4.1 Model performance
We evaluated the models’ performance against our “gold standard” hand-coded corpus using ‘precision’, ‘recall’,
and ‘F1’ measures. Precision quantifies how many of the identified metaphors correctly match our corpus, recall
captures how many of the metaphors in the corpus the model successfully identified, and F1 combines both into
a single harmonic score. These scores were calculated at the token level, meaning that each individual token was
assessed independently as either coded or uncoded, rather than requiring an exact span match in the case of multi-word
metaphorical expressions. This approach allowed us to capture cases where the model’s output overlapped only partially
with the human annotations and to award proportional credit for these partial matches. All examples used in few-shot
prompts were kept in the corpus but excluded from the performance assessment.
Figure 1: Median F1 scores with corresponding distributions across models and core methods. The line inside each box
represents the median, the box spans the interquartile range (IQR), notches approximate 95% confidence intervals for
the median, and whiskers indicate variability beyond the quartiles.
Figure 1 shows the median F1 scores and distributions of model performance across the three core methods and the 10
LLMs tested. Fine-tuning yielded the highest accuracy, followed by prompt engineering, and RAG. Within fine-tuning,
the best-performing model was GPT-4.1 mini, which achieved a median F1 score of 0.79 (IQR: 0.73–0.83). The
reasoning o3 model achieved the highest performance for both prompt engineering (median: 0.75, IQR: 0.71–0.79) and
RAG (median: 0.74, IQR: 0.69–0.78). Overall, Figure 1 shows that closed-source models outperformed open-source
models, with the latter displaying not only lower F1 scores but also greater variability in accuracy.
To test whether these differences are statistically significant, we fit a mixed-effects beta regression model using the
glmmTMBpackage in R (Brooks et al., 2017). The model included method, model type (open- vs closed-source), and
text length as fixed effects, with random intercepts for model, text ID, and experimental run (i.e. the five repetitions of
each approach). F1 scores were adjusted using the Smithson–Verkuilen transformation to fit within the (0,1) interval
(Smithson and Verkuilen, 2006), and text length was centred to improve model stability.
The analysis revealed a significant effect of method on F1 scores. Compared to RAG, both prompt engineering ( β
= 0.11, p < 0.001, ∆F1 = 0.02) and fine-tuning ( β= 0.43, p < 0.001, ∆F1 = 0.10) achieved significantly better
performance, with fine-tuning also reliably outperforming prompt engineering ( β= 0.33, p < 0.001, ∆F1 = 0.07).
Overall, closed-source models performed substantially better than open-source ones ( β= 0.84, p < .001, ∆F1 = 0.20).
As shown in Figure 1, closed-source models were also more responsive to the choice of method. For instance, GPT-4.1
showed much larger estimated gains when moving from RAG to fine-tuning ( β= 0.73, p < .001, ∆F1 = 0.15) compared
to an open-source model such as Llama 3.1 8B ( β= 0.08, p < .001, ∆F1 = 0.02). Text length had a significant negative
effect ( β= -0.10, p < 0.001, ∆F1 = -0.02 per one SD increase), indicating that longer texts were more challenging to
annotate accurately.
9

Metaphor identification using large language modelsA PREPRINT
Figure 2: Median F1 scores with corresponding distributions across models and core prompt engineering strategies.
The line inside each box represents the median, the box spans the interquartile range (IQR), notches approximate 95%
confidence intervals for the median, and whiskers indicate variability beyond the quartiles.
Next, we zoomed in on prompt engineering to examine performance differences across the three core prompting
strategies. Figure 2 reveals a clear performance gap between open-source and closed-source models, with the former
showing both lower accuracy and less sensitivity to the prompting strategy. Among the closed-source models, we
observed a marked difference between reasoning and non-reasoning models, with the former substantially outperforming
the latter. The performance of closed-source models progressively improved across the three prompting strategies, with
chain-of-thought prompts consistently outperforming the others. This suggests that, at least for large closed-source
models, providing examples and explicit reasoning steps does enhance performance. The best-performing configuration
used the o3 model with chain-of-thought prompting, achieving a median F1 score of 0.76 (IQR: 0.72–0.80). These
findings indicate that chain-of-thought prompting with best-in-class reasoning LLMs can achieve accuracy levels
comparable to fine-tuning, but with the significant advantage of requiring only a small number of annotated examples.
Even with a basic zero-shot prompt, closed-source reasoning LLMs demonstrated robust performance, with o3 achieving
a median F1 score of 0.73 (IQR: 0.69–0.77). This indicates that these LLMs already have a rich internalized “knowledge”
that enables them to carry out this task quite successfully.
To assess whether differences between the three core prompting strategies were statistically significant, we fitted a
mixed-effects beta regression model with prompting strategy, model type, and text length as fixed effects, and random
intercepts for model, text ID, and experimental run. As above, F1 scores were adjusted using the Smithson–Verkuilen
transformation, and text length was mean-centred. The results confirm a significant effect of prompting strategy
on model performance. Compared to the zero-shot baseline, both few-shot prompting ( β= 0.10, p < .001, ∆F1 =
0.02) and chain-of-thought prompting ( β= 0.20, p < .001, ∆F1 = 0.05) led to significant improvements in F1 scores.
Chain-of-thought prompting yielded the best results, significantly outperforming not only zero-shot prompting but also
few-shot prompting ( β= 0.10, p < .001, ∆F1 = 0.02). This result suggests that adding both examples and explanations
substantially enhances performance. Closed-source models significantly and substantially outperformed open-source
ones ( β= 0.84, p < .001, ∆F1 = 0.20). As observed above, text length was negatively associated with performance ( β=
-0.15, p < .001,∆F1 = -0.04).
Lastly, we tested the effects of (i) increasing the number of examples included in the prompt and (ii) varying the ratio of
creative to conventional metaphors within those examples. The results show that both including four examples ( β=
0.13, p < .001, ∆F1 = 0.03) and eight examples ( β= 0.16, p < .001, ∆F1 = 0.04) led to significant improvements in
model performance compared to the zero-shot baseline. There was also a statistically reliable but small improvement
when increasing the number of examples from four to eight ( β= 0.03, p < .001, ∆F1 = 0.01), indicating a marginal
benefit from additional in-context data but with diminishing returns as the number of examples grows. Example sets
10

Metaphor identification using large language modelsA PREPRINT
that preserved the original ratio of conventional-to-creative metaphor ratio yielded a small but significant performance
gain over those with an even ratio (β= 0.04, p < .001,∆F1 = 0.01).
4.2 Analysis of human-LLM discrepancies
After completing the quantitative evaluation of LLM performance, we turned to a more in-depth analysis of the
discrepancies between the LLM-generated and human-generated metaphor coding. We focused on the annotations
produced by the fine-tuned GPT 4.1 mini model, both because it achieved the highest accuracy and because its output is
of a manageable size, as 80% of the corpus was used to train the LLM. The sample we examined comprises 19 texts.
The aim of this analysis was to identify the nature of the discrepancies, to better understand the possible reasons for
them.
We examine both ‘false negative’ discrepancies, where the human coders identified a stretch of text as metaphor but the
model did not, and ‘false positive’ discrepancies, where the model identified a stretch of text as metaphor when the
human coders did not. Two coders (both authors of this paper) worked together to identify reasons for each instance of
a discrepancy. They began by discussing 10% of the discrepancies in order to create a preliminary set of categories.
They then coded the remaining discrepancies independently in Nvivo (Lumivero, 2025), discussing and refining the
categories as they did so. They then met for a number of joint coding sessions where they discussed and resolved
borderline cases. The full list of possible reasons that were identified for false negative and false positive discrepancies
is shown in Table 4. Note that a single word or stretch of text could be coded into more than one category. As a result,
the numbers reported here should not be interpreted as precise measures of output accuracy (which is instead provided
above) but rather as heuristics for identifying the predominant sources of errors in the LLM coding. In a small number
of cases (26), it was impossible to propose a reason for the discrepancy, so they were not included in the analysis below.
Table 4: Types of discrepancies observed in output from o3, fine-tuning method.
Discrepancy Type False negative False positive
Decomposability 5 4
Degree of conventionality 59 12
Difficulties in perceiving a source–target distinction 34 56
Explicit comparisons 10 0
Grammatical uses that cue metaphoricity 21 7
Difficulties in perceiving a relationship based on comparison 0 12
Personification 30 7
Phrase-level meaning 22 26
Source target confusion and twice true 8 6
Total 189 129
Below we discuss each of these reasons in turn, beginning with those that were more likely to lead to false negative
discrepancies, then discussing those that led, more or less equally, to discrepancies in both directions and ending with
those that were more likely to lead to false positive discrepancies. In each sub-section, the reasons are discussed in
order of their occurrences, beginning with those that occurred most frequently.
4.2.1 Principal reasons for false negative discrepancies
Degree of conventionality.A first reason for false negative discrepancies was that the model appeared to miss
metaphors that are highly conventional in English, especially metaphors that do not have a literal alternative, such as
those used for time. For example, in a number of cases, the model failed to identify cases of metaphors where time was
expressed in terms of space, as in the case ofthe verynearfutureand cases of metaphor where time was expressed
as a resource, as in:anyone who hasn’tspenttheir life in seclusion. Many of the instances identified in this category
involved source domains relating to size, space, or containment (e.g.the film contains. . . ).
Personification.A second reason for false negative discrepancies was that the model appeared to miss metaphor that
could be considered to involve personification. An example of one such item in this category is:the fury mother nature
can unleash, where the whole phrase was missed by the model. Here the human coders sawnatureas personified as a
(female) human capable of expressing emotions (in this casefury).
11

Metaphor identification using large language modelsA PREPRINT
Explicit comparisons.A third source of discrepancy where the model missed examples that had been coded as
metaphor by the human coders was that of explicit comparisons. In some cases, the metaphors were flagged in such a
way as to make them resemble similes. For example, one of the films under review is described as beingabout as funny
as a root canal. This was coded as metaphorical by the human coders, as a metaphorical comparison is being drawn
between the experience of watching the film and the experience of having painful dental treatment. In this case, the
model may have been operating on the premise that similes and metaphors are different entities; a piece of folk wisdom
that is often taught in schools. However, this distinction (between similes and metaphors) is somewhat arbitrary as it
simply singles out two ways of signalling that a metaphorical comparison is intended (likeandas) and treats them as if
they are somehow “special”. In reality, intended metaphorical meanings are signalled in many different ways, through
the use of ‘tuning devices’ such askind of,sort of,metaphorically speaking, and even the wordliterally(see Cameron
and Deignan, 2003). For this reason, many metaphor researchers do not draw a tight distinction between metaphors and
similes, although some coding schemes such as the MIPVU (Steen et al., 2010) draw a distinction between ‘indirect’
metaphors (metaphors that are not signalled as such) and ‘direct’ metaphors (metaphors that are signalled as such).
Grammatical uses that cue metaphoricity: crossing word class boundaries.The final source of false negative
discrepancies related to examples where the grammatical structure of the language seemed to cue metaphoricity, a
phenomenon that was first observed by Deignan (2005). For example, the use of words of particular word classes at
times favoured metaphorical interpretation. One example of this category is found in the expressionthings. . .gave me
glassyeyes(i.e. they brought the writer to tears). This is a way of describing tear-filled eyes by likening them to a pane
of glass. However, the adjectival derivation ofglassis always metaphorical, which may explain why the model missed
this metaphor.
Another example that relates to this category is the use of the wordapproachin the sentenceI found thisapproachto
be very clever. The nominal use ofapproachtends to be associated with metaphorical meanings, and the verbal use
with literal meanings. Again, the model did not code this example as metaphorical, although the human coders did.
This category was also used for cases where the metaphoricity of the term came out of its grammatical role in the
sentence. For example, the phrasebathed in, when used in a literal context (e.g.I bathed in the sea) tends to carry a
human actor. However, when used metaphorically (e.g.she was bathed in soft light), the phrase tends to appear in the
passive voice. The model tended to miss such cases, suggesting that human coders were more attuned to grammatical
cues that signal metaphoricity.
4.2.2 Principal reasons for discrepancies that occurred in both directions
In some cases, the discrepancies did not skew towards one type or another. There were four main reasons behind such
discrepancies.
Difficulties in perceiving a source-target distinction.The first reason for discrepancies seen equally in false negative
and false positive contexts occurred when it was difficult to draw a distinction between the source and target domains.
These examples lie on the cusp of being literal or metaphorical. For example, in the following extract, the model coded
the phrasejoin the listas metaphorical:
With many big-budget science fiction films, great ideas are often wasted by bad scripts, cheesy
plot twists, and terrible acting. The Fifth Element, The Abyss, and Godzilla had great concepts
squandered by bad acting, writing, or both. At first glance, The Matrix, Larry & Andy Wachowski’s
sci-fi/kung-fu/shoot-em-up spectacular, looks like a prime candidate tojoin the listof high-concept
bad movies [. . . ]
In this example, one might say that the reviewers had actually produced a list of threehigh concept bad moviesso the
use of the term list here is, in some ways, literal. On the other hand, there is a degree of ambiguity in this example.
The so-called list may just have been a way of saying that these three movies were all terrible. When the review talks
about the list of high concept bad movies, there is a degree or irony and a faint suggestion that this is some sort of
“official” list, which of course does not exist. The model may have picked up on this idea, which goes some way towards
explaining why it coded it as metaphorical: because an “official” list such as this does not actually exist.
In a second example, the model codedprediction gameas metaphorical in the following sentence:
You can tell a script is terrible when you are able to predict what will happen minutes before it does.
This littleprediction gameis a very fun exception.
Here the viewer may well be playing a game with themselves; although the game itself does not meet all the criteria for
being a prototypical game, it does meet some of them, as the activity is fun and playful, and involve a degree of chance.
12

Metaphor identification using large language modelsA PREPRINT
A number of items in this category involved the use of sight terms as a source domain (e.g.examiningconcepts). As
Littlemore (2015) discuss, there is often a degree of conflation between the literal and the metaphorical in cases such
as these. Interestingly, this conflation seems to have been picked up by our coding scheme. Sometimes, a potential
relationship of metonymy could be perceived in these examples. For example, in the phrasepoundingdrum beats, there
is no clear separation between the source and target domains; instead, the action ofpoundinga drum is used to describe
the resulting sound.
Phrase level meaning.A second reason for the discrepancies that occurred in both directions was related to the
fact that it is not always clear whether the metaphoricity lies at the level of the word or at the level of the phrase. In
such cases, both the model and the human coders sometimes tagged the whole phrase as metaphorical, and sometimes
simply tagged the part of the phrase that was most saliently metaphorical. For example for the phrasea member of that
undistinguished club, the model identified the wordclubas metaphorical whereas the human coders coded the whole
phrase as such. Similarly, in the phrase,left behind in the romantic wake, the model identifiedleft behindandwake, but
not the other words in the phrase. The human raters coded the whole phrase as metaphorical as it represented a single
metaphorical scene.
Decomposability.A third reason for the discrepancies that occurred in both directions was that in some cases the
metaphorical component could be found only in one part of a lexical item and could therefore be considered to operate
below the level of the word. An example of this is the termeye-catching, which was identified as metaphorical by the
model but not by the human coders. Here the wordeyeis used metonymically to refer to the act of seeing but the term
catchingis more metaphorical as it reifies the reception of the visual information.
Source-target confusion and twice-true metaphors.A fourth reason for discrepancies that appeared in both
directions concerned metaphorical expressions that were also literally true in the context of the film reviews. This
situation proved challenging for both the human coders and for the model. This is clearly a grey area in metaphor
identification, and perhaps needs to be addressed explicitly when instructing LLMs in the practice of metaphor
identification. One example of a word that was coded by the human coders as metaphorical but not the model was the
worddarkin the following passage:
While the original Alien film was a dark, enclosed horror film [. . . ]
Here, the worddarkcould be being used literally but it could also be being used metaphorically, and in fact was
probably being used in both ways.
Another example of a stretch of text that was coded as metaphorical by the human coders but not by the model involved
the phrasetells the talein the following sentence:
[. . . ] The Mod Squad tells the tale of three reformed criminals under the employ of the police to go
undercover.
This sentence could be paraphrased as saying that the film [The Mod Squad] isaboutthree reformed criminals. However,
the writer chooses to use the formulationtells the taleinstead. Although it is highly conventional to talk about a film
telling a particular story, the idea of “telling a tale” creates an image of someone (perhaps a grandparent) perhaps sitting
down in an armchair and starting to recount a long and traditional story. There is therefore potential for a metaphorical
comparison here though it is by no means clearcut.
4.2.3 Principal reasons for false positive discrepancies
Difficulties in perceiving a contemporary relationship between the source and target that was based on com-
parison.There was only one clear reason for false positive discrepancies. This related to the ease with which it was
possible to identify a contemporary relationship between the source and target that was clearly based on similarity and
comparison. For example, the model codedred herringas metaphor whilst the human coders did not do so. Whilst the
expressionred herringis clearly not literal, it is difficult to understand its meaning in context by identifying any kind of
similarity between misleading clues and red herrings. This may be due to the metaphors being historically older, where
the historical mappings have fallen away.
4.2.4 Discussion of human-LLM discrepancies
Interestingly, many of the discrepancies closely mirror issues that are often discussed in the metaphor literature. The
first issue concerns the nature of the different types of figurative language, and the relationships between them. We saw
13

Metaphor identification using large language modelsA PREPRINT
examples above of discrepancies relating to the blurred relationship between personification (the fury mother nature
can unleash) and metaphor, and between metonymy and metaphor (eye-catching). There is some debate over whether
personification should be viewed as a kind of metaphor, and a persuasive body of literature showing how personification
can cross the spectrum from metaphorical to literal language (Dorst, 2011).
The second issue concerns the level at which metaphor operates. In the metaphor literature, there is much debate
over whether metaphor operates at the level of the word or at the level of the phrase, and over the role played by
metaphorically used words within a whole phrase. Some words contribute disproportionately to the overall metaphoricity
of a phrase, and the model appears to be picking these words up. However, the human coders had a stronger tendency to
code the whole phrase as metaphorical as it was thought to constitute a single metaphorical scene (Johansson Falck and
Okonski, 2023). Equally, most metaphor coding schemes do not allow for cases where metaphor is operating below the
level of the lexical unit, and this includes cases of hyphenated words, such aseye-catching.
Conventionality sometimes shades into semantic bleaching. Some words do not retain sufficiently strong links to their
original concrete meaning to be considered metaphor. For instance, metaphors representing time in terms of space fall
into this category as they are often highly conventional and therefore have the potential to be easily be missed by both
human coders and models. Issues such as these, which are pervasive throughout the metaphor literature, explained a
large number of the inconsistencies between the human and the LLM coding. In both types of coding, “rules” relating
to the relationship between metaphor and other types of figurative language, the level at which metaphor could be
located (word, phrase, or sub-lexical unit), and conventionality were difficult to apply as all of these phenomena operate
along continua and do not fall neatly into discrete categories. Humans struggle with artificial dichotomies, and perhaps
because LLMs are based on human data, they also appear to struggle with them. This finding is unsurprising. The very
act of attempting to say whether or not a particular language item is or is not metaphorical is an artificial endeavour as
language does not operate in either/or categories. Metaphoricity, like any other aspect of language is best seen as a
radial category with more or less prototypical examples (Taylor, 2003) and there are no universal, context-independent
cut-off points. Therefore, the fact that LLMs fail to consistently apply false dichotomies renders them human-like.
5 Conclusion
As a pervasive feature of discourse, metaphor offers a productive lens for examining cognition, emotion, and ideology.
Yet large-scale analysis has so far been limited by the need for manual annotation, which stems from the context-
sensitive nature of metaphor. In this study, we explored the potential of LLMs to automate metaphor identification,
with the ultimate aim of making metaphor analysis more scalable and its findings more robust and generalisable. We
assessed, for the first time, the ability of LLMs to detect and annotate metaphors in full texts. This approach offers
greater ecological validity than previous work, which largely relied on single, decontextualized sentences, and produces
fully annotated texts that can be used directly in downstream analyses. In addition, we extended prior research by
evaluating not only prompt engineering but also retrieval-augmented generation (RAG) and fine-tuning.
The results of our experiments demonstrate that state-of-the-art closed-source LLMs can identify and annotate metaphor
in full texts with a high level of accuracy. The best median F1 score of 0.79, achieved through fine-tuning, is
substantially higher than results from previous studies that evaluated LLMs on the simpler task of annotating individual
words in isolated sentences (Liang et al., 2025; Hicke and Kristensen-McLachlan, 2024), and comes close to the best
scores reported for the even simpler task of annotating pre-selected single words (Puraivan et al., 2024; Tian et al.,
2024). Crucially, humans themselves often disagree on what constitutes a metaphor. Even well-established metaphor
identification protocols fall short of perfect inter-coder agreement, with reported kappa scores ranging from 0.56 to
0.70 for MIP (Group, 2007) and from 0.74 to 0.88 for MIPVU (Steen et al., 2010). Thus, while F1 and ‘kappa’ are not
directly comparable, it seems reasonable to conclude that LLM performance is not far from human levels of reliability.
Our comparative analysis of human and LLM outputs lends further support to this conclusion by showing that most
discrepancies are not random but reflect well-known grey areas and ontological challenges that metaphor scholars have
long grappled with.
Based on these results, we propose that LLMs offer a viable means of at least semi-automating metaphor identification
and annotation. As noted at the outset, LLMs provide an accessible and flexible way to efficiently annotate large text
corpora, thereby freeing up time and resources for higher-level interpretation and theory-building. Crucially, we are not
advocating the use of LLMs as “black boxes” whose outputs are accepted uncritically. Instead, we envisage a recursive,
human-in-the-loop workflow consisting of three main stages. In the development stage, different LLM methods are
tested and iteratively refined through close analysis of small output samples, building on best practices from previous
research, including our own study. Once the most effective approach has been identified and its accuracy optimized,
it can then be applied to the full corpus. Finally, if preliminary tests indicate that the model’s accuracy falls short of
the project’s requirements, a human analyst can manually correct the LLM-generated annotations. Even when this
14

Metaphor identification using large language modelsA PREPRINT
step is necessary, it should be considerably less time-consuming and cognitively demanding than annotating the texts
entirely from scratch. The manually corrected dataset could then be used to fine-tune an LLM to further expand the
analysis. Given the robust results we obtained using a small corpus, we expect there to be considerable scope for further
improvement with larger training datasets.
Crucially, however, we do not view LLMs merely as a methodological tool for increasing the scale of analyses. Our
examination of discrepancies between LLM and human output highlights their potential as a testbed for developing
and refining theoretical understandings of metaphor, including the boundaries between metaphor and other figurative
tropes. In addition, LLMs can serve as a means to benchmark different metaphor annotation procedures and iteratively
refine them, in much the same way that inter-coder reliability testing is used to evaluate and refine human annotation
protocols (Fuoli, 2018).
More broadly, our study contributes to the rapidly growing body of evidence that establishes LLMs as a useful tool for
automating a wide range of linguistic annotation tasks (e.g. Gao and Feng, 2025; Mahmoudi-Dehaki and Nasr-Esfahani,
2025; Morin and Marttinen Larsson, 2025; Yu et al., 2024b,a). Our results offer a counterpoint to some of the concerns
raised by Curry et al. (2025, p. 9) regarding the suitability of LLMs for corpus annotation. First, contrary to the
authors’ claim, our study shows that LLMs do not require extensive training data to achieve good performance (beyond,
of course, the data used for pre-training the model itself). The hand-coded dataset used for fine-tuning in our study
included only a few thousand words, and even with as few as eight annotated sentences we obtained high-quality
results. Second, LLMs do not demand an excessive upfront time investment – certainly no more than the effort typically
required to develop a codebook for manual annotation and to test its reliability through inter-coder agreement. Moreover,
the LLM-based approach offers clear incremental advantages: now that we have carried out the groundwork of setting
up and evaluating different methods and prompt variants, the most effective approach can be further refined in future
iterations. Third, while LLM output does contain errors (just as human coding does), these errors are largely systematic
and interpretable, making them amenable to targeted correction and improvement. Finally, while Curry et al. (2025) are
certainly right to highlight the environmental concerns associated with LLMs, these concerns should be weighed against
the human costs and ethical implications of employing research staff – often graduate students and junior researchers –
to carry out labour-intensive and tedious annotation tasks for extended periods of time.
Our study provides several promising directions for future research to further explore the potential and applications
of LLMs in metaphor analysis. First, future work could build on our analysis of discrepancies between human and
LLM output by refining prompts and providing tailored instructions and examples to reduce ambiguity and improve
accuracy. The three main methods we tested could also be combined to assess whether such integration has a multiplier
effect on performance. Second, future studies could replicate our experiments using alternative annotation procedures,
such as MIP(VU), to determine which performs best and to test whether our improved results compared to previous
LLM-based studies are due to our techniques and prompts or to the phraseological approach we adopted. Put differently,
is a phraseological approach to metaphor inherently more intuitive, and does this explain why we obtained better results
than earlier studies relying on MIP(VU)? Third, the approaches developed here could be applied and evaluated in
other discursive domains to examine whether factors such as genre or topic influence LLM performance. Additional
LLMs could also be tested, including large open-source models such as Llama 3 405B and Mistral 124B. Finally, while
our study focused on metaphor identification, an equally important and challenging task for future LLM evaluation
is metaphor labelling: identifying the metaphorical target and source domains. In sum, our findings suggest that
LLMs have the potential to become a valuable component of the metaphor analysis workflow, and we hope this study
encourages further research into this emerging area.
Acknowledgments
We are grateful to Jack Grieve and Marcus Perlman (University of Birmingham) for their helpful feedback and
discussions on this study.
References
Birke, J. and Sarkar, A. (2006). A clustering approach for nearly unsupervised recognition of nonliteral language.
InProceedings of the 11th Conference of the European Chapter of the Association for Computational Linguistics,
Trento, Italy, pages 329–336. Association for Computational Linguistics.
Boonstra, L. (2024). Prompt engineering. https://www.kaggle.com/whitepaper-prompt-engineering.
Brooks, M. E., Kristensen, K., van Benthem, K. J., Magnusson, A., Berg, C. W., Nielsen, A., Skaug, H. J., Maechler, M.,
and Bolker, B. M. (2017). glmmTMB balances speed and flexibility among packages for zero-inflated generalized
linear mixed modeling.The R Journal, 9(2):378–400.
15

Metaphor identification using large language modelsA PREPRINT
Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., others, and Amodei, D. (2020). Language
models are few-shot learners.Advances in neural information processing systems, 33:1877–1901.
Cameron, L. (2003).Metaphor in Educational Discourse. A&C Black.
Cameron, L. (2010a). Responding to the risk of terrorism: the contribution of metaphor.DELTA: Documentação de
Estudos em Lingüística Teórica e Aplicada, 26:587–614.
Cameron, L. (2010b). What is metaphor and why does it matter? In Cameron, L. and Maslen, R., editors,Research
Practice in Appied Linguistics, Social Sciences and the Humanities, pages 3–25. Equinox Publishing.
Cameron, L. and Deignan, A. (2003). Combining large and small corpora to investigate tuning devices around metaphor
in spoken discourse.Metaphor and Symbol, 18(3):149–160.
Curry, N., McEnery, T., and Brookes, G. (2025). A question of alignment–AI, GenAI and applied linguistics.Annual
Review of Applied Linguistics.
Deignan, A. (2005).Metaphor and Corpus Linguistics. John Benjamins, Amsterdam.
Ding, B., Qin, C., Liu, L., et al. (2023). Is GPT-3 a good data annotator? In Rogers, A., Boyd-Graber, J., and Okazaki,
N., editors,Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1,
Long Papers), pages 11173–11195. Association for Computational Linguistics.
Dong, Z., Tang, T., Li, J., and Zhao, W. X. (2023). A survey on long text modeling with transformers.arXiv preprint
arXiv:2302.14502.
Dorst, A. G. (2011). Personification in discourse: Linguistic forms, conceptual structures and communicative functions.
Language and Literature, 20(2):113–135.
Fuoli, M. (2018). A stepwise method for annotating appraisal.Functions of Language, 25(2):229–258.
Fuoli, M., Littlemore, J., and Turner, S. (2022). Sunken ships and screaming banshees: Metaphor and evaluation in film
reviews.English Language & Linguistics, 26(1):75–103.
Gao, Q. and Feng, D. (2025). Deploying large language models for discourse studies: An exploration of automated
analysis of media attitudes.PloS One, 20(1):e0313932.
Ge, M., Mao, R., and Cambria, E. (2023). A survey on computational metaphor processing techniques: From
identification, interpretation, generation to application.Artificial Intelligence Review, 56(Suppl 2):1829–1895.
GlobalGPT (n.d.). GPT 4.1 model: Advanced OpenAI language model & features.GlobalGPT. Retrieved 2025-08-08.
https://www.glbgpt.com/sitepage/gpt-4-1.
Group, P. (2007). MIP: A method for identifying metaphorically used words in discourse.Metaphor and Symbol,
22(1):1–39.
Gururangan, S., Marasovic, A., Swayamdipta, S., Lo, K., Beltagy, I., Downey, D., and Smith, N. A. (2020). Don’t
stop pretraining: Adapt language models to domains and tasks. In58th Annual Meeting of the Association for
Computational Linguistics, ACL 2020, pages 8342–8360. Association for Computational Linguistics (ACL.
He, X., Lin, Z., Gong, Y ., Jin, A.-L., Zhang, H., Lin, C., Jiao, J., Yiu, S. M., Duan, N., and Chen, W. (2024). AnnoLLM:
Making large language models better crowdsourced annotators. InProceedings of the 2024 Conference of the North
American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 6:
Industry Track), pages 165–190. Association for Computational Linguistics.
Hicke, R. M. M. and Kristensen-McLachlan, R. D. (2024). Science is exploration: Computational frontiers for
conceptual metaphor theory. InProceedings of the Computational Humanities Research Conference 2024 (CHR
2024), pages 1–12, Aarhus, Denmark.
Johansson Falck, M. and Okonski, L. (2023). Procedure for identifying metaphorical scenes (PIMS): The case of spatial
and abstract relations.Metaphor and Symbol, 38(1):1–22.
Lakoff, G. and Johnson, M. (1980).Metaphors We Live By. University of press, Chicago.
Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V ., Goyal, N., others, and Kiela, D. (2020). Retrieval-augmented
generation for knowledge-intensive nlp tasks.Advances in Neural Information Processing Systems, 33:9459–9474.
Liang, J., Dorst, A. G., Prokic, J., and Raaijmakers, S. (2025). Using GPT-4 for conventional metaphor detection in
English news texts.Computational Linguistics in the Netherlands Journal, 14:307–341.
Littlemore, J. (2015).Metonymy: Hidden Shortcuts in Language, Thought and Communication. Cambridge University
Press.
Littlemore, J. and Turner, S. (2020). Metaphors in communication about pregnancy loss.Metaphor and the Social
World, 10(1):45–75.
16

Metaphor identification using large language modelsA PREPRINT
Lumivero (2025). Nvivo (version 14.24) [computer software]. https://lumivero.com/products/nvivo/.
Mahmoudi-Dehaki, M. and Nasr-Esfahani, N. (2025). Automated vs. manual linguistic annotation for assessing
pragmatic competence in english classes.Research Methods in Applied Linguistics, 4(3):100253.
Mohammad, S., Shutova, E., and Turney, P. (2016). Metaphor as a medium for emotion: An empirical study. In
Proceedings of the Fifth Joint Conference on Lexical and Computational Semantics, pages 23–33.
Morin, C. and Marttinen Larsson, M. (2025). Large corpora and large language models: a replicable method for
automating grammatical annotation.Linguistics Vanguard.
Musolff, A. (2016).Political Metaphor Analysis: Discourse and Scenarios. Bloomsbury.
OpenAI (2023). Best practices for prompt engineering with openai api. https://platform.openai.com/docs/guides/prompt-
engineering.
Puraivan, E., Renau, I., and Riquelme, N. (2024). Metaphor identification and interpretation in corpora with ChatGPT.
SN Computer Science, 5(8):1–10.
Semino, E. (2008).Metaphor in Discourse. Cambridge University Press, Cambridge.
Semino, E., Demjén, Z., Hardie, A., Payne, S., and Rayson, P. (2017).Metaphor, Cancer and the End of Life: A
Corpus-based Study. Routledge.
Shyr, C., Hu, Y ., Bastarache, L., Cheng, A., Hamid, R., Harris, P., and Xu, H. (2024). Identifying and extracting rare
diseases and their phenotypes with large language models.Journal of Healthcare Informatics Research, 8(2):438–461.
Smithson, M. and Verkuilen, J. (2006). A better lemon squeezer? maximum-likelihood regression with beta-distributed
dependent variables.Psychological Methods, 11(1):54.
Steen, G. (2009). Three kinds of metaphor in discourse: A linguistic taxonomy. In Musolff, A. and Zinken, J., editors,
Metaphor and Discourse. Palgrave Macmillan, London.
Steen, G. J., Dorst, A. G., Krennmayr, T., Kaal, A. A., and Herrmann, J. B. (2010).A Method for Linguistic Metaphor
Identification: From MIP to MIPVU. John Benjamins.
Taylor, J. R. (2003).Linguistic Categorization. Oxford University Press.
Tian, Y ., Xu, N., and Mao, W. (2024). A theory guided scaffolding instruction framework for LLM-enabled metaphor rea-
soning. InProceedings of the 2024 Conference of the North American Chapter of the Association for Computational
Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 7731–7748.
Turner, S. and Littlemore, J. (2023). Literal or metaphorical? Conventional or creative? Contested metaphoricity in
intense emotional experiences.Metaphor and the Social World, 13(1):37–58.
Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., others, and Zhou, D. (2022). Chain-of-thought
prompting elicits reasoning in large language models.Advances in Neural Information Processing Systems, 35:24824–
24837.
Yang, C., Chen, P., and Huang, Q. (2024). Can ChatGPT’s performance be improved on verb metaphor detection tasks?
Bootstrapping and combining tacit knowledge. InProceedings of the 62nd Annual Meeting of the Association for
Computational Linguistics (Volume 1), pages 1016–1027.
Yu, . D., Bondi, M., and Hyland, K. (2024a). Can GPT-4 learn to analyse moves in research article abstracts?Applied
Linguistics.
Yu, . D., Li, L., Su, H., and Fuoli, M. (2024b). Assessing the potential of LLM-assisted annotation for corpus-based
pragmatics and discourse analysis: The case of apology.International Journal of Corpus Linguistics, 29(4):534–561.
17