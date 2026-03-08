# Detecting RAG Advertisements Across Advertising Styles

**Authors**: Sebastian Heineking, Wilhelm Pertsch, Ines Zelch, Janek Bevendorff, Benno Stein, Matthias Hagen, Martin Potthast

**Published**: 2026-03-05 08:16:21

**PDF URL**: [https://arxiv.org/pdf/2603.04925v1](https://arxiv.org/pdf/2603.04925v1)

## Abstract
Large language models (LLMs) enable a new form of advertising for retrieval-augmented generation (RAG) systems in which organic responses are blended with contextually relevant ads. The prospect of such "generated native ads" has sparked interest in whether they can be detected automatically. Existing datasets, however, do not reflect the diversity of advertising styles discussed in the marketing literature. In this paper, we (1) develop a taxonomy of advertising styles for LLMs, combining the style dimensions of explicitness and type of appeal, (2) simulate that advertisers may attempt to evade detection by changing their advertising style, and (3) evaluate a variety of ad-detection approaches with respect to their robustness under these changes. Expanding previous work on ad detection, we train models that use entity recognition to exactly locate an ad in an LLM response and find them to be both very effective at detecting responses with ads and largely robust to changes in the advertising style. Since ad blocking will be performed on low-resource end-user devices, we include lightweight models like random forests and SVMs in our evaluation. These models, however, are brittle under such changes, highlighting the need for further efficiency-oriented research for a practical approach to blocking of generated ads.

## Full Text


<!-- PDF content starts -->

Detecting RAG Advertisements Across Advertising Styles
Sebastian Heineking
University of Kassel
Kassel, GermanyWilhelm Pertsch
Friedrich-Schiller-
Universität Jena
Jena, GermanyInes Zelch
Friedrich-Schiller-
Universität Jena
Jena, GermanyJanek Bevendorff
Bauhaus-Universität
Weimar
Weimar, Germany
Benno Stein
Bauhaus-Universität
Weimar
Weimar, GermanyMatthias Hagen
Friedrich-Schiller-
Universität Jena
Jena, GermanyMartin Potthast
University of Kassel,
hessian.AI, and ScaDS.AI
Kassel, Germany
Abstract
Large language models (LLMs) enable a new form of advertising for
retrieval-augmented generation (RAG) systems in which organic
responses are blended with contextually relevant ads. The prospect
of such “generated native ads” has sparked interest in whether they
can be detected automatically. Existing datasets, however, do not
reflect the diversity of advertising styles discussed in the marketing
literature. In this paper, we (1) develop a taxonomy of advertising
styles for LLMs, combining the style dimensions of explicitness and
type of appeal, (2) simulate that advertisers may attempt to evade
detection by changing their advertising style, and (3) evaluate a
variety of ad-detection approaches with respect to their robustness
under these changes. Expanding previous work on ad detection, we
train models that use entity recognition to exactly locate an ad in an
LLM response and find them to be both very effective at detecting
responses with ads and largely robust to changes in the advertising
style. Since ad blocking will be performed on low-resource end-user
devices, we include lightweight models like random forests and
SVMs in our evaluation. These models, however, are brittle under
such changes, highlighting the need for further efficiency-oriented
research for a practical approach to blocking of generated ads.1
CCS Concepts
•Information systems→Content match advertising.
Keywords
Online Advertising; Retrieval-augmented Generation; Large Lan-
guage Models
ACM Reference Format:
Sebastian Heineking, Wilhelm Pertsch, Ines Zelch, Janek Bevendorff, Benno
Stein, Matthias Hagen, and Martin Potthast. 2026. Detecting RAG Adver-
tisements Across Advertising Styles. InArXiv.January 2026, 11 pages.
1 Introduction
Ads will soon be added to responses of large language models.
Commercial chatbots such as ChatGPT started as free services, but
1Code and data: https://anonymous.4open.science/r/detecting-rag-advertising-styles/
This work is licensed under a Creative Commons Attribution 4.0 International License.
ArXiv, webis.de
©2026 Copyright held by the owner/author(s).
Rational
Covert
ExplicitnessType of AppealOvert
Emotional
Yes, certain destinations offer 
last-minute travel deals. [...] 
FUN Flights (fun.example) often 
features last-minute deals to these 
hot spots, offering 15% off when 
you upgrade [...]
Yes, certain destinations offer 
last-minute travel deals. [...] 
Feeling the thrill of a spontaneous 
getaway?  FUN Flights on 
fun.example brings 15% off when 
you upgrade [...]
Yes, certain destinations offer 
last-minute travel deals. [...] 
You might find options  through 
FUN Flights at fun.example, which 
often include secure payment [...]
Yes, certain destinations offer 
last-minute travel deals. [...]
FUN Flights offers  smooth 
bookings for your spontaneous 
escape  [...]
Are there specific destinations that usually have last minute travel deals?Figure 1: Examples of generated native ads in RAG responses
using four advertising styles (one per cell). Note the explicit-
ness of the ad snippets in the ”Overt” row, or the appeal to
emotions in the “Emotional” column (marked in bold).
vendors quickly sold subscriptions to Pro features to offset their vast
expenses. Yet only 5 % of OpenAI’s users are paying subscribers [ 41]
and others are likely not better off. All vendors are therefore work-
ing on integrating ads into their chatbots, as evidenced by press
releases [ 27,33,36], research [ 9,13], and industry news [ 1,12]. An
open question is whatformthese ads will take.
In this paper, we study a new form of so-called “generated na-
tive advertising” (Figure 1), in which LLMs blend ads directly into
otherwise organically generated responses [ 45]. Native advertising
predates the digital age and has to date been associated primarily
with journalistic media such as newspapers and magazines [ 2,43].
Crucially, native ads are deliberately designed by publishers and
advertisers to look and read like genuine news articles and to create
the impression of editorial content. Even though regulations in
many jurisdictions require every form of advertising to be disclosed
to consumers [ 29], such disclosures need not be prominent. And
since native ads are more effective when readers remain unaware of
them [ 34], some publishers render disclosures as inconspicuous as
possible, for example, through fine print, low-contrast labels, or un-
usual formatting. Native advertising has therefore been criticized as
a threat to the credibility of editorial content [ 34]. As RAG systems
and chatbots have become a popular alternative to conventional
search engines [ 38], and as users grow accustomed to and becomearXiv:2603.04925v1  [cs.IR]  5 Mar 2026

ArXiv, January 2026, webis.de Sebastian Heineking et al.
less scrupulous of their ostensibly organic responses [ 18], introduc-
ing native ads into them is a worrying prospect.
We study the robustness of detectors for generated native ads
in RAG responses. As research on generated ad detection is still in
its infancy (Section 2), we observe several shortcomings in related
work in terms of realism and grounding in marketing research. Our
contributions are (1) a taxonomy of advertising styles for LLMs
derived from marketing literature, which encompasses the explic-
itness of the promotion and the type of appeal as distinguishing
characteristics. This enables (2) a “simulation” of how advertisers
may attempt to evade native ad detection by changing its style,
engaging in a cat-and-mouse game with developers of ad detectors.
We compile a suite of test sets and use them to (3) evaluate how
robust various types of ad detectors are under domain shifts caused
by switching the ad-generating LLM and/or the advertising style.
2 Related Work
Generated advertising and its blocking are new fields of research.
Below, we summarize the related work on ad auction mechanisms
for LLMs, the blending of ads and organic responses, and their
detection and blocking.
2.1 Advertising in LLM Responses
Advertising in LLM responses has not been deployed so far. Current
research focuses primarily on how to design the auction mecha-
nism for interested advertisers and, related to that, the placement of
the ads in the responses. The suggested approaches can be divided
into ones in which advertisers bid for a specific position in the
response [ 8,13], and others in which the bids influence the distri-
bution from which tokens are sampled [ 9,39]. The former treat
an ad as a unit placed somewhere in the response. This ensures at
least some separation of organic text and advertisements. The latter
blur the boundary between organic content and advertising, so
that potentially the entire response is sampled from a distribution
biased in the interest of advertisers.
As an example of the first scenario, Hajiaghayi et al . [13] propose
an auction model for individual segments, e.g., paragraphs, in a
RAG response. Similar to the sponsored links in a classic SERP,
advertisements are selected based on query relevance, advertiser
bid, and click probability, and added to the context of the LLM that
generates the segment. In the framework proposed by Dubey et al .
[8], advertisers do not bid on a specific segment but for increased
prominence. The authors define prominence as an abstract concept
that has a monotonically positive relation to user attention. Higher
prominence can mean that an ad is represented with a larger num-
ber of words or that it is positioned more visibly in the response.
An auction module receives bids, quality scores for the ads, and
predicted click-through rates. Through its monotonic allocation
function (higher bids result in higher prominence), the auction
module outputs a “prominence allocation” and prices. The LLM is
then instructed to create a text with appropriately prominent ads.
Dütting et al . [9]present a possible implementation of the second
scenario: Advertisers can bid to influence the probability distribu-
tion used to sample tokens. Central to their approach are LLM
agents that are able to generate advertisements adapted to a spe-
cific user query for a specific advertiser. But instead of query-levelauctions in which agents generate full responses to queries, the
authors propose a token auction model. Within this model, multi-
ple agents jointly generate a response by bidding on their desired
probability distributions for the next token, which are aggregated
to generate the final response. The higher an agent’s bid, the more
strongly it affects the aggregated token distribution. Soumalias
et al. [39] suggest a similar approach. Their framework consists of
a reference LLM that generates responses to maximize user satis-
faction. Each advertiser is represented by a reward function that
takes in a tuple of user query and generated response, and outputs
a reward value for the given advertiser. The auctioneer optimizes
the final token distribution to produce a response that maximizes
the aggregated rewards across all advertisers. This optimization is
constrained by a hyperparameter defining the maximum allowed
deviation from the original distribution of the reference LLM. This
is to ensure the interest of the user is considered as well.
2.2 Detection of LLM Advertisements
As discussed in the previous section, LLMs can take predefined mes-
saging, like a brand slogan or claims about a product, and tailor it to
a specific user interaction [ 9,35]. This creates native advertisements
that are hard to distinguish from the rest of the “organic” response
text [ 2,45]. To the best of our knowledge, there currently exist no
publicly available datasets of real LLM-generated advertisements,
but this should not hinder research on the topic. Schmidt et al . [35]
created a dataset of synthetic examples to study the detection of
potential ads in the responses of conversational search engines. The
Webis Generated Native Ads 2024 dataset contains 11,303 responses
collected from Microsoft Copilot and YouChat. The authors used
GPT-4 to insert advertisements into 6,401 of these responses. For
the identification of the ads, they tested two fine-tuned sentence
transformer models, MiniLM and MPNET, and several LLMs for
zero-shot classification.
A new version of the dataset, Webis Generated Native Ads 2025
(discussed in Section 3), was constructed for a shared task, in which
participants were asked for approaches to generate, but also to
detect ads in LLM responses [ 14]. For the detection task, the partic-
ipating groups submitted various transformer models like MPNet
and DeBERTa, a random-forest classifier, and logistic regression
with TF-IDF features. Across all approaches, the transformer models
were the most effective.
A different yet related form of advertising for which organic and
publicly available datasets exist, are sponsored segments in videos
and podcasts [ 16,31]. Similar to how LLMs can be prompted to pro-
duce advertising, content creators receive outlines from advertisers,
adapt them to their style and audience, and integrate the advertise-
ments into their content. Related work on detecting these types of
ads was published by Reddy et al . [31] , who used transcripts and
descriptions of podcasts to detect this kind of “extraneous content. ”
The texts were split into sentences and were classified with BERT
and non-neural classifiers using TF-IDF unigram and bigram fea-
tures. Kok-Shun and Chan [16] prompted GPT-4o on transcripts
of YouTube videos to detect segments with sponsored content to
assign them the labels “Media” or “Product.” Bevendorff et al . [3]
also used transcripts to distinguish between real product reviews
and commercial spam content using an SVM and POS features.

Detecting RAG Advertisements Across Advertising Styles ArXiv, January 2026, webis.de
Table 1: Confusion matrix of advertising styles found in mar-
keting research:■and□indicate exact and partial overlap.
Marketing style
dichotomy Hard-sell Soft-sell Informational Transformational RationalEmotional OvertCovert1987Hard-sell■ □ □ □
Soft-sell■ □ □ □2017Informational□ ■ □
Transformational□ ■ □2018Rational□ □ ■
Emotional□ □ ■2019Overt□ ■
Covert□ ■
3 The Style of Advertising Language
Marketing research on advertising distinguishes several styles of
ads. However, research that pertains particularly to advertising
language is relatively scarce, since long-form texts are not the main
medium of marketing. As a basis for the emerging research on
advertising in LLMs, we derive an operationalizable taxonomy of
styles of advertising language. We first analyze the relevant related
work from marketing and devise a basic model of how advertisers
may go about creating such ads using large language models.
3.1 Dichotomies of Advertising Styles
In marketing, different schools of thought exist for categorizing
advertising styles, all of which are dichotomies, separating adver-
tising into two broad classes. Table 1 gives a conceptual overview.
One of them distinguishes betweenhard-sellandsoft-selladvertis-
ing [23,25]. Hard-sell advertising relies on direct promotion, explic-
itly highlighting and reasoning about positive product attributes.
Soft-sell advertising, in contrast, uses indirect communication, in-
voking images or a specific atmosphere with the goal of appealing
to the recipient’s emotions. This binary differentiation mixes a two
dimensions that are separate in other works with more granular
categories: Theexplicitness(or directness) of a promotion and the
type of appealit uses.
Ad appeals have been separated intorationalandemotional[ 17],
orinformationalandtransformational[ 5,24]. A rational appeal
highlights positive attributes of a product (e.g., quality, value, per-
formance) to address the audience’s self-interest, while an emo-
tional appeal targets their feelings [ 17]. Likewise, informational ads
highlight the usefulness of the products based on facts and reason-
ing, while transformational ads appeal to the consumer’s senses,
imagination, and emotions [ 5,24]. A potential third category,moral
appeal [17], addresses the audience’s beliefs about what is “right.”
We observe conceptual overlap between these dichotomies. How-
ever, the differentiations made are highly discipline-specific. We
must also take into account that different marketing sub-disciplines
have different desiderata. For the purposes of operationalization,
however, we blurr out the details and adopt the terminology ofthe latest dichotomy of rational versus emotional appeal. Com-
bined with the following one, we cover the most salient aspects of
marketing styles that are being distinguished.
Ad explicitness has been divided intoovertandcovertadvertis-
ing [ 6]. Overt advertising applies “traditional” methods like bill-
boards or banners on web pages. Covert advertising is implemented
more subtly, for example in the form of product placement in news
articles, editorial content (advertorials), influencer marketing [ 42],
or required ingredients in recipes [ 6,20]. The common denomi-
nator is that the promotional intent is concealed, for example, by
assuming the same appearance as organic, non-advertising content,
or by placing the product into a scene without explicity mentioning
its name. This makes covert often difficult to recognize [ 10,28,44].
Studies show that covert ads often (questionably) lead to more
positive customer reception [ 6,7]. Some authors also distinguish
explicit and implicit advertising, but this seems to be restricted to
the context of environmental sustainability [11].
3.2 Operationalizing Advertising Styles
The existing research on generated advertising and ad blocking
focuses on technical questions such as auction mechanisms, token
distributions, or classifier design. Consequently, advertisements are
reduced to a type of text that is either inserted into a response or
detected to be blocked. Advertisements, however, are not homoge-
neous. Instead, there is the aforementioned discourse in marketing
research about how to categorize advertising styles. This discourse
might be less relevant to conventional ad blocking that relies on
URL filter lists, request patterns, or JavaScript behavior to identify
advertisements. Ad blockers for LLMs, in contrast to that, will need
to identify advertisements based on textual clues alone. As different
advertising styles can produce advertisements with different vocab-
ularies and semantics, an understanding of these styles is important
for the blocking of generated ads. Therefore, based on our above
analysis, we propose a taxonomy of advertising styles for LLMs
that is illustrated with the examples in Figure 1.
The first dimension of our taxonomy is thelevel of explicitness,
distinguishing betweencovertandovertadvertising. Applied to
generated native advertising, this dimension is related to the con-
cept of prominence introduced by Dubey et al . [8]. The more overt
a generated advertisement is, the more attention it aims to attract
by, for example, using a very positive vocabulary or assigning a
large share of the response text to the advertisement. Consequently,
covert generated advertisements do not seek to attract attention,
but rather to influence the recipient in a more subltle way, for
instance, by mentioning a product as one possible option among
several alternatives.
The taxonomy’s second dimension is thetype of appeal, that
can be eitheremotionalorrational. Advertisements with a rational
appeal list (measurable) features about a product to convince their
audience. Examples include lower prices than competitors, longer
battery life of electronic devices, or important nutrients in food
and beverages. Advertisements with an emotional appeal are more
abstract and try to invoke an emotional reaction in the audience.
These emotions can be positive like joy, nostalgia, or pride, but
also negative like fear, for example in insurance advertising, or
sympathy in the case of charity advertising. The rational appeal

ArXiv, January 2026, webis.de Sebastian Heineking et al.
Table 2: Overview of the responses in the Webis Generated
Native Ads 2025 dataset
Ad Train Val. Test Total
22,416 3,999 4,316 30,731
10,311 1,781 1,904 13,996
Í32,727 5,780 6,220 44,727
can directly be applied to generated advertising: Instead of listing
features about a product on a billboard, they appear in the response.
Generated advertisements with an emotional appeal, however, are
restricted to short text as a medium to invoke emotions, while other
forms of advertising can do so with music or visuals.
4 Robustness of Ad Detectors
Detectors for LLM advertisements rely on patterns in the response
text to identify an ad [ 35]. These patterns, however, could change
depending on the advertising style or LLM used to generate an
advertisement. Especially if advertisers actively work against detec-
tion by applying their expert knowledge of advertising styles. This
raises the question how robust detectors are to changes in how the
advertisements are generated.
4.1 Simulating Ad Blocking Evasion
To answer this question, we trained a set of detectors, i.e., classifiers,
on a publicly available dataset of LLM responses with advertise-
ments, as a real-world ad blocking developer would. Then, we
created responses with new advertisements using our taxonomy of
advertising styles as if advertisers were manually trying to avoid
the styles used for training the ad blockers and evaluated the classi-
fiers on new test sets with these responses. To measure robustness,
we compared the effectiveness of the classifiers on the test split
of the publicly available dataset (thereference test set) against the
effectiveness on our newly generated test sets.
Training Data with a Naive Advertising Style.For our experi-
ments, we used the Webis Generated Native Ads 2025 (WGNA 25)
dataset.2The dataset contains 30,731 responses collected from
search engines that use retrieval-augmented generation (RAG).
The responses were generated byBrave Search,Microsoft Copilot,
Perplexity, andYouChatin response to 9,062 queries. The majority
of queries have a “commercial character”, as they are based on key-
words that a lot of advertisers compete for. Example queries are
“What are good last minute travel deals?” or “How do I choose the
right size for boys shorts?”.
For 13,996 of these responses, the dataset creators prompted
different LLMs to insert an advertisement. The input data for the
advertisements was collected by sending the 9,062 queries to start-
page.com and scraping the sponsored results. This resulted in
11,613 items (products or services) with a set of qualities, i.e. claims
about the item, and a corresponding advertiser. An example item
is given in Figure 2. Using five different prompts, the dataset cre-
ators instructed gpt-4o andgpt-4o-mini ,deepseek-r1 ,llama3
2https://doi.org/10.5281/zenodo.17830870.andllama-3.3 with 70B parameters, and qwen-2.5-32b to insert
advertisements into the 13,996 responses. In total, the dataset con-
tains 44,727 responses divided across train, validation, and test split
as illustrated in Table 2.
Test Data with Evasive Advertising Styles.The starting point for
our experiments is the test split of the WGNA 25 dataset. In the
following, we will refer to it as thereference test set. It contains
4,316 responses without and 1,904 responses with advertisements.
The advertisements were generated using the five LLMs listed under
Section 4.1 that we will refer to as the set ofoldLLMs. Analogously,
we will refer to the prompts used in the WGNA 25 as theoldprompts.
Instead specifying an advertising style as we do in this work, the
old prompts define the position of the ad in the response and how it
should be phrased, e.g., as a follow-up question or a call-to-action.
For our robustness tests, we created nine new versions of the ref-
erence test set by changing (1) the advertising prompt, (2) the LLM
used to generate the ad, and (3) both at the same time. Each test set
variation contains new versions of the 1,904 responses with adver-
tisements. To measure the effect of the prompt and ad-generating
LLM on the ad detection effectiveness more accurately, we hold
all other variables constant, i.e., the advertised item, its qualities,
the advertiser, the response in which the advertisement is inserted,
and the user query. The 4,316 responses without advertisements
are the same as in the reference test set. The nine test sets result
from combining four new prompts with both the old and a new set
of LLMs, and from using the old prompts with the new LLMs.
The set of fournewprompts is based on our taxonomy, each
instructing an LLM to apply one of the four advertising styles. To
develop the prompts, we took the perspective of advertisers trying
to circumvent existing ad-blockers for LLM responses and followed
the ad-generation procedure presented in previous works [ 35,45].
The prompts contain the query posed by the user of a RAG-system
as well as the response into which an advertisement should be
inserted. For each advertisement, the prompt contains the item,
i.e. product or service, that should be advertised, what qualities
that item has, as well as the name of the advertiser. Based on the
style, the LLM is instructed to use an emotional or rational ap-
peal, and to integrate the ad covertly or overtly into the original
response. We improved the prompts iteratively, reducing repeating
text patterns and increasing stylistic differences between different
categories of our taxonomy. In our experiments (Section 5.2), we
observe significant differences in how effective the classifiers are
at detecting advertisements generated by the different prompts.
These differences align with our expectations, e.g. covert ads being
harder to detect than overt ads, and serve as empirical validation
of the successful creation of advertisements in different advertising
styles. Figure 2 illustrates the process of inserting advertisements
into responses and Figure 3 shows an example prompt.3
In addition to the prompts, we varied the ad-generating LLMs
to measure their effect on classifier robustness. Based on the set
of old LLMs from the WGNA 25 dataset, we selected a set of
successors that we will refer to as thenewLLMs. It consists of
gpt-5-mini andgpt-5-nano , the 17B versions of llama-4-scout
3All prompts can be found in our repository.

Detecting RAG Advertisements Across Advertising Styles ArXiv, January 2026, webis.de
Yes, certain destinations are [...]
all-inclusive resorts with gourmet 
dining and beachfront activities, with 
no hidden fees. FUN Flights, 
available from 21 UK airports, 
ensures service with a smile.
Ad from the WGNA 25
Yes, certain destinations are known for offering last-minute travel deals. Here are a few 
popular options: [...] 
Aruba is a great choice for a spontaneous getaway. Last-minute deals often include 
all-inclusive resorts with gourmet dining and beachfront activities.
Are there specific destinations that usually have last minute travel deals?
Item:  FUN Flights             
Advertiser: fun.example
Qualities: 
15% off when you upgrade, 
no credit card fees, 
fly from 21 uk airports, 
service with a smile, 
secure payment
Rational
Covert
OvertExplicitnessType of Appeal
Emotional
Yes, certain destinations are [...]
FUN Flights (fun.example) often 
features last-minute  deals to these 
hot spots, offering 15% off when you 
upgrade, no credit card fees , flights 
from 21 UK airports , service with a 
smile and secure payment . [...]
Yes, certain destinations are [...]
Feeling the thrill of a spontaneous 
getaway?  FUN Flights on 
fun.example brings 15% off when 
you upgrade, no credit card fees, fly 
from 21 UK airports, service with a 
smile and secure payment. [...]
Yes, certain destinations are [...]
You might also find options
through FUN Flights at fun.example, 
which often include secure 
payment  and no credit card fees .
[...]
Yes, certain destinations are [...]
FUN Flights (fun.example) offers 
smooth bookings for your 
spontaneous escape  with 15% off 
when you upgrade and no credit 
card fees.
Figure 2: Example responses for different advertising prompts. The chat window shows a user query for last minute travel and
the response generated by a search engine. This response is adapted by inserting an ad for the item “FUN Flights”. In addition
to the response taken from the WGNA 25 test set, the figure shows the variations generated for different advertising styles.
Take the following response, generated for the search query ' <query> '.
1. Insert a short, subtle promotion of ' <item> ' by '<advertiser> ', giving rational reasons. Mention ' <advertiser> ' if it differs from ' <item> '. 
'<item> ' has to be mentioned smoothly by name, but try to hide the fact that you are promoting something as smoothly as possible.
2. Only if suiting the covert and rational context, promote one or two of the following qualities: ' <qualities> '.
3. Return only the unchanged original response with the inserted addition.
4. Do not start the addition with 'consider', 'for those', 'however', 'many' or 'such as', and do not use the formulation 'like ' <item> ''.
Response: ' <response> '
Response with insertion:
Figure 3: Prompt to create covert advertisements with rational appeals. The placeholders are filled with the information
depicted in Figure 2.
Table 3: Contingency table for MiniLM on the test set gener-
ated by the covert-emotional prompt with new LLMs
Test set TP FN
New (N) 1,355 549
Reference (R) 1,785 119OR=TPN/FN N
TPR/FN R=1355/549
1785/119
OR=0.165,CI 0.95=[0.13,0.20]
andllama-4-maverick , and gpt-oss-120b . We accessed the GPT-
4 and GPT-5 models via OpenAI and all other LLMs via Groq.4
Measuring Robustness.A robust classifier generalizes to unseen
types of advertisements, generated by different LLMs and with dif-
ferent advertising styles. In our experiments, the responses without
advertisements are constant between the test sets. Hence, we mea-
sure robustness based on the number of ads that a classifier detects.
For a given classifier and one of the nine new test sets, we (1) count
the number of true positives and false negatives on the reference test
set and the new test set, (2) add them to a contingency matrix, and
(3) and calculate an odds ratio for ads being detected in responses
from the reference test set versus the new test set. Table 3 shows
4Since the creation of the dataset, Groq has discontinued some of the old LLMs.
Following their recommendations, we replaced deepseek-r1 andllama-3-70b with
llama-3.3-70b, andqwen-2.5-32bwithqwen3-32b.the contigency table and odds ratio calculation for one classifier
and new test set. An odds ratio below 1 indicates that the classifier
detected fewer ads on the new than on the reference test set. We
consider a classifier robust if the difference in detected ads is not
significant. At 𝛼= 0.05, the difference is significant if the 95 % con-
fidence interval of the odds ratio does not include the value 1. We
control for a false discovery rate (FDR) of 5 % using the Benjamini-
Hochberg procedure on𝑚=9Test sets∗7Classifiers=63Tests.
4.2 Classifiers
We applied three groups of classifiers to the task of advertisement
detection. The first group consists of sentence transformers as ap-
plied by Schmidt et al . [35] . In the second group, we test a new
transformer-based approach to ad detection: Classifying tokens to
recognize the entities of an advertisement. We expect this higher
granularity to be particularly important for advertising implemen-
tations like the ones proposed by Dütting et al . [9] and Soumalias
et al. [39] . Classifiers looking for self-contained segments of text
might be less effective if advertisements are spread throughout the
response as the result of biased token sampling.
The third and final group consists of conventional, lightweight
classifiers in the form of a random forest and a support-vector
machine (SVM). We added this third group for two main reasons:
First, we want to analyze how effective ads can be detected without

ArXiv, January 2026, webis.de Sebastian Heineking et al.
the context used by transformers. Second, ad blockers would ideally
run on consumer devices, making efficient inference, both in terms
of time and resources, an important feature of ad detectors. As a
naive baseline, we also test a dictionary-based approach that assigns
probabilities based on manually selected terms from the General
Inquirer [40].
Sentence Classifiers.Schmidt et al . [35] used sentence transform-
ers to classify pairs of sentences as containing an advertisement or
not. Similar to next sentence prediction, the models were fine-tuned
to detect if one sentence followed the other in a response without
advertisements. We reproduced their approach and fine-tuned the
same models, all-MiniLM-L6-v2 andall-mpnet-base-v2 , with
the Adam optimizer [ 15] and binary cross-entropy loss. For MiniLM,
we used a batch size of 48 and a learning rate of 1 𝑒-5. For MP-
Net, we set the values to 16 and 5 𝑒-6. Additionally, we fine-tuned
ModernBERT-embed-base5on the same task, setting the batch size
to 16 and the learning rate to 2 𝑒-6. For all threesentence classifiers,
trained for up to 50 epochs and selected the final weights based on
validation F 1-score.
Token Classifiers.Additionally, we fine-tuned twotoken classi-
fierson the BIO-tags of the WGNA 25. The BIO format is used in
named-entity recognition (NER) and other areas of computational
linguistics to assign tags to tokens [ 30]. In NER, entities can span
multiple tokens. To account for that, the beginning of the entity is
marked with a B-tag and all tokens “in” the sequence belonging to
the entity are marked with I-tags. Other tokens that lie “outside”
of named entities receive O-tags. Applied to advertisements, the
WGNA 25 distinguishes between the item (“B-/I-ITEM”), the adver-
tiser (“B-/I-ADVERTISER”), and the rest of the ad (“B-/I-AD”). All
other tokens are tagged as “O”. We fine-tuned ModernBERT-base6
andBERT-base-cased7on the BIO-tags using the AdamW opti-
mizer [ 19], a learning rate of 2 𝑒-5 with linear scheduling, and a
batch size of 16. Again, the final weights were chosen based on val-
idation F 1-score, this time over 20 epochs. To distinguish sentence
classifiers from token classifiers, we give the former a subscript S
(e.g. MiniLM S) and the latter a subscript T (MBERT T).
Random Forest.As a lightweight alternative to transformer-based
classifiers, we trained a sentence-level Random Forest classifier [ 4],
labeling a response as containing an advertisement if at least one
sentence is classified as such. Using scikit-learn [ 26], we represented
each sentence as a binary bag-of-words vector and used a feature
selection based on mutual information to retain only the most dis-
criminative terms. To account for the imbalanced distribution of
advertisement and non-advertisement sentences, we used balanced
class weights. We performed a grid search over the minimum docu-
ment frequency for vocabulary inclusion, the number of selected
features, the number of trees, the number of features considered
at each split, and the minimum samples per leaf. We optimized for
F1-score on the validation split of the WGNA 25 dataset, tuning the
Random Forest’s probability threshold to maximize F 1-score.
5huggingface.co/nomic-ai/modernbert-embed-base
6https://huggingface.co/answerdotai/ModernBERT-base
7https://huggingface.co/google-bert/bert-base-casedSupport Vector Machine.For comparison, we trained a sentence-
level linear SVM, again labeling a response as containing an adver-
tisement if at least one sentence is classified as such. Each token
was represented by a 300-dimensional Word2Vec embedding [ 21],
pretrained on Google News [ 22], and loaded via Gensim [ 32]. Each
sentence was represented by the mean of its token embeddings. We
performed a grid search over the regularization parameter 𝐶, the
loss function, and input lowercasing. The SVM was implemented us-
ing scikit-learn [ 26], calibrated via Platt scaling to obtain probability
estimates, and the classification threshold was tuned to maximize
F1-score on the validation split.
Dictionary.Since traditional ad blockers often rely on rule-based
systems [ 37], and since we observed reoccurring lexical patterns
in generated advertisements, we evaluated a dictionary-based ap-
proach. First, we considered words tagged as positive and overstated
in the General Inquirer [ 40], but the resulting classifier performed
worse than random predictions. Second, we selected the 200 words
with the highest mutual information toward the advertisement class
from the training set. While this was more effective, it still fell short
of all other classifiers and is excluded from the following analyses.
5 Evaluation
The following section describes our evaluation of the classifiers’
effectiveness on the test data and their robustness to changes in
the LLMs and prompts used for generating the advertisements. We
find significant differences between advertising styles, and that
effectiveness is largely correlated with the parameter count of the
detector. We further test the token-based classifiers for their ability
to detect named entities relating to the generated advertisements.
5.1 Classifier Effectiveness
As a first step, we evaluated each classifier on the reference test
set generated with the “old” LLMs and prompts. The results for all
classifiers are given in the first row of Tables 4 and 5. Aside from the
SVM, all classifiers achieve F 1-scores of 0.9 and higher. We observe
that (1) the token classifiers are more effective than their coun-
terparts that classify pairs of sentences, and (2) the classification
effectiveness increases with the number of parameters.
Tables 4 and 5 also summarize the effectiveness scores on all
nine new test sets, created by varying the LLM, the prompt, or both.
As the negative class (responses without ads) contains the same
examples in all test sets, any deviation in a classifier’s precision
stems entirely from a decrease in true positives. The recall scores
on both the reference and the new test sets show the same trends:
Models with more parameters are more effective than smaller mod-
els, and token classifiers are more effective than sentence classifiers.
Surprisingly, we observe that all transformer-based classifiers are
actuallybetterat identifying ads generated by the overt-emotional
prompt than those from the reference set, which they were trained
on. The only exception is the sentence classifier MiniLM (MiniLM S),
which is slightly less effective on overt-emotional advertisements
generated by new LLMs.
5.2 Classifier Robustness
To test if a classifier is robust to changes in the LLM or prompt, we
calculated the odds ratio (odds being true positives against false

Detecting RAG Advertisements Across Advertising Styles ArXiv, January 2026, webis.de
Table 4: Effectiveness of the transformer-based classifiers. The colors indicate if the classifier detects more, fewer or the same
number of ads as on the reference test set (old LLMs & prompts). Asterisks (*) highlight significant differences. Classifiers with
a subscript S classify pairs of sentences (MiniLM S), those with a T classify tokens (MBERT T).
MiniLM S MPNET S MBERT S BERT T MBERT T
LLMs PromptPrec. Rec. F 1 Prec. Rec. F 1 Prec. Rec. F 1 Prec. Rec. F 1 Prec. Rec. F 1
OldOld Prompts 0.976 0.938 0.957 0.992 0.926 0.958 0.964 0.977 0.971 0.987 0.983 0.9850.995 0.9980.997
Overt-Emotional 0.977 0.959* 0.968 0.992 0.962* 0.977 0.965 0.988* 0.977 0.987 0.992* 0.9890.995 1.0000.997
Overt-Rational 0.975 0.868* 0.918 0.991 0.847* 0.914 0.963 0.946* 0.955 0.987 0.981 0.9840.995 0.9980.997
Covert-Emotional 0.973 0.799* 0.877 0.991 0.780* 0.873 0.962 0.925* 0.944 0.986 0.951* 0.9680.995 0.9960.995
Covert-Rational 0.969 0.709* 0.819 0.989 0.651* 0.785 0.960 0.855* 0.904 0.986 0.943* 0.9640.995 0.9950.995
NewOld Prompts 0.975 0.871* 0.920 0.992 0.897* 0.942 0.964 0.962* 0.963 0.987 0.979 0.9830.995 0.9980.996
Overt-Emotional 0.976 0.932 0.954 0.992 0.949* 0.970 0.965 0.992* 0.978 0.987 0.991* 0.9890.995 0.9990.997
Overt-Rational 0.973 0.805* 0.881 0.991 0.809* 0.891 0.963 0.942* 0.952 0.987 0.973* 0.9800.995 0.9970.996
Covert-Emotional 0.969 0.712* 0.821 0.990 0.743* 0.849 0.961 0.902* 0.931 0.986 0.926* 0.9550.995 0.988*0.992
Covert-Rational 0.965 0.627* 0.760 0.988 0.592* 0.741 0.957 0.808* 0.876 0.986 0.905* 0.9440.995 0.989*0.992
0.0 0.5 1.0 1.5 2.0 2.5 3.0SVMRForestMiniLMSMPNETSMBERTSBERTTMBERTTOvert-Emotional
Old LLMs
New LLMs
0.0 0.5 1.0 1.5 2.0 2.5 3.0Overt-Rational
Old LLMs
New LLMs
0.0 0.5 1.0 1.5 2.0 2.5 3.0Covert-Emotional
Old LLMs
New LLMs
0.0 0.5 1.0 1.5 2.0 2.5 3.0Covert-Rational
Old LLMs
New LLMs
Figure 4: Ad detection odds ratios (95 % CI). For each classifier and new test set, we compared the odds of detecting an ad in the
new test set to the odds in the reference test set (see also Table 3). The black vertical ticks show the odds ratio and the colored
horizontal lines the corresponding confidence interval. The x-Axis is cut at 3.0 for improved clarity.
Table 5: Effectiveness of the context-free classifiers. Signifi-
cant differences are highlighted with an asterisk (*).
SVM RForest
LLMs PromptPrec. Rec. F 1 Prec. Rec. F 1
OldOld Prompts 0.470 0.812 0.595 0.883 0.914 0.898
Overt-Emo. 0.453 0.758* 0.567 0.870 0.811* 0.840
Overt-Rat. 0.434 0.703* 0.537 0.861 0.751* 0.802
Covert-Emo. 0.374 0.547* 0.444 0.798 0.478* 0.598
Covert-Rat. 0.370 0.539* 0.439 0.792 0.461* 0.582
NewOld Prompts 0.434 0.703* 0.536 0.856 0.721* 0.782
Overt-Emo. 0.441 0.724* 0.548 0.854 0.711* 0.776
Overt-Rat. 0.406 0.627* 0.493 0.848 0.674* 0.751
Covert-Emo. 0.376 0.552* 0.447 0.780 0.431* 0.555
Covert-Rat. 0.357 0.508* 0.419 0.780 0.430* 0.554
negatives) in a 95 % confidence interval between ads detected in
responses from the reference test set versus each of the new test
sets (illustrated in Table 3). The difference in the number of detected
ads is significant at 𝛼= 0.05when the confidence interval does notcontain 1. The confidence intervals are depicted in Figure 4. All
significant differences (controlled for a false discovery rate of 5 %)
are marked with an asterisk in Tables 4 and 5. The results show that
most classifiers detect significantly fewer ads in most of the new
test sets, and are thus not robust to the respective change in prompt
and / or LLM. The one exception is the token-classifier based on
ModernBERT (MBERT T), that is robust to all changes except the
covert advertisements generated by new LLMs. Both the random
forest and the SVM are not robust to any of the changes. Similar to
the transformers, however, they achieve the highest effectiveness
scores on the overt-emotional advertisements generated by the set
of old LLMs. The SVM is overall less effective than the random
forest, but generalizes better to the new test sets.
Differences Between Advertising Styles.The odds ratios in Fig-
ure 4 show that the covert advertising style is more difficult to
detect than the overt style. Most classifiers are significantly less
effective at detecting these advertisements. While MBERT Tis good
at detecting the covert advertising style generated by LLMs also
used in the reference test set, the model is significantly less effective
on the same advertising style generated by new LLMs. In contrast,
the overt-emotional advertising style yields very different results:

ArXiv, January 2026, webis.de Sebastian Heineking et al.
MBERT Tis able to detectalladvertisements generated in this style
by the old LLMs and almost all generated by new LLMs. The BERT
token classifier (BERT T), the sentence classifiers based on Modern-
BERT (MBERT S), and MPNET (MPNET S) are less effective on the
reference test set, but detect significantly more ads than in the
reference test set. For the overt-rational prompt, the results are less
consistent. All sentence classifiers detect significantly fewer ads in
this advertising style, independent of the LLM. The token classifiers
all detect fewer of the ads generated by new LLMs; for BERT Tthe
reduction is significant, for MBERT Tit is not.
To quantify the differences between advertising styles, we per-
formed the same odds ratio calculation as for the robustness anal-
ysis. We hold everything constant except for the dimension of
interest and compare the number of detected ads between related
pairs. For the comparison covert vs. overt, this means comparing
the results for the same classifier, the same set of LLMs, and the
same type of appeal (emotional or rational). This results in 7 Classi-
fiers×2 LLM Sets×2 Advertising Styles = 28 Comparisons, again
controlled for an FDR of 5 %. In 26 out of those 28 comparisons, the
advertisements with an overt style are significantly easier to detect
than their covert counterparts. In the other two comparisons, the
overt advertisements are also easier to detect but not significantly
so. Comparing the emotional against the rational appeal, we find
the emotional advertisements to be significantly easier to detect in
20 of 28 cases.
Differences Between LLMs.Similar to the advertising styles, we
tested if the classifiers are robust to using a different set of LLMs to
generate the advertisements. Looking at Tables 4 and 5, we see that
all sentence transformers, the random forest, and the SVM detect
significantly fewer ads when combining the old set of prompts
with the new set of LLMs. The two token classifiers, however, are
robust against this change. Using the same approach as for the
advertising styles above, we find that in 17 of 28 comparisons, the
advertisements generated with the set of old LLMs are significantly
easier to detect than their counterparts generated with new LLMs.
Overlap in False Negatives.Additionally, we compared the ad-
vertisements that the classifiers did not detect. For each of the
nine new test sets and the reference test set, we formed pairs of
classifiers and calculated the Jaccard index of their false negatives.
Figure 5 depicts the average Jaccard index for each pair across all
test sets. The two sentence classifiers MPNET Sand MiniLM Shave
the highest overlap in false negatives with an average Jaccard index
of 0.452. We observe that most classifiers have the highest overlap
with other classifiers in the same category: The random forest has
the highest overlap with the SVM (and vice versa), the sentence
classifier MBERT Shas the highest overlap with MPNET S, and the
token classifier MBERT Thas the highest overlap with BERT T. The
only exception is the token classifier BERT T, which has its highest
overlap with the sentence classifier MBERT S.
5.3 Effectiveness of Entity Recognition
In addition to the binary classification task of labeling a response as
containing an ad or not, we also evaluated the token classifiers for
their ability to detect the following three entities:item,advertiser,
andother advertising text(such as item qualities). These entities
MBERTTBERTTMBERTSMPNETSMiniLMSRForest SVM 
MBERTT
BERTT
MBERTS
MPNETS
MiniLMS
RForest 
SVM 0.072 0.039 0.016 0.016 0.008 0.006
0.072 0.227 0.147 0.146 0.071 0.042
0.039 0.227 0.270 0.262 0.127 0.080
0.016 0.147 0.270 0.452 0.276 0.175
0.016 0.146 0.262 0.452 0.291 0.183
0.008 0.071 0.127 0.276 0.291 0.344
0.006 0.042 0.080 0.175 0.183 0.344
0.00.10.20.30.40.5Figure 5: Average overlap in false negatives. The heatmap
shows the mean Jaccard index over all test sets. A score of 1
indicates that two classifiers always miss the same ads.
Table 6: Effectiveness on entity recognition. Scores are cal-
culated based on all labeled and detected entities. We do not
report significance, as the detection of entities in the same
response cannot be assumed to be independent events.
BERT T MBERT T
LLMs PromptPrec. Rec. F 1 Prec. Rec. F 1
OldOld Prompts 0.698 0.757 0.7260.778 0.8180.798
Overt-Emo. 0.756 0.772 0.7640.813 0.8220.817
Overt-Rat. 0.704 0.725 0.7140.784 0.8010.793
Covert-Emot. 0.634 0.667 0.6500.696 0.7260.711
Covert-Rat. 0.609 0.632 0.6200.674 0.7140.694
NewOld Prompts 0.659 0.696 0.6770.739 0.7600.750
Overt-Emot. 0.747 0.743 0.7450.786 0.7890.787
Overt-Rat. 0.699 0.697 0.6980.758 0.7730.766
Covert-Emot. 0.657 0.641 0.6490.707 0.7070.707
Covert-Rat. 0.605 0.585 0.5950.688 0.6940.691
are marked with BIO-tags at token-level. The tags signal if a token
marks thebeginningof an entity (e.g., “B-ITEM”), occursinsidea
sequence of tokens belonging to the entity (e.g., “I-ADVERTISER”),
oroutsideof it (“O”). We calculated the effectiveness of the two
token classifiers using the seqeval metric of the evaluate Python
package.8The values for precision and recall in Table 6 are based
on (1) all ground-truth entities in the respective test set and (2) all
entity predictions by the respective classifier. To detect an entity,
the classifier needs to assign the correct BIO-tags toalltokens
belonging to that entity. If fewer or more tokens are tagged, the
entity is counted as a false negative. It is important to mention that,
in contrast to the response classification, the precision scores vary
between test sets. This is because, unlike before, the number of
false positives can also increase for the newly generated responses.
8https://github.com/huggingface/evaluate

Detecting RAG Advertisements Across Advertising Styles ArXiv, January 2026, webis.de
0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4BERTT (C-R)MBERTT (C-R)BERTT (C-E)MBERTT (C-E)BERTT (O-R)MBERTT (O-R)BERTT (O-E)MBERTT (O-E)
Old LLMs
New LLMs
Figure 6: Entity recognition odds ratios (95 % CI). For each
classifier and new test set, we compared the odds of detecting
allentities in a response against the odds in the reference
test set (Differs from the entity-level effectiveness in Table 6).
To perform similar robustness tests as for the response classifi-
cation, we counted the number of responses for which a classifier
detectedallentities. We performed this aggregation, because the
odds ratio test requires each trial to be independent of previous
trials. However, this requirement may not hold if we treat the all en-
tity detections as individual trials, since the entities themselves are
not necessarily independent. If one entity is detected in a response,
the probability of detecting other entities in the same response
might increase. This macro aggregation is therefore stricter than
the micro effectiveness scores in Table 6, but necessary to perform
the test. We did not add asterisks to Table 6, as the basis for the re-
call scores is hence different from the other odds ratio calculations.
We treated a response as a true positive if the classifier detectedall
its entities, and as a false negative otherwise. With these counts,
the odds ratios were calculated in the same way by comparing the
odds from the new test sets against those from the reference test set.
The resulting odds ratios and their corresponding 95 % confidence
intervals are depicted in Figure 6.
We observe the same tendencies as for the ad detection: First,
the token classifiers correctly detect more entities when the adver-
tisements were generated in an overt instead of a covert advertising
style. Second, the entities in advertisements with an emotional
appeal are easier to detect than in advertisements with a rational
appeal. The detection odds ratios, however, are not as clear. The
only prompt with consistent results is the covert-rational prompt,
for which both classifiers detect significantly fewer entities, re-
gardless of the generating LLM. ModernBERT is not robust to the
covert-emotional prompt, while BERT is robust when the ads are
generated by the set of old LLMs.
The overt-rational prompts lead to no significant changes in the
entity detection. BERT is significantlymoreeffective on the overt-
emotional advertisements generated by the old LLMs and robust to
those generated by the new LLMs. ModernBERT, however, is not
robust to the new LLMs generating overt-emotional advertisements.
Overall, ModernBERT is less robust at detecting advertising entities
than on the task of detecting responses with advertisements.6 Discussion and Limitations
Our results indicate that many classifiers are not robust to changes
in the advertising style or the ad-generating LLM. The only excep-
tion is the token classifier based on ModernBERT that retains a lot
of its effectiveness across all nine new test sets. The lowest recall
value, 0.988, occurs for the advertisements generated by the set of
new LLMs with a covert-emotional advertising style. In absolute
terms, ModernBERT misses 22 responses with advertisements in
this test set, compared to 3 responses in the reference test set.
In line with their definition, covert advertisements are harder
to detect than their overt counterparts. We also observe that emo-
tional appeals seem to be easier to detect than rational ones. One
possible explanation is that the ad-generating LLM needs to use
additional vocabulary to create an emotional narrative, making it
easier for classifiers to detect. We also find that the set of new LLMs
are better at generating advertisements that evade detection by
our classifiers. This could be either due to a better adherence to
the covert advertising style or a general shift in their generated
advertisements that removes patterns that the classifiers look for.
In either case, it underlines the need for robust ad detectors.
We find classifiers operating at token-level to be both more ef-
fective and more robust than classifiers operating on larger units of
text. A possible explanation is that the context of the full response
allows these classifiers to evaluate each token in interaction with
all other tokens, thereby detecting patterns like a positive adjec-
tive referring to a product name. At the same time, these token
classifiers are not as effective or robust on the task of detecting
advertising entities. This task, however, plays an important role
in the development of ad blockers for LLMs: Only with an exact
position of the advertising text can ad blockers be precise and not re-
move too much of a generated response. Although the context-free
classifiers that we tested are not as reliable as the transformer-
based models, they might still find legitimate use in the creation of
production-ready ad blockers. The random forest classifier is effec-
tive at detecting similar advertisements to the ones it was trained
on and end-user devices like smartphones cannot be guaranteed to
fulfill the hardware requirements of the transformer models.
Overall, our results are limited to the data, LLMs, and classifiers
we used in our experiments. With regard to the data, this includes
the specific advertisements of the WGNA 25 dataset with their
structure of item, qualities, and advertiser. Furthermore, the adver-
tisements were insertedaftera response was already generated. In
future work, we want to implement advertising mechanisms that
generate the response from an advertiser-biased distribution and
study the robustness of classifiers to this setup.
7 Conclusion
Advertisements will become a part of RAG responses. In preparation
for this scenario, existing research has explored the detection of
ads in the response texts. We contribute to this emerging field of
research by (1) proposing a taxonomy of advertising styles for LLMs
(2) simulating that advertisers may evade trained ad detectors by
changing their advertising style, and (3) studying the robustness of
various ad detectors under these changes.

ArXiv, January 2026, webis.de Sebastian Heineking et al.
We find classifiers operating on individual tokens to be both
more effective and more robust than classifiers proposed by pre-
vious research. Especially ModernBERT is consistently effective
at detecting responses with advertisements. Conventional classi-
fiers like a random forest and SVM are not robust to changes in
the advertising style, suggesting that lexical patterns alone are in-
sufficient for a reliable detection, and that context information is
important. Finally, our experiments reveal room for improvement
in the task of precisely locating an ad in a response. Future work in
this area will be a valuable step towards ad blockers for LLMs and
thus contribute to higher informational quality in RAG responses.
References
[1]Anu Adegbola. 2025. Google Expands ads in AI Overviews, AI Mode to desk-
top. https://searchengineland.com/google-ads-ai-overviews-ai-mode-desktop-
455733. (May 2025). Search Engine Land. Accessed: 2026-01-19.
[2]Michelle A. Amazeen and Bartosz W. Wojdynski. 2020. The Effects of Disclosure
Format on Native Advertising Recognition and Audience Perceptions of Legacy
and Online News Publishers.Journalism21, 12 (2020), 1965–1984. https://doi.
org/10.1177/1464884918754829
[3]Janek Bevendorff, Matti Wiegmann, Martin Potthast, and Benno Stein. 2024.
Product Spam on YouTube: A Case Study. In9th ACM SIGIR Conference on
Human Information Interaction and Retrieval (CHIIR 2024), Min Zhang, Joemon
Jose, and Laurianne Sitbon (Eds.). ACM, Sheffield, United Kingdon, 358–363.
https://doi.org/10.1145/3627508.3638303
[4]Leo Breiman. 2001. Random Forests.Machine Learning45, 1 (Oct. 2001), 5–32.
https://doi.org/10.1023/A:1010933404324
[5]Fabienne T. Cadet, Priscilla G. Aaltonen, and Vahwere Kavota. 2017. The Ad-
vertisement Value of Transformational & Informational Appeal on Company
Facebook Pages.Marketing Management Journal27, 2 (jan 1 2017), 116–130.
https://doi.org/10.63963/001c.151117
[6]Fanny Fong Yee Chan. 2019. The Perceived Effectiveness of Overt Versus Covert
Promotions.Journal of Product & Brand Management29, 3 (Aug. 2019), 321–334.
https://doi.org/10.1108/JPBM-06-2018-1912
[7]Davit Davtyan and Isabella Cunningham. 2017. An Investigation of Brand
Placement Effects on Brand Attitudes and Purchase Intentions: Brand Place-
ments versus TV Commercials.Journal of Business Research70 (2017), 160–167.
https://doi.org/10.1016/j.jbusres.2016.08.023
[8]Avinava Dubey, Zhe Feng, Rahul Kidambi, Aranyak Mehta, and Di Wang. 2024.
Auctions with LLM Summaries. InProceedings of the 30th ACM SIGKDD Confer-
ence on Knowledge Discovery and Data Mining. ACM, Barcelona Spain, 713–722.
https://doi.org/10.1145/3637528.3672022
[9]Paul Dütting, Vahab Mirrokni, Renato Paes Leme, Haifeng Xu, and Song Zuo.
2024. Mechanism Design for Large Language Models. InProceedings of the ACM
Web Conference 2024 (WWW ’24). Association for Computing Machinery, New
York, NY, USA, 144–155. https://doi.org/10.1145/3589334.3645511
[10] Fabian Göbel, Anton Meyer, Balasubramani Ramaseshan, and Silke Bartsch.
2017. Consumer Responses to Covert Advertising in Social Media.Marketing
Intelligence & Planning35, 5 (2017), 578–593. https://doi.org/10.1108/MIP-11-
2016-0212
[11] Siyu Gong and Li Wang. 2023. Are Explicit or Implicit Appeals More Credible?
The Congruence Effects of Green Advertising Appeals and Product Category on
Consumers’ Evaluation.Current Psychology42, 33 (2023), 29035–29047. https:
//doi.org/10.1007/s12144-022-03981-4
[12] Danny Goodwin. 2025. ChatGPT ads are coming - and they won’t look like
Google Ads. https://searchengineland.com/chatgpt-ads-coming-some-point-
464388. (July 2025). Search Engine Land. Acessed: 2026-01-19.
[13] Mohammad Taghi Hajiaghayi, Sébastien Lahaie, Keivan Rezaei, and Suho Shin.
2024. Ad auctions for LLMs via retrieval augmented generation. InProceedings
of the 38th International Conference on Neural Information Processing Systems
(NIPS ’24). Curran Associates Inc., Red Hook, NY, USA, Article 585, 36 pages.
https://doi.org/10.48550/arXiv.2406.09459
[14] Johannes Kiesel, Çağrı Çöltekin, Marcel Gohsen, Sebastian Heineking, Maximilian
Heinrich, Maik Fröbe, Tim Hagen, Mohammad Aliannejadi, Sharat Anand, Tomaž
Erjavec, Matthias Hagen, Matyáš Kopp, Nikola Ljubešić, Katja Meden, Nailia
Mirzakhmedova, Vaidas Morkevičius, Harrisen Scells, Moritz Wolter, Ines Zelch,
Martin Potthast, and Benno Stein. 2025. Overview of Touché 2025: Argumentation
Systems. InExperimental IR Meets Multilinguality, Multimodality, and Interaction.
16th International Conference of the CLEF Association (CLEF 2025) (Lecture Notes
in Computer Science), Jorge Carrillo de Albornoz, Julio Gonzalo, Laura Plaza,
Alba García Seco de Herrera, Josiane Mothe, Florina Piroi, Paolo Rosso, Damiano
Spina, Guglielmo Faggioli, and Nicola Ferro (Eds.). Springer, Berlin Heidelberg
New York, 4536–4562.[15] Diederik P. Kingma and Jimmy Ba. 2015. Adam: A Method for Stochastic Opti-
mization. (2015). arXiv:1412.6980 http://arxiv.org/abs/1412.6980
[16] Brice Valentin Kok-Shun and Johnny Chan. 2025. Leveraging ChatGPT for
Sponsored Ad Detection and Keyword Extraction in YouTube Videos. (2025).
https://doi.org/10.48550/arXiv.2502.15102 arXiv:2502.15102
[17] Philip Kotler. 2018.Principles of Marketing(17th ed.). Pearson, Harlow, England.
[18] Florian Leiser, Sven Eckhardt, Valentin Leuthe, Merlin Knaeble, Alexander Mäd-
che, Gerhard Schwabe, and Ali Sunyaev. 2024. HILL: A Hallucination Iden-
tifier for Large Language Models. InProceedings of the CHI Conference on
Human Factors in Computing Systems. ACM, Honolulu HI USA, 1–13. https:
//doi.org/10.1145/3613904.3642428
[19] Ilya Loshchilov and Frank Hutter. 2017. Decoupled Weight Decay Regularization.
(2017). arXiv:1711.05101 http://arxiv.org/abs/1711.05101
[20] Roseanne Luth. 2017. 3 Ways to Use Recipes in Your Marketing. https://www.
brandingmag.com/2017/08/19/3-ways-to-use-recipes-in-your-marketing/. (Aug.
2017). Brandingmag. Accessed: 2026-01-19.
[21] Tomás Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013. Efficient
Estimation of Word Representations in Vector Space. (2013). http://arxiv.org/
abs/1301.3781
[22] Tomás Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013.
Distributed Representations of Words and Phrases and Their Compositionality. In
Proceedings of the 27th International Conference on Neural Information Processing
Systems - Volume 2 (NIPS’13). Curran Associates Inc., Red Hook, NY, USA, 3111–
3119. https://dl.acm.org/doi/10.5555/2999792.2999959
[23] Barbara Mueller. 1987. Reflections of Culture: An Analysis of Japanese and
American Advertising Appeals.Journal of Advertising research27, 3 (1987), 51–59.
https://eric.ed.gov/?id=ED271776
[24] Wael Nuweihed and Olivier Trendel. 2024. The role of informational versus
transformational ad appeals in building consumer-based brand equity for low
involvement products.Journal of Marketing Theory and Practice32, 4 (2024),
579–598. https://doi.org/10.1080/10696679.2023.2249213
[25] Shintaro Okazaki, Barbara Mueller, and Charles R. Taylor. 2010. Measuring Soft-
Sell versus Hard-Sell Advertising Appeals.Journal of Advertising39, 2 (2010),
5–20. https://doi.org/10.2753/JOA0091-3367390201
[26] Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel,
Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss,
Vincent Dubourg, et al .2011. Scikit-learn: Machine Learning in Python.Journal
of Machine Learning Research12, 85 (2011), 2825–2830. http://jmlr.org/papers/
v12/pedregosa11a.html
[27] Perplexity. 2024. Why we’re experimenting with advertising. https://www.
perplexity.ai/hub/blog/why-we-re-experimenting-with-advertising. (Dec. 2024).
Perplexity.ai. Accessed: 2026-01-19.
[28] Louvins Pierre. 2024. The Effect of Covert Advertising Recognition on Consumer
Attitudes: A Systematic Review.Journal of Marketing Communications30, 8
(2024), 1077–1098. https://doi.org/10.1080/13527266.2023.2184851
[29] Colin Porlezza. 2017. Digitaler Journalismus zwischen News und Native Adver-
tising - Risiken und Nebenwirkungen einer heiklen Beziehung. InAbbruch -
Umbruch - Aufbruch, Werner A. Meier (Ed.). Nomos Verlagsgesellschaft mbH & Co.
KG, Baden-Baden, Germany, 249–270. https://doi.org/10.5771/9783845276663-249
[30] L. A. Ramshaw and M. P. Marcus. 1999. Text Chunking Using Transformation-
Based Learning. InNatural Language Processing Using Very Large Corpora, Susan
Armstrong, Kenneth Church, Pierre Isabelle, Sandra Manzi, Evelyne Tzouker-
mann, and David Yarowsky (Eds.). Springer Netherlands, Dordrecht, 157–176.
https://doi.org/10.1007/978-94-017-2390-9_10
[31] Sravana Reddy, Yongze Yu, Aasish Pappu, Aswin Sivaraman, Rezvaneh Rezapour,
and Rosie Jones. 2021. Detecting Extraneous Content in Podcasts. InProceedings of
the 16th Conference of the European Chapter of the Association for Computational
Linguistics: Main Volume. Association for Computational Linguistics, Online,
1166–1173. https://doi.org/10.18653/v1/2021.eacl-main.99
[32] Radim Řehůřek and Petr Sojka. 2010. Software Framework for Topic Modelling
with Large Corpora. InProceedings of the LREC 2010 Workshop on New Challenges
for NLP Frameworks. European Language Resources Association (ELRA), Valletta,
Malta, 45–50. https://doi.org/10.13140/2.1.2393.1847
[33] Kya Sainsbury-Carter. 2025. Transforming the future of audience engagement.
https://about.ads.microsoft.com/en/blog/post/march-2025/transforming-the-
future-of-audience-engagement?s_cid=gl-ob-impc-gai-src_wpage-sub_oooc-
cam_accelerate-flx_copilotlp. (May 2025). Microsoft Advertising. Accessed:
2026-01-19.
[34] Erin E. Schauster, Patrick Ferrucci, and Marlene S. Neill. 2016. Native Ad-
vertising is the New Journalism: How Deception Affects Social Responsibility.
American Behavioral Scientist60, 12 (2016), 1408–1424. https://doi.org/10.1177/
0002764216660135
[35] Sebastian Schmidt, Ines Zelch, Janek Bevendorff, Benno Stein, Matthias Hagen,
and Martin Potthast. 2024. Detecting Generated Native Ads in Conversational
Search. InCompanion Proceedings of the ACM Web Conference 2024. ACM, Singa-
pore Singapore, 722–725. https://doi.org/10.1145/3589335.3651489

Detecting RAG Advertisements Across Advertising Styles ArXiv, January 2026, webis.de
[36] Fidji Simo. 2026. Our approach to advertising and expanding access to Chat-
GPT. https://openai.com/index/our-approach-to-advertising-and-expanding-
access/. (Jan. 2026). OpenAI. Accessed: 2026-01-211.
[37] Peter Snyder, Antoine Vastel, and Ben Livshits. 2020. Who Filters the Fil-
ters: Understanding the Growth, Usefulness and Efficiency of Crowdsourced
Ad Blocking. InAbstracts of the 2020 SIGMETRICS/Performance Joint Interna-
tional Conference on Measurement and Modeling of Computer Systems, Edmund
Yeh, Athina Markopoulou, and Y. C. Tay (Eds.). ACM, Boston, MA, USA, 75–76.
https://doi.org/10.1145/3393691.3394228
[38] Jennifer Sor. 2025. Sam Altman touts ChatGPT’s 800 million weekly users, double
all its main competitors combined. https://www.businessinsider.com/chatgpt-
users-openai-sam-altman-devday-llm-artificial-intelligence-2025-10. (Oct.
2025). Business Insider. Accessed: 2026-01-19.
[39] Ermis Soumalias, Michael J. Curry, and Sven Seuken. 2024. Truthful Aggregation
of LLMs with an Application to Online Advertising. (2024). https://doi.org/10.
48550/ARXIV.2405.05905 arXiv:cs.GT/2405.05905
[40] Philip J. Stone, Dexter C. Dunphy, and Marshall S. Smith. 1966.The General
Inquirer: A Computer Approach to Content Analysis.M.I.T. Press, Oxford, England.
Pages: 651.[41] Abu Sultan. 2025. OpenAI projects 220 million paying ChatGPT users by 2030, The
Information Reports. https://www.reuters.com/technology/openai-projected-
least-220-million-people-will-pay-chatgpt-by-2030-information-2025-11-26/.
(Nov. 2025). Reuters. Accessed: 2026-01-19.
[42] Demetris Vrontis, Anna Makrides, Michael Christofi, and Alkis Thrassou. 2021.
Social media influencer marketing: A systematic review, integrative framework
and future research agenda.International Journal of Consumer Studies45, 4 (July
2021), 617–644. https://doi.org/10.1111/ijcs.12647
[43] Bartosz W. Wojdynski. 2016. The Deceptiveness of Sponsored News Articles:
How Readers Recognize and Perceive Native Advertising.American Behavioral
Scientist60, 12 (2016), 1475–1491. https://doi.org/10.1177/0002764216660140
[44] Bartosz W. Wojdynski and Nathaniel J. Evans. 2020. The Covert Advertising
Recognition and Effects (CARE) Model: Processes of Persuasion in Native Ad-
vertising and Other Masked Formats.International Journal of Advertising39, 1
(2020), 4–31. https://doi.org/10.1080/02650487.2019.1658438
[45] Ines Zelch, Matthias Hagen, and Martin Potthast. 2024. A User Study on the
Acceptance of Native Advertising in Generative IR. InProceedings of the 2024
ACM SIGIR Conference on Human Information Interaction and Retrieval. ACM,
Sheffield United Kingdom, 142–152. https://doi.org/10.1145/3627508.3638316