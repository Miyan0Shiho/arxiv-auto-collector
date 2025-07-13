# Ragged Blocks: Rendering Structured Text with Style

**Authors**: Sam Cohen, Ravi Chugh

**Published**: 2025-07-09 00:26:52

**PDF URL**: [http://arxiv.org/pdf/2507.06460v1](http://arxiv.org/pdf/2507.06460v1)

## Abstract
Whether it be source code in a programming language, prose in natural
language, or otherwise, text is highly structured. Currently, text
visualizations are confined either to _flat, line-based_ decorations, which can
convey only limited information about textual structure, or _nested boxes_,
which convey structure but often destroy the typographic layout of the
underlying text. We hypothesize that the lack of rich styling options limits
the kinds of information that are displayed alongside text, wherever it may be
displayed.
  In this paper, we show that it is possible to achieve arbitrarily nested
decorations while minimally disturbing the underlying typographic layout.
Specifically, we present a layout algorithm that generates _ragged blocks_, or
_rocks_, which are rectilinear polygons that allow nested text to be compactly
rendered even when styled with borders and padding.
  We evaluate our layout algorithm in two ways. First, on a benchmark suite
comprising representative source code files in multiple programming languages,
we show that the (ragged block) layouts produced by our algorithm are
substantially more compact than the (rectangular block) layouts produced by
conventional techniques, when uniformly styling every element in the syntax
tree with borders and padding. Second, through a small gallery of usage
scenarios, we demonstrate how future code editors, word processors, and other
document-rendering GUIs might convey rich semantic information through
domain-specific styling of ragged blocks.

## Full Text


<!-- PDF content starts -->

Ragged Blocks: Rendering Structured Text with Style
Sam Cohen
samcohen@uchicago.edu
University of Chicago
Chicago, IL, USARavi Chugh
rchugh@cs.uchicago.edu
University of Chicago
Chicago, IL, USA
Flat Text Layout BoxesLayout RocksLayout RocksLayout, Simplified
Figure 1: A code fragment rendered without nested styling, with nested boxes, with rocks, and with simplified rocks.
Abstract
Whether it be source code in a programming language, prose in
natural language, or otherwise, text is highly structured. Currently,
text visualizations are confined either to flat, line-based decorations,
which can convey only limited information about textual structure,
ornested boxes , which convey structure but often destroy the ty-
pographic layout of the underlying text. We hypothesize that the
lack of rich styling options limits the kinds of information that are
displayed alongside text, wherever it may be displayed.
In this paper, we show that it is possible to achieve arbitrarily
nested decorations while minimally disturbing the underlying ty-
pographic layout. Specifically, we present a layout algorithm that
generates ragged blocks , orrocks , which are rectilinear polygons
that allow nested text to be compactly rendered even when styled
with borders and padding.
We evaluate our layout algorithm in two ways. First, on a bench-
mark suite comprising representative source code files in multiple
programming languages, we show that the (ragged block) layouts
produced by our algorithm are substantially more compact than the
(rectangular block) layouts produced by conventional techniques,
when uniformly styling every element in the syntax tree with bor-
ders and padding. Second, through a small gallery of usage sce-
narios, we demonstrate how future code editors, word processors,
and other document-rendering GUIs might convey rich semantic
information through domain-specific styling of ragged blocks.
CCS Concepts
•Human-centered computing →Graphical user interfaces ;Vi-
sualization ;•Software and its engineering →Integrated and
visual development environments ;•Computing methodologies
→Computer graphics .
Keywords
Structured Text, Text Layout, Ragged Blocks, Program Visualization1 Introduction
Document-rendering GUIs currently do not provide expressive
mechanisms for styling and rendering structured text.
In rich text editors and word processors, such as Google Docs
and Microsoft Word, document authors call attention to parts of
their documents using text decorations . These decorations include
underlines, italics, foreground and background colors, and inline
glyphs, among others. When communicating information back to
users, these tools also use such decorations; for example, wavy
underlines are commonly used to identify
speling errors and op-
portunities to make grammar
gooder . These decorations are flat:
they can appear only within or between lines of text, and are thus
inherently limited in their capacity to identify structures of text
besides tokens and lines.
In traditional code editors and IDEs, such as CodeMirror and VS
Code, program authors do not directly apply styles to the source
text. But these interfaces typically render programs with syntax
highlighting—where tokens are colored based on lexical and “parts-
of-speech”–type information—and employ flat decorations, such as
wavy underlines and simple glyphs, to identify errors.
In contrast to traditional code editors, block-based editors , such as
Scratch [Resnick et al .2009] and Snap! [Harvey et al .2013], employ
shapes that allow certain code fragments to be “snapped” together
via mouse-based direct manipulation. Other structure editors , such
as Sandblocks [Beckmann et al .2020, 2023], render rectangular
blocks (i.e., boxes) for all syntactic substructures in an otherwise
text-based, keyboard-driven editor.
Sadly, current text layout algorithms for all of the above sys-
tems compromise too much in order to surface structure. They are
typically line-based, where visual indications of structure cannot
cross the line boundary, or block-based, which often force the un-
derlying text into a layout which does not resemble its source. The
“Boxes Layout” in Figure 1 shows an example deficiency of the
latter approach.arXiv:2507.06460v1  [cs.HC]  9 Jul 2025

Sam Cohen and Ravi Chugh
A
 C
 B
Figure 2: Factorial, List Comprehension, and Color Picker. Each is shown rendered as rocks, with and without simplification.
Ragged Blocks. The idea that drives this work is to use ragged
blocks (orrocks ), rather than rectangles, to render nested substruc-
tures of text for rich styling. Analogous to how “ragged right” text
is aligned on the left margin but unaligned on the right, a rock is
an arbitrary rectilinear polygon, which is “ragged on all sides.”
Figure 2 demonstrates three code snippets rendered as ragged
blocks:
A a factorial function written in Haskell,
B a list compre-
hension with multiple iterables in Python, and
C an error message
component in TypeScript. The top row of this figure shows com-
pact rock layouts, where rocks for different substructures have
sometimes highly irregular boundaries in order to tightly fit the
text contained within. Once tight-fitting rock layouts have been
computed, it is possible to derive simplified rock layouts (the sec-
ond row of Figure 2), where some corners have been removed (i.e.,
some edges have been straightened) without further affecting the
position of laid-out elements.
Although our primary motivation is text, the notion of ragged-
block layout is agnostic to the contents of individual elements. For
example, Figure 2
C shows a color-picking GUI compactly embed-
ded within the surrounding program text. Projectional editors , such
as Hazel [Omar et al .2021] and MPS [JetBrains 2024], enrich code
editors with GUIs in this way, but—like block-based and structure
editors—these systems do not render nested blocks as compactly as
shown in Figure 2.
Contributions. This paper makes several contributions. First, we
present a family of layout algorithms that compactly render highly
structured text documents as nested ragged blocks. The key chal-
lenge is the presence of newlines in the source text. Whereas con-
ventional nested-box layout algorithms limit the visual effect of a
newline to the box containing it, a ragged-block layout algorithm
must visually reconcile arbitrarily large subtrees separated by a
newline. We formulate a data structure, called a region , that reifies
the layered steps in our layout algorithm in a way that allows cross-
line visual constraints to be permeated and reconciled efficiently.Second, we evaluate the layout algorithm on a benchmark suite
comprising 7.6k lines of code written in Haskell, Python, and Type-
Script, by measuring the compactness of rock layouts compared to
conventional nested-box layout techniques. For this evaluation, we
uniformly render all substructures in the syntax tree with border
and padding; these “blocks-everywhere” visualizations serve as a
stress test for nested text layout.
Third, we present a small gallery of examples that, compared
to visualizing all blocks, include finer-grained stylistic choices for
specific tasks: highlighting differences between two versions of
document, as well as conveying error messages and other seman-
tic information about its content. These examples involve both
programming- and natural-language documents, hinting at how
future code editors and document-rendering GUIs could employ
domain-specific styling of ragged blocks as a rich channel for com-
municating information to users.
Outline. The rest of this paper is organized is follows. In §2, we
discuss related work with a focus on program visualization since it
is a particularly rich source of examples for structured text layout.
In §3, we motivate and present the region data structure, which
underlies our family of ragged block layout algorithms. We describe
the simplest version of our layout algorithm, called L1P, in detail.
The label “P” indicates the use of pure regions in this algorithm.
The next section presents several extensions to the base algo-
rithm, L1P. In §4.1 and §4.2, we describe two modest variations,
called L2a PandL2bP, which provide more nuanced treatment of
whitespace characters for code and prose, respectively. In §4.3, we
present an alternative representation of regions, called stateful re-
gions, which enables sharing among regions and thus significantly
improves performance; the correspondingly revised algorithms,
called L1S,L2a S, and L2bS, respectively, are labeled “S.”
Each of these layout algorithms computes tightly-packed ragged
blocks. In § 5, we define an optional post-processing algorithm,
called simplification , that makes borders less ragged where possible
without affecting the positions of leaf fragments in the layout.

Ragged Blocks
Having presented our family of optimized layout and simplifica-
tion algorithms, the next two sections serve as evaluation. In §6,
we devise a metric for measuring the compactness of a text layout,
and show that L1Sproduces more compact text layouts than those
produced by a conventional nested-box algorithm. In §7, we discuss
further applications of rocks layout (specifically, versions L2a Sand
L2bS), before concluding in §8.1
2 Related Work
2.1 Editors for Structured Text
Structured text abounds. How is it currently rendered?
Code Editors. There is a long history of work on block-based ,
structure , and projectional code editors that blend source code text
and visual elements in myriad ways. Figure 3 summarizes the design
space of such editors using several representative example systems,
organized around two dimensions: the primitive shapes used for
layout, and whether or not these shapes can be nested horizontally
and/or vertically.
The simplest shape for rendering nested elements is the rectan-
gle. In a typical flat code editor, such as CodeMirror, VS Code, or
Vim, each line of text can be understood as a sequence of adja-
cent rectangles. Compared to this line-based approach, in Sand-
blocks [Beckmann et al .2020, 2023] rectangles can be arbitrarily
nested; this approach completely reveals the text structure, at the
cost of large layouts that differ significantly from the underlying,
unstyled text. In between such approaches are systems, such as
Fructure [Blinn 2019] and tylr [Moon et al .2023], which allow
nested decorations to affect horizontal spacing but limit the use of
vertical spacing—the fundamental challenge that stands in the way
of compactly rendering nested elements.
Another type of primitive layout element, dubbed s-blocks by
Cohen and Chugh [2025], are rectangles but where the top-left and
bottom-right corners are potentially missing. As such, s-blocks re-
semble the way in which selected-text regions are displayed in many
GUIs. Their system, called Hass, includes a layout algorithm that
generates s-blocks to render nested, potentially multiline strings.
Like ours, their algorithm attempts to minimally disturb the text
layout as compared to unstructured text. However, it is essentially a
line-by-line approach, with judicious use of spacer glyphs through-
out the text to give the appearance of nesting. Their results are
somewhat more compact than nested rectangles, but even on small
1In the rest of the paper, we write Rocks (typeset in small caps) to refer collectively to
our implementations of the layout algorithms, together with a simple parsing pipeline
that allows us to generate the examples depicted throughout. Our implementation will
be publicly released after publication.
Nestable?Horiz/Vert Sandblocks Hass Rocks
Horiz Fructure, tylr — —
No text editors text selection Deuce
Rectangles
 S-Blocks
 Rocks
Primitive Shapes
Figure 3: Design Space of Structure Editors for Codeexamples the resulting layouts include undesirable visual artifacts,
such as unusually large gaps between certain lines, and misalign-
ment of characters from the preformatted text.
Compared to the four corners of a rectangle and at most eight
corners of an s-block, arbitrary rectilinear polygons (which we call
ragged blocks ) provide additional opportunities for tightly packing
nested text. When hovering over the source text, the Deuce edi-
tor [Hempel et al .2018] renders a ragged block that tightly wraps
the corresponding substructure. But unlike in Rocks , Deuce does
not allow ragged blocks to be nested.
Beyond the axes in Figure 3—shapes and ability to nest them—
there are others we might use to categorize the diverse set of tools
which visualize structured text. The set of primitive glyphs, for ex-
ample, can take many forms. Projectional editors, such as MPS [Jet-
Brains 2024] and Hazel [Omar et al .2021], among others [Andersen
et al.2020; Erwig and Meyer 1995; Ko and Myers 2006; Omar et al .
2012], variously permit tables, images, and other domain-specific
visuals to intermix with source text. In this regard, the algorithms
in this paper take a lower-level view of the world: the leaf elements
are rectangles, and clients of Rocks are responsible for drawing
content within them, be it text or graphics.
Furthermore, the design space we present is concerned with the
display a single “node” of text. “Nodes-and-wires” interfaces, such
as Scratch [Resnick et al .2009] and LabVIEW [National Instruments
2024], among others, display multiple program fragments, which
can be freely rearranged by the user on a two-dimensional canvas.
Manual positioning of multiple nodes is a separate concern from
ours—the presentation of any single node of program text.
Interfaces for Program Analyzers. Another rich source of exam-
ples are interfaces which are designed for the manipulation, analy-
sis, or synthesis of programs and text.
The authors of LAPIS [Miller and Myers 2002], for example, ex-
plain how text editors could be extended to use multiple, potentially
overlapping text selections. They recognize a fundamental concern
with rendering multiple selections using flat decorations, noting
that two adjacent selections “would be indistinguishable from a
single selection spanning both regions.” The STEPS [Yessenov et al .
2013] system uses programming by example to extract structure
from unstructured “raw” text. Users can then perform queries and
transformations on the structure-annotated output. The system
uses a blocks-like display for visualizing the extracted structure.
Inspired by LAPIS and STEPS, FlashProg [Mayer et al .2015] is a
tool for searching through the large space of specification-satisfying
programs for structured text extraction. The user interface devel-
oped for the system visualizes the structure of the extracted text
with a line-by-line approach. The authors also note the importance
of nested text selections, lamenting that “LAPIS unfortunately does
not have good support for nested and overlapping regions, which
are essential for data extraction tasks.”
Several other systems, such as reCode [Ni et al .2021], CoLad-
der [Yen et al .2024], and Ivie [Yan et al .2024], use blocks, inline
diffs, and other visualizations to help interactively generate and
explain synthesized code fragments.
Prose Editors. Some editors for natural language also visualize
structure. In Textlets [Han et al .2020], users can manipulate con-
current text selections to help maintain consistency, perform large

Sam Cohen and Ravi Chugh
edits, or generate summary information in legal documents. In
FFL [Wu et al .2023], users can apply flat text decorations and at-
tach annotations to structural components of math formulas. Rocks
may provide methods to imbue these kinds of visualizations with
even more structural detail.
2.2 Layout Algorithms for Text
A separate source of related works are those concerning text layout
in general, independent of particular application domains. In Rocks ,
we adapt the TeX [Knuth and Plass 1981] line-breaking algorithm
in our own algorithm ( L2bS) for rendering justified text.
The HTML and CSS box model [Mozilla 2025] supported by
browsers provide the option to display rectangles either inline or
as blocks, corresponding to the flat and nested extremes discussed
above. Penrose [Ye et al .2020] and Bluefish [Pollock et al .2024]
are systems for specifying diagrams; the former uses numerical
optimization to compute layouts, while the latter uses a more local
constraint propagation model.
3 Initial Layout Algorithm ( L1P)
In order to describe text layout, it is convenient to forget that
the fundamental objects to be laid out are words or letters, and
instead consider only rectangles. The layout algorithm need not be
concerned with what is inside these rectangles. There could be a
single letter, a whole word, a sequence of words, or even an image
or an interactive visualization. Importantly, these rectangles will
be considered indivisible for the purposes of layout. To differentiate
these atomic rectangles from, say, the rectangles in a boxes layout,
we will refer to them as fragments .
Let us forget for a moment that we care about rendering struc-
tured text, and consider the case of unstructured text. Then, we
could represent a text layout by the following grammar:
𝑢 ::=[atom ]
atom ::=𝑥𝑖|Newline
An unstructured layout, 𝑢, is a sequence of atom s, each of which
is either a fragment, 𝑥𝑖, or a Newline .
Then, finding the position of each 𝑥𝑖amounts to laying out
each 𝑥𝑖in sequence, placing the left edge of each 𝑥𝑖+1at position
left𝑥𝑖+width 𝑥𝑖, unless we encounter a Newline , in which case the
left edge of the next 𝑥𝑖is 0. The amount that we translate a fragment
𝑥𝑖+1from the left side of its left neighbor is called the advance of the
fragment. In the case of unstructured text, advance 𝑥𝑖=width 𝑥𝑖,
but we will retain the notion of advance since it more easily gener-
alizes to the case of structured text layout.
We can describe the input to a structured text layout algorithm
by enriching the input grammar with a constructor for Node s:
𝑒 ::=Atom(atom)|Node id([e],padding)
atom ::=𝑥𝑖|Newline
Here, 𝑒is asyntax tree , and 𝑥𝑖are fragments. Figure 4
A shows
an example syntax tree for a small text layout involving three
fragments. The grammar of syntax trees is useful for expressing a
tree structure inherent to a document, but it is not convenient for
specifying the semantics of structured text layout. The problem is
that the constructors for Node andNewline are not on equal footing;
Node is a node in a rose tree, whereas Newline is a delimiter in alist of subtrees. In a rocks layout, however, Node s and Newline s
are equivalently important sources of structure. It is convenient,
then, to treat Node sandNewline s as non-leaf nodes.
The grammar of layout tree s,𝑡, given in Figure 5 does just this,
introducing the JoinV ,JoinH , and Wrap constructors, which to-
gether supplant the combination of Node andNewline . Figure 4
B
shows the layout tree which corresponds to the syntax tree given
in Figure 4
A . This grammar breaks Node s into two distinct con-
cepts: the concept of Wrap -ping a subtree in some padding, and the
horizontal concatenation of subtrees (denoted by JoinH ).Newline s
are expressed by vertical concatenation (denoted by JoinV ).
Translation from syntax to layout tree is a straightforward re-
parsing of the children in each Node . We can obtain an equivalent
x0
x2x1x2x0
x1Stack( region=[
],
,
,[Cell0(8),Cell1(8),Cell2(4)]),
Stack( [Cell0(8),Cell1(8),Cell3(4)]),
Stack( [Cell0(0)]),Node0( ,0)
Node1( ,4)
Node2( ,4)Node3( ,4)
Atom(x0)Atom(x1)Atom(x2) Newline
Wrap0( ,0)
Wrap1( ,4)
Wrap2( ,4)Wrap3( ,4)JoinV( , )
JoinH( , )
Atom(x0)Atom(x1)Atom(x2)SyntaxTree
LayoutTree
Region
RenderedRegionA
DB
C
Figure 4: A syntax tree using Node s and Newlines , its cor-
responding layout tree using Wrap ,JoinV , and JoinH , its
derived region, and a rendering of the region.

Ragged Blocks
layout tree by treating the children of each Node in the syntax tree
as a sequence of tokens, and parsing the token list as an application
of infix operators, JoinH between two non-newline nodes, and JoinV
between a node and a Newline .
Given a layout tree, the purpose of layout is to determine the
location of its leaves (i.e. the fragments), and furthermore the di-
mensions of a ragged block surrounding each Node . Our algorithm
involves several operations on a data structure, called regions, and
several options on layout trees; we describe each group below.
3.1 Regions
We can define the semantics of a layout over a layout tree by way
of analogy to layout over the unstructured text. We do this first
by re-defining advance as an operation which considers the local
padding around a fragment.
Advance. Unfortunately, we cannot define advance in terms of
the raw fragments. We need to construct an auxiliary data structure
containing information about the ancestry (in the layout tree) of
each fragment. This is because the space between two fragments
in the layout is a function of the fragments’ position in the layout
tree. Two sibling fragments (like 𝑥0and𝑥1in Figure 4) which are
wrapped by a common ancestor ( Wrap1in Figure 4), ought to be
placed closer together than two fragments which don’t share the
ancestor (such as 𝑥0and𝑥2, for example). This is the mechanism by
which we can allocate space for borders around Wrap -ped nodes
in the layout. We call this auxiliary data structure a region .
A region is a list of Stack s, and a Stack is a fragment paired with
a list of Cells. Each Cell records a Wrap node which occurs above
itsStack ’s fragment in the layout tree. For example, in Figure 4,
theStack s for fragments 𝑥0and 𝑥1each have three Cells since
a path from 𝑥0or𝑥1up the layout tree to the root encounters
three Wrap nodes. The argument to a Cell records the cumulative
padding applied due to a Wrap node (see Cell 0, for example, in
Figure 4; its corresponding Wrap node applies nopadding, and
soCell 0’s argument is unchanged from its successor). By storing
the cumulative padding, the wrapped rectangle corresponding to a
Stack can be found without chasing the list of Cells.
We provide here two plausible (but incorrect) definitions of ad-
vance which illustrate the importance of the Cell in determining if
twoStack s are compatible .
advance unsound
Stack (𝑥𝑎,Cell(id𝑎,Σ𝑝𝑎):rest𝑎)
Stack (𝑥𝑏,Cell(id𝑏,Σ𝑝𝑏):rest𝑏)
= width( 𝑥𝑎)
x2x1x0
advance conservative
Stack (𝑥𝑎,Cell(id𝑎,Σ𝑝𝑎):rest𝑎)
Stack (𝑥𝑏,Cell(id𝑏,Σ𝑝𝑏):rest𝑏)
= width( 𝑥𝑎) +Σ𝑝𝑎+Σ𝑝𝑏
x0
x2x1
The first definition is a naive re-interpretation of our notion of
advance from flat text layout which ignores padding completely.
This leads to unsound layouts because space allocated for a Wrap
in the layout tree might overlap unrelated rectangles. The second
definition takes the outermost, maximal padding at each Cell, and
says that the advance is the width of 𝑥𝑎plus the maximal padding
that can occur between 𝑥𝑎and𝑥𝑏. This definition of advance issound, but it is too conservative. Elements in the layout tree which
are underneath the same Wrap node ought to share space.
The proper definition of advance in Figure 5 uses a helper func-
tion, spaceBetween, which traverses the Stack s in parallel, shedding
Cells which derive from the same Wrap node. Alternatively, we can
think of spaceBetween as finding the padding due to the lowest
common ancestor Wrap node of the arguments.2Once we reach
the end of the stack, or a pair of Cells which derive from different
parts of the layout tree (as determined by the Cell’sid, which corre-
sponds to the idof aWrap node in the layout tree), the cumulative
padding of these Cells is the padding that must be applied between
the fragments.
Leading. Careful readers of the definition of spaceBetween may
note that this operation doesn’t consider the dimension of the
rectangles associated with each cell. The minimum space between
two fragments is only a function of the fragment’s Stack (and, hence,
thestructure of the layout tree), not the dimension or position of
the fragment. Therefore, we can use spaceBetween to not only
decide the horizontal spacing of rectangles (i.e. advance), but also
thevertical space between rectangles, known as the leading .3
We define leadingX𝑥𝑖𝑥𝑗as the amount of space needed to put
rectangle 𝑥𝑗entirely below 𝑥𝑖, or 0 if 𝑥𝑖and𝑥𝑗don’t overlap hori-
zontally. This definition can be easily extended to Stack s by finding
two rectangles which are centered on 𝑥𝑖and𝑥𝑗, but are “inflated”
by the space between 𝑥𝑖and𝑥𝑗respectively. The leading between
two regions is the maximum of the leadingSbetween each pair of
stacks in the regions.
The process of layout will combine regions together until we
have a single region representing each line. Then, these lines are
combined into a single region. The leadingRfunction is used to find
the minimum space to leave between two successive lines such that
no two Stack s improperly overlap.
Union & Wrap. But how do we actually combine regions? One
of the principal advantages of representing regions as an implicit
union of rectangles is that combining regions is equivalent to con-
catenating their Stack s. The order of a region’s stacks does not
matter (but as we’ll see in §5 it can be exploited), and we make no
attempt to resolve collisions between stacks (this is the concern of
layout, which we will address next).
3.2 Layout
Algorithm L1Pworks analogously to layout of unstructured, pre-
formatted text. First, each line is laid out individually through the
repeated application of horizontal and vertical concatenation and
wrapping operations. Figure 6 shows the application of these op-
erators to produce a layout for Abs. The lines, once laid out, are
combined at the end to form a finished layout (see Figure 6
C ).
Figure 5 gives the type of an L1Playout. It is a list of lines, each
line represented by a single region. Alongside each region is a
vector, v, which is the advance of the region. The advance vector
records the position, relative to a region’s origin, that a region
2Strictly speaking, the padding due to the lowest common ancestor (LCA) Wrap node,
plus the padding ascribed to any Wrap nodes on a path from the LCA to the root of
the tree, since a Cellstores cumulative padding, not the padding due to a single Wrap .
3We use the term leading in this paper to mean the space between baselines (as opposed
to the inter-line spacing).

Sam Cohen and Ravi Chugh
𝑡 ::=Atom(𝑥𝑖)|Wrapid(𝑡,padding)|JoinV(𝑡, 𝑡)|JoinH(𝑡, 𝑡)
stack ::=Stack(𝑥𝑖,[Cell id(Σ𝑝)])
region ::=[Stack]
layout ::=[(region ,v)]
Advance
spaceBetween[][] =(0,0)
spaceBetween[]Cell id𝑏(Σ𝑝𝑏):rest𝑏=(0,Σ𝑝𝑏)
spaceBetween Cell id𝑎(Σ𝑝𝑎):rest𝑎[]=(Σ𝑝𝑎,0)
spaceBetween(Cell id𝑎(Σ𝑝𝑎):rest𝑎)(Cell id𝑏(Σ𝑝𝑏):rest𝑏)
=(Σ𝑝𝑎,Σ𝑝𝑏) when id𝑎≠id𝑏
=spaceBetween rest𝑎rest𝑏 otherwiseadvance Stack(𝑥𝑎,cells𝑎)Stack(𝑥𝑏,cells𝑏)=width 𝑥𝑎+Σ𝑝𝑎+Σ𝑝𝑏
where(Σ𝑝𝑎,Σ𝑝𝑏)=spaceBetween cells𝑎cells𝑏
Leading
leadingX𝑥𝑖𝑥𝑗=bottom 𝑥𝑖−top𝑥𝑗when 𝑥𝑖and𝑥𝑗horizontally overlap
=0 otherwise
leadingSStack(𝑥𝑖,cells𝑎)Stack(𝑥𝑗,cells𝑏)=leadingX(inflate 𝑥𝑖Σ𝑝𝑎)(inflate 𝑥𝑗Σ𝑝𝑏)
where(Σ𝑝𝑎,Σ𝑝𝑏)=spaceBetween cells𝑎cells𝑏
leadingRregion𝑎region𝑏=maximum(leadingSstack𝑖stack𝑗)
for all stack𝑖∈region𝑎,stack𝑗∈region𝑏
Union & Wrap
union R𝑎 𝑏=concatenate 𝑎 𝑏 wrapSid padding Stack(𝑥𝑖,cells)=Stack(𝑥𝑖,Cell id(padding):cells)
wrapRid padding stacks =[wrapSid padding stack]for each stack instacks
Layout
union L(𝑎,v𝑎)(𝑏,v𝑏)=(union R𝑎 𝑏,v𝑎+v𝑏) wrapLid padding(𝑟𝑒𝑔𝑖𝑜𝑛, v)=(region′,v+⟨2×padding ,0⟩)
where region′=(wrapSid padding)translated right by padding .
joinV𝑎 𝑏=concatenate 𝑎 𝑏
layout 𝑥𝑖=[([Stack(𝑥𝑖,[])],⟨width 𝑥𝑖,0⟩)]
layout JoinV(𝑡1, 𝑡2)=join𝑉(layout 𝑡1)(layout 𝑡2)
layout JoinH(𝑡1, 𝑡2)=join𝐻(layout 𝑡1)(layout 𝑡2)
layout Wrapid(𝑡,padding)=wrap𝐿id padding(layout 𝑡)joinH[]𝑏=𝑏
joinH𝑎[]=𝑎
joinH𝑎 𝑏=concatenate 𝑎′𝑙 𝑏′
where
𝑎′are all but the last line of 𝑎,
𝑏′are all but the first line of 𝑏, and
𝑙is the union Lof the last line of 𝑎and𝑏T,
where
𝑏Tis the first line of 𝑏translated by the
last line of 𝑎’s advance.
Figure 5: Algorithm L1P

Ragged Blocks
joinHjoinH
joinH
joinH
joinHjoinHjoinH
wrapLwrapLwrapL
wrapLjoinVInput Fragments:
Layout Construction
Line Merging18 omitted steps
r1=
r2=
r3=
leadingRr2r3leadingRr1r2
Finished LayoutA
B
C
Figure 6: Example L1Playout steps.following it should be placed. When two regions in the layout are
combined with union R, or a region in the layout is wrapped, the
advance vector is updated accordingly. Figure 7 shows an example
of how the advance vector is updated for each of these operations.
x0
x2x1
x0x1x0
x2x1
x3 x3
Region Originx0x1unionL
wrapL
World Origin
Figure 7: Example applications of union Land wrapL.
Layout proceeds as a structural recursion over the layout tree, t.
During this recursion, we apply one of two join operations, joinH
andjoinV. These join functions combine two layouts into a sin-
gle layout, either joining them horizontally ( joinH) or vertically
(joinV). Vertical concatentation amounts to concatenation of the
layout’s lines. Horizontal concatentation splices the two abutting
lines together, using union to combine them.
Figure 6 gives several examples of joinHandjoinVin action. The
figure can be read as a sequence of stack operations, representing a
possible order of operations in the recursive evaluation of layout.
Each element of the stack is a single layout, and the top of the
stack is on the right. joinHandjoinVoperate on the two topmost
layouts in the stack, joining them either horizontally or vertically
respectively. (In the case where a new element is pushed and then
immediately joined with joinH, the push is omitted.)
3.3 Spacers
Figure 8: Example L1Play-
out without spacers.The algorithm given in Figure 5 is
enough to produce usable struc-
tured text layouts, but it does
not elegantly handle whitespace.
Specifically, preceding whitespace
on a line must be represented by a
fragment. Futhermore, this frag-
ment is not exempt from wrap-
ping. In practice, this results in un-
sightly layouts where preceding whitespace draws unneeded visual
attention (e.g., see Figure 8).
In order to fix this visual defect, we introduce a new kind of leaf
called a Spacer . The Spacer behaves similarly to an Atom except
it has no height and does not participate in wrapping. Figure 9
details the modifications to L1Pneeded to support this new object.
TheSpacer ’s invariance to wrapping is implemented in the new
definition of advance, which ignores the spaceBetween any object
and a Spacer .
4 Extensions
Next, we describe several extensions to the basic algorithm. We
sketch the main ideas with less detail than in the previous section.

Sam Cohen and Ravi Chugh
𝑡 ::=Atom(𝑥𝑖)|Spacer(𝑤𝑖)|Wrapid(t,padding)|JoinV(t,t)|JoinH(t,t)
stack ::=Stack(𝑥𝑖,[Cell id(Σ𝑝)])|Spacer(𝑤𝑖)
Advance
advance Stack(𝑥𝑎,cells𝑎)Stack(𝑥𝑏,cells𝑏)=width 𝑥𝑎+Σ𝑝𝑎+Σ𝑝𝑏
where(Σ𝑝𝑎,Σ𝑝𝑏)=spaceBetween cells𝑎cells𝑏
advance Stack(𝑥𝑎,cells𝑎)Spacer(𝑤𝑏)=width 𝑥𝑎
advance Spacer(𝑤𝑎)𝑦=𝑤𝑎
Leading
leadingSStack(𝑥𝑖,cells𝑎)Stack(𝑥𝑗,cells𝑏)=leadingX(inflate 𝑥𝑖Σ𝑝𝑎)(inflate 𝑥𝑗Σ𝑝𝑏)
where(Σ𝑝𝑎,Σ𝑝𝑏)=spaceBetween cells𝑎cells𝑏
leadingS𝑥 𝑦=0
Figure 9: Algorithm L1P, continued (with spacers).
4.1 Layout with Column Constraints ( L2a P)
The finished layout produced by L1P(see Figure 6) succeeds in
visualizing the structure of the document. When writing code, au-
thors often align certain columns of text to aid readability, or draw
attention to some symmetry between adjacent lines of source text.
In the case of Figure 6, the original source program aligned the “ ?”
and “ :” characters, but this alignment was destroyed by the addi-
tion of padding. Layouts could be improved further by maintaining
additional constraints regarding formatting in code.
TheRocks layout shown in Figure 1, by contrast, was gener-
ated by algorithm L2a P, which maintains the alignment of columns
through the use of constraints. In algorithm L2a P, the spaceBe-
tween two fragments is not used to directly set the position of the
fragments, but to generate a lower bound constraint , ensuring that
no fragments come closer than the spaceBetween them, but are
permitted to space apart if necessary. The layout tree is modified to
include a notion of named constraint variables, allowing the client
to specify constraints on the horizontal position of fragments in
the layout. Appendix §A discusses this extension in more detail.
4.2 Layout with Automatic Line Breaks ( L2b P)
Algorithms L1PandL2a Ptake as input formatted text, where the
position of newlines is known. But, it is common that when laying
out prose, the document author does not choose explicit newlines.
Algorithm L2bPincludes a pre-processing step which finds good
locations for line breaks, inserts them, generating a new layout tree,
then generates a layout of the resulting tree.
Algorithm L2bPis based on the algorithm by Knuth and Plass
[1981] for choosing the line breaks in a paragraph. Because this
algorithm, too, defines a notion of advance between two adjacent
fragments, it is straightforward to substitute our new notion of ad-
vance, which considers the space between two fragments from their
relationship in the layout tree. Algorithm L2bPis demonstrated in
§7.2, and a more complete discussion of its implementation can be
found in Appendix §B.4.3 Stateful Regions ( L1S,L2a S,L2b S)
The pure regions presented in § 3 are conceptually simple and
convenient, but they are not efficient in practice. This is especially
true if we need to maintain several regions simultaneously. The
algorithms presented so far have no cause to do this, but as we’ll
see in §5, it is convenient to be able to annotate a node in the layout
tree with the region it corresponds to. However, pure regions make
this costly due to a lack of sharing.
For example, Figure 10 shows the regions due to the root and
the root’s child in the small example given in Figure 4. Most of
Figure 10
A and
B are the same; the difference being that the
fragments under the former region lack Cells due to Wrap0. In fact,
it is always the case that the Stack s in a parent node 𝑡will contain
all of the Cells of their children. Stateful regions capitalize on this
fact in order to reduce duplication between regions.
The generous sharing between pure regions in a layout also has
a consequence on a translate operation, not shown in Figure 5. If we
doannotate each region in the layout tree with its region, translate
becomes costly, the location of each fragment, 𝑥𝑖being duplicated
among nodes.
Stateful regions solve these problems by taking advantage of
two observations about how regions are used in layout. The first
observation is that during layout, no rectangles (and by extension,
noStack s) are added or removed. Said another way, it is possible, by
traversing the layout tree prior to layout, to know the complete set
of atomic fragments which will be involved in layout. We can use
this fact to modify the representation of Stack s so that they contain
areference to the rectangle they wrap, rather than a representation
of the rectangle itself:
region ::=[stack]
stack ::=Stack(𝑖 ,[Cell(id,Σ𝑝)])
rv ::=[𝑥0, . . . , 𝑥𝑛]
Now, each Stack points into a rectangle vector, rv, which is shared
between every region. This change obviates the need to traverse
the layout tree during layout translations, since a translation of a

Ragged Blocks
x2x0
x1Stack([
],
,
,[Cell0(8), Cell1(8), Cell2(4)]),
Stack( [Cell0(8), Cell1(8), Cell3(4)]),
Stack( [Cell0(0)]),regionOf(Wrap0( , 0)) = A
x2x0
x1Stack([
],
,
,[Cell1(8), Cell2(4)]),
Stack( [Cell1(8), Cell3(4)]),
Stack( []),regionOf(JoinV( , )) = BWrap0( , 0)
Wrap1( , 4)JoinV( , )
Atom(x2)
Figure 10: Regions of Figure 4
B , root and child.
layout node has the side effect of translating all of the regions in
any layout below it.
The second observation is that layout never takes the union of
two regions which are not adjacent. Indeed, if layout didtake the
union of two non-adjacent regions, it would correspond to a re-
ordering, omission, or duplication of fragments in the underlying
text, which are all operations which should obviously be avoided.
We can capitalize on this observation by representing regions not
as a list of Stack s, but as a span in a table of Cells which we call the
backing table . To maintain the partial persistenceof the region data
structure, invoking wrap𝑅on a region mutates the table by adding
a new row, thus ensuring that regions referring to less-wrapped
spans are unaffected.4
region ::= Region (begin ,end,depth )
Figure 11 shows the backing table that would be generated for
the small layout given in Figure 4. Two example regions correspond-
ing to Wrap0(purple) and Wrap1(orange) are highlighted in the
rendered output along with their span in the backing table.
The backing table is a function only of the structure of the layout
tree; it does not depend on the size or position of the layout’s
rectangles. This important quality means that the backing table
can be constructed before layout, by traversing the layout tree
and “simulating” the effect of wrapping and vertical and horizontal
concatenation.
The backing table bears strong resemblance to the pure region
formulation; each column contains the same Cells that a Stack cor-
responding to the same fragment would contain. However, whereas
Stack s are permitted to have different lengths, each column in the
4We call the stateful region partially persistent because the effects of translations are
visible to existing regions, but the effects of wraps are not.
Rectangle IndexDepth0
1
2
32 1
Cell0(8)Cell1(8)Cell2(4)
Cell1(8)Cell3(4) —
—
Cell0(8)Cell0(0)x0
x2x1Figure 11: Backing Table
backing table has the same depth. Furthermore, some Cells in the
backing table are duplicated within a single column. This is to en-
sure that every region corresponds to a span in a single row in the
backing table. (For example, the outermost region in Figure 11, cor-
responding to Wrap0occupies a single row, despite the fact that its
constituent rectangles have been wrapped between 1 and 3 times.)
The single-row property is enforced by a simple rule: when building
the backing table, whenever a vertical or horizontal concatenation
is encountered, the columns corresponding to the regions on the
left-hand side and right-hand side of the operator are “filled” to
the same depth by repeating the topmost element (or an empty
element, if the column is empty, rendered as “—” in Figure 11).
5 Simplification
The layouts we’ve demonstrated so far succeed in visually present-
ing the structure of the underlying text, but the polygonal outlines
of the regions generated by layout can be rather intricate. According
to usage scenarios and user preferences, we might like to generate
outlines which continue to visualize structure, but with less ragged
edges (i.e., fewer corners) if possible. Simplification is a way to
approach this goal.
However, when considering the simplicity of a rock, we are not
concerned with the simplicity of the underlying region representa-
tion (the union of a set of small rectangles), but rather the simplicity
of the boundary polygon which is a result of taking the union of
this set. Furthermore, the objects that we are concerned with in
simplification—corners, intersections, edges, and so on—are not the
objects that are easy to describe using regions. For this reason, sim-
plification will be an operation on rectilinear polygons, not regions.
It is a post-processing step that occurs only after layout.
An additional concern arises when we introduce rectilinear poly-
gons as another object in the pipeline of rocks layout and presen-
tation. In particular, it will be necessary to phrase operations in
terms of both the region being simplified (which is not always
the topmost, finished region, but perhaps the region due to some
subtree in t), and the layout tree. For example, we might want to
say, “simplify a region, but ensure that its bounds do not exceed
the bounds of its parent.” This is not possible to express using only
one region (we have a region to be simplified, and a region to stay
inside of), and it is not possible to express if we forget the structure
of the tree (since we mention a parent , which can only refer to a
node in the layout tree).

Sam Cohen and Ravi Chugh
…
…A B C D
E F G H I
J K L M
Figure 12: Excerpted steps of the simplification of Abs.
As mentioned in §4.3, pure regions are costly to associate with
nodes in the layout tree. So, a practical implementation of simplifi-
cation is predicated on the use of stateful regions, so that we might
efficiently perform the required annotation.
5.1 An Example
Figure 12 gives an example of simplifying Abs, as rendered by
algorithm L2bS. Simplification is a top-down process which is im-
plemented as a tree traversal over the layout tree.
At every stage in the simplification process, two polygons are
maintained: a keepIn polygon, which describes the boundary that
the region should say inside of, and a keepOut polygon, which
describes an area which is “off limits” for the region being simplified.
In Figure 12, the keepIn polygon is rendered with a thick orange
border, and the keepOut polygon is rendered as a cross-hatched area.
Figure 12
A shows the unsimplified output of layout. Figure 12
B
shows a step of simplification just after the outermost outline has
been fully simplified. This outermost outline serves as the keepIn
polygon for simplification of its child subtree. Figure 12
C shows
the new keepIn polygon, which has been offset by the padding
applied to the outermost outline in the layout. This ensures that
even after simplification, a parent node’s border is unoccupied by
the outlines of its children. Figure 12
D shows the keepOut polygon
for this stage of simplification. The keepOut polygon is derived from
thesibling elements of the term currently being simplified.
Once the keepIn and keepOut polygons are resolved, counter-
clockwise corners and antiknobs are removed from the current
rock. Figure 12
E through
I show the removal of these features.
The finished outline serves as the keepIn polygon of the subtree’s
simplification, as shown in Figure 12
K . The final outlines after
the recursion has been exhausted are shown in Figure 12
M .5.2 Core Operations
The simplification process is a top-down traversal of the layout tree,
but also requires access to the region associated with each node (so
that we might construct a polygonal outline from it). In order to
facilitate the retrival of the relevant region, we can annotate each
node in the layout tree with its corresponding region.
𝑡 ::=Atom(𝑥𝑖)|Wrapid(T,padding)
|JoinV(T,T)|JoinH(T,T)
𝑇 ::=(𝑡,region)
Generally speaking, our goal is to modify the rectilinear polyg-
onal outlines of each region in the layout tree by removing both
anti-knobs and counter-clockwise wound corners (we assume that
the outlines are wound clockwise). However, we cannot perform
these modifications without restriction. In order to maintain that
the tree structure that the layout presents before simplification is
the same as the structure it presents after simplification, we enforce
the following two rules:
(1)The outline of a child node shall not exceed the outline of
its parent, less the padding applied to its parent.
(2)For a given node of the tree, the outlines of the node’s
children may not intersect.
The first rule ensures that simplification does not permit a child
node’s boundary to cross the boundary of its parent. In fact, it is
slightly more restrictive; a child boundary is restricted to a boundary
smaller than that of its parent, to account for the padding applied
to the parent node.
Whereas the first rule is concerned with the parent-child re-
lationship between nodes, the second rule is concerned with the
relationship between siblings. If two siblings do not intersect prior
to simplification, we ensure that they continue to be disjoint af-
ter simplification. This rule is somewhat conservative, as we will
discuss in §5.3.

Ragged Blocks
The following excerpt of simplify sketches a way in which we
can write a simplification routine which respects these two rules.
simplify keepIn(Wrapid(t,padding),region)
=simplify(offsetPolygon(−padding)keepIn)t
simplify keepIn(JoinV(𝑡1, 𝑡2),region)
=(JoinV(simplify ps1𝑡1,simplify ps2𝑡2),region)
where ps1=simplifyPolygon 𝑝1𝑝2keepIn ,
ps2=simplifyPolygon 𝑝2𝑝1keepIn ,
𝑝1=polygonOf(regionOf 𝑡1),and
𝑝2=polygonOf(regionOf 𝑡2)
The simplify operation takes two parameters: a keepIn polygon,
serving as the boundary that child terms’ boundaries ought to stay
inside of, and the term itself. Two of the interesting cases are shown.
When we encounter a Wrap node, the keepIn boundary is offset
to account for the Wrap node’s padding, for child terms should
not encroach on the space allocated for this node’s padding. The
case for JoinV is somewhat more involved. We find unsimplified
polygonal outlines for 𝑡1and𝑡2(𝑝1, and 𝑝2, respectively), and pro-
ceed to simplify them using simplifyPolygon. This function is the
workhorse of the simplification algorithm; it accepts a polygon to
be simplified, a polygon to stay out of, and a polygon to stay inside
of. When simplifying the outline of 𝑡1, for example, the first call
to simplifyPolygon encodes that we ought not simplify so much
that we encroach on our sibling, 𝑡2, or our parent’s outline, keepIn .
The case for JoinH is identical to JoinV , and the case for Atom s is
straightforward, as there’s no simplification to be done.
simplifyPolygon p keepOut keepIn
=𝑝 when 𝑝=𝑝′
=simplifyPolygon 𝑝′keepOut keepIn otherwise
where 𝑝′=simplifyCorner keepOut keepIn 𝑝or,
simplifyAntiKnob keepOut keepIn 𝑝or,
𝑝, if no others apply.
simplifyPolygon calls simplifyCorner and simplifyAntiKnob un-
til a fixpoint. simplifyCorner and simplifyAntiKnob are partial func-
tions which perform the simplifications shown in Figure 13. They
are partial not only because the simplification might not apply
anywhere along the polygon, but because they also check that
the simplification operation would not cause the outline to exceed
keepIn , or intersect keepOut .
5.3 Limitations of Simplification
The definition of simplification relies on two rules; the first of which
concerns constraints of parent-child relationships in the layout tree,
simplifyCorner simplifyAntiKnob
Figure 13: Example polygonal features, and simplifications.
padding=1 padding=2
padding=3 padding=4Figure 14: Simplifications with different padding values.
and the second of which concerns constraints of sibling relation-
ships. In particular, the second rule is important for ensuring that if
the regions of two siblings are disjoint prior to simplification, they
will remain disjoint afterwards. This rule is too conservative, and
Figure 14 demonstrates why. The layouts in Figure 14 all have the
same layout tree. Every fragment in the layout has the same Wrap
node ancestor, so the entire layout could sensibly be represented by
a single region. And, when there is sufficient padding, this is exactly
the result; the entire layout can be wrapped in a single outline rep-
resenting the Wrap . However, when insufficient padding is applied
to the Wrap node (as in the top row of Figure 14), simplification is
unable to merge the regions into one.
Solving this problem in general is challenging. It is difficult to
know when two close, but disjoint, region outlines are compat-
ible. (This information is easy to know if we have a region, but
simplification operates on region boundaries, not regions, and so
we would need to attach some region information to its outline.)
Another challenge is finding a way to construct a bridge between
the disjoint regions, in a way which avoids incompatible regions.
Intuitively, it seems like finding a convex hull might be what
we need. The convex hull of a set of disjoint polygons is, after
all, a well-behaved polygon which tightly wraps the disjoint set.
However, the shape of a rock is decidedly concave, and so some extra
post-processing would be necessary to treat the resulting convex
polygon so that it doesn’t collide with its neighbors. Additionally,
there exists a notion of a rectilinear convex hull [Ottmann et al .
1984]. Unfortunately, although this notion sounds promising, it does
not closely match our visual expectation for well-formed regions.
Future work could address these challenges in order to construct
outlines which are less prone to generating disjoint “islands” when
the padding around a fragment is insufficient to merge with its
compatible neighbors.
6 Benchmarks
The layouts shown in this paper are generated from Rocks , our
implementation of algorithms L1P,L1S,L2a S, and L2bS.Rocks is
implemented in approximately 6k LOC of Haskell. In addition to the
rocks algorithms, our implementation contains Boxes , a boxes-like
layout algorithm which is implemented using the same primitives
as the other algorithms.
To evaluate the compactness of L1Slayouts, we ran our imple-
mentation on six representative source files, and compared the
resulting output with that produced by Boxes . These six source
files (7.6k LOC in total) were collected from popular repositories

Sam Cohen and Ravi Chugh
on GitHub, with the exception of one example (layout), which
was written by the first author. The examples are written in three
programming languages: Python (simplex, functional), TypeScript
(core, diff-objs), and Haskell (solve, layout). The examples also cover
a range of coding styles, from imperative, statement-centric pro-
gramming (simplex, functional, core), to highly nested, expression-
centric programming (diff-objs, solve, layout).
As a baseline for comparison, we implemented two box layout
strategies, Boxes , and Boxes NS. The difference between these al-
gorithms is only that Boxes renders spacers, whereas Boxes NS
ignores them. Figure 15 shows their results on the Abs example
program. In Boxes , since Newline s are local to their subtree, white-
space cannot escape the nearest Wrap node. Most implementations
of boxes layout, therefore, ignore preceding whitespace, and let the
padding applied to each Wrap node convey the nesting.
Figure 16 summarizes our experiments, measuring compactness
in terms of both line- and fragment-based metrics, each of which
we describe next.
6.1 Compactness by Line
The “Mean Line Width” columns in Figure 16 report mean widths
in pixels, as well as relative performance compared to Boxes NS, for
the rendered lines pertaining to that benchmark.
Among the structured layouts, Rocks generates the narrowest
lines for several benchmarks (diff-objs, solve, and layout) but not all;
Boxes NSproduces narrower lines for the remaining benchmarks
(core, functional, and simplex), because it ignores indentation. This
is especially true of examples in which indentation is the primary
source of structure (i.e., in block-based languages such as TypeScript
and Python). In the expression-heavy TypeScript example (diff-objs)
and the Haskell examples, algorithm L1Sproduces shorter lines
on average. ( Boxes is worse than Boxes NS(and L1S) in every case,
which is not surprising given that it does not have the advantage
of ignoring preceding whitespace, and also cannot escape newlines
as demonstrated in Figure 15. Given this sanity check, we do not
report measurements for Boxes in subsequent comparisons.)
The preceding compares the structured layouts to one another.
For evidence that the structured text layouts are not unreasonably
larger than their unstyled, flat counterparts, Figure 16 also reports
mean line width for Unstyled layouts. For each benchmark, the
mean line width for L1Sis less than 20% larger than this baseline.
6.2 Compactness by Fragment
Measuring the width of lines is useful since narrower text layouts
are likely easier for programmers to comprehend. But the number of
BoxesLayout
BoxesNSLayout
Figure 15: Boxes vsBoxes NSlines is generally much larger than fragments, so the latter offer an
opportunity to obtain a more detailed picture about compactness.
We devise a new metric, called mesh distance , which we believe
to be a fair way to judge the amount of stretching orerror between
two text layoutsAandB. This notion has two components; one
which measures the vertical segments, and one which measures the
horizontal. We require layouts AandBto have the same number
of vertical and horizontal segments. Indeed, the mesh distance is
a measure between two layouts which differ only in the position
of their constituent rectangles. The mesh distance is undefined for
layouts which represent different text, or the same text but with
differently chosen line breaks.
Definition: Mesh Distance. Formally, we define the mesh distance
as follows. LetAbe a text layout. In keeping with our definition of
the layout problem in §3, we say that a text layout is represented
by the position of its constituent fragments (we don’t evaluate the
polygonal outlines when computing the mesh distance, so they can
be ignored). So, let [𝑥A
1, . . . , 𝑥A𝑛]∈A be the laid-out fragments of
layoutA.
We can further subdivide the set of fragments in each layout by
their line. We might say that in A, for example:
[𝑥A
1, . . . , 𝑥A
𝑛]=[𝑥A
1, . . . 𝑥A
𝑙1]+···+[ 𝑥A
𝑙𝑚−1+1, . . . 𝑥A
𝑙𝑚]
Here, the layout has 𝑚lines, 𝑥A
1, . . . 𝑥A
𝑙1occuring on the first line,
𝑥A
𝑙1+1, . . . 𝑥A
𝑙2occuring on the second, and so on.
For each fragment, we choose a representative point. We’ll choose
the upper-left corner of each fragment as our canonical point, and
define a function, ul 𝑥to retrive this point. For each line in A, we
construct a set of segments which connect every adjacent pair of
fragments. So, the first line of Ayields 𝑙1−1segments:
AH=[(ul𝑥A
1,ul𝑥A
2), . . . ,(ul𝑥A
𝑙1−1,ul𝑥A
𝑙1)].
We call these the horizontal segments of the layoutA.
In a similar fashion, we can construct the vertical segments of
layoutA. The vertical segments of Aconnect the first fragment
on each line:
AV=[(ul𝑥A
1,ul𝑥A
𝑙1+1), . . . ,(ul𝑥A
𝑙𝑚−2+1,ul𝑥A
𝑙𝑚−1+1)].
Then, the mesh distance of two layouts, AandB, is defined as
follows:
meshDistance HAB=∑︁
𝑖∈|A H|=|BH|lengthBH𝑖−lengthAH𝑖and,
meshDistance VAB=∑︁
𝑖∈|A V|=|BV|lengthBV𝑖−lengthAV𝑖.
Comparing Mesh Distances. For each of the benchmarks in Fig-
ure 16, we calculated the mesh distance between a layout with
no padding applied anywhere (implemented by L1S) and a layout
with uniform padding applied to every node in the input tree. If no
padding is applied anywhere, then L1Sdegrades to ordinary flat
text layout as one might expect from an ordinary text editor. Thus,
it serves as our ground truth. Of course, an effective structured text
layout must necessarily introduce some error from this unstructured
ground truth; the unstructured layout has no space between boxes
by which to show structure. Accordingly, we are not interested in
layouts with zero error, only in minimizing the layout’s error, as

Ragged Blocks
Mean Line Width ( px) Mean Horizontal
Mesh Distance ( px)Mean Vertical
Mesh Distance ( px)LOC
Num Fragments
Unstyled
L1S
Boxes NS
Boxes
L1S
Boxes NS
L1S
Boxes NS
L1SRuntime
core.ts 3020 22k (264, 0.9) (310, 1.1) (290, 1.0) ( 460, 1.6) (4.6, 0.59) (7.8, 1.0) (9.4, 0.51) (18., 1.0) 4.6s
diff-objs.ts 25 0.2k (280, 0.5) (340, 0.6) (560, 1.0) ( 600, 1.1) (5.4, 0.21) (26., 1.0) (22., 0.87) (26., 1.0) 0.2s
functional.py 2233 7.2k (177, 1.0) (210, 1.1) (180, 1.0) ( 350, 2.0) (7.7, 0.36) (21., 1.0) (6.9, 0.69) (10., 1.0) 1.3s
simplex.py 339 1.2k (180, 1.3) (220, 1.6) (140, 1.0) ( 380, 2.7) (7.5, 0.95) (7.9, 1.0) (9.3, 0.83) (11., 1.0) 0.4s
solve.hs 1736 8.4k (305, 0.3) (360, 0.4) (960, 1.0) (1300, 1.3) (6.3, 0.07) (87., 1.0) (14., 0.95) (14., 1.0) 2.2s
layout.hs 285 2.2k (241, 0.6) (280, 0.7) (410, 1.0) ( 480, 1.2) (4.0, 0.19) (21., 1.0) (17., 0.90) (18., 1.0) 0.4s
Figure 16: Benchmarks
Boxes
Error(Pixels)L1SBoxesNSDistributionofErrorinHorizontalSegmentsofdiff-objsRocks-L1S
NumberofHorizontal
Segments
BA160
140
120
100
80
60
40
20
00255075100150200250300350400450500
Figure 17: Boxes-everywhere layouts for the diff-objs benchmark.
measured by the mesh distance. The mesh distances in Figure 16
are normalized to Boxes NS, since Boxes NSis the “state of the art”
in structured text visualization.
Figure 16 shows that algorithm L1Sdoes indeed produce more
compact layouts than our Boxes NSreference implementation. In
all benchmarks, and for both horizontal and vertical mesh distance,
L1Sproduces the most compact layouts, but the degree of com-
pactness as compared to the Boxes NSimplementation varies. In
particular, we notice that for examples and langauges which are
more expression-oriented, L1Sproduces significantly more compact
layouts, whereas the difference is not as pronounced for examples
which demonstrate a more statement-oriented programming style.
Case Study: diff-objs. In those examples which use an expression-
oriented style, it is often the case that in deeply nested expressions,
theBoxes NSlayout places its output rectangles very far from the
reference. One such example is demonstrated in Figure 17, which
shows a fragment of the rendered output of the diff-objs benchmark
using the Boxes NSandL1Salgorithms.For this program, both layouts contain many horizontal segments
with small error, but Boxes NScontains a few horizontal segments
with very large error. Compare, for example, the error due to the
fragment useWith , which is small in L1S(see Figure 17
A ), but large
inBoxes NS(see Figure 17
B ). Since the entire subexpression of a
call is required to fit inside a box in Boxes NS, function arguments
which span multiple lines are often pushed much further rightward
than in the unformatted text. Algorithm L1Savoids this problem by
permitting the boundary of a subexpression to flow into the shape
of an arbitrary polygon, and allowing newlines to “cut” through
the entire layout tree.
Of course, algorithm L1S’s more flexible outlines come with
tradeoffs. The layout produced by L1Sis arguably harder to read
since the more complex outlines are harder to parse. Furthermore,
it is possible for a subexpression to be represented as two or more
dijoint polygons in L1S(see the application of the function pipe in
Figure 17, for example), a situation that cannot occur in Boxes NS
(see also Limitations of Simplification).

Sam Cohen and Ravi Chugh
A
Code Diffing. A code editor 
visualizes the edits necessary to 
transform fibfrom using 
structural recursion to using tail 
recursion with two accumulators. Type Error. A code editor visualizes two nested errors: a failure 
to unify the type of out[i](A) and as[i](B), and the fact that 
the value of outis used before it is initialized.
B
Figure 18: Example code visualizations.
Prose Diffing. A word processor shows two distinct 
edits: a find-and-replace action, and the addition of a 
new paragraph.
Dependency Error. A word processor identifies that 
a term has been used before its definition.
Nested Errors. A word processor visualizes three 
nested errors: A spelling error, a grammatical error, 
and a suggestion to merge two sentences into one.A
B
C
Figure 19: Example prose visualizations.6.3 Smoothness
The experiments reported in Figure 16 do not invoke simplification,
and we don’t report the performance impact of this post-processing
step. In addition to performance impact, it could be interesting to
count the number of corners reduced by simplification, but this is
a task left for future work.
Regardless, judging the quality and preferences of the polygonal
outlines resists quantitative treatment as we’ve done in evaluating
compactness. The various examples given throughout the paper,
especially in Figure 18 and Figure 19, indicate the nature and variety
of polygonal outlines generated by algorithms L1S,L2a S, and L2bS,
with and without simplification. But ultimately, these will need to
be evaluated by users; an important direction for future work.
7 Applications
In the previous section, we evaluated our layout approach when
systematically rendering all nested substructures uniformly with
border and padding. Next, we present an example gallery with more
domain-specific, fine-grained text visualizations. These serve as
more detailed demonstrations of the text-rendering capabilities we
have developed in Rocks , and are suggestive of how ragged blocks
might be incorporated into future GUIs.
7.1 Code
Figure 18 demonstrates how structured text could be applied to the
task of visualizing code errors and diffs.
Visualizing Errors in Code. The code fragment in Figure 18
A
shows an example of nested errors in TypeScript. In the example,
the programmer forgot to apply the function fto the elements of
the input list as, causing outto be assigned to a value of the wrong
type. Additionally, the programmer failed to initialize out, and that
error is alsosurfaced at the point of assignment.
A structured text visualization can show overlapping error con-
texts, such as the context for the type error which is indicated with
orange underlines, as opposed to the context for the initialization
error, which is shown with red underlines. The green and orange
boxes mark the two expressions with incompatible types.
Code Diffs. In example Figure 18
B , a Python definition of a fib
function written in “natural” recursive style is transformed into a
tail-recursive version which uses two accumulator parameters. The

Ragged Blocks
original definition is wrapped in a new function (the outermost
green box), new parameters are added, and the inner definition
offibis renamed to go(purple box). Structured diffs, like that
shown, are an alternative to traditional line-based diffs. Because a
structural diff is aware of the tree underlying the source program,
it can render a change as a sequence of more meaningful semantic
changes to the program, such as the wrapping of one function in
another.
7.2 Prose
Analogous to the previous examples, Figure 19 includes examples of
visualizing errors and differences in natural-language documents.
Visualizing Errors in Prose. In the face of an error depicted in
Figure 19
A , the editor provides suggestions for how the user might
modify the text. But instead of presenting them individually, an
editor endowed with a structured text visualization can present all
options simultaneously, so that the user can view and apply them
in any order they see fit.
Grammar and spelling checkers are ubiquitous in writing tools,
but we might also imagine a different, semantic analysis which
ensures that terms are not used before their definition. Figure 19
B
shows a visualization for this hypothetical analysis, which shows a
term, in purple, that is used before its definition, and its definition
which is marked in gray.
Prose Diffs. Figure 19
C shows two unrelated, but overlapping
modifications to the text. Nesting permits the visualization to imply
the ordering of the edits, despite the fact that they overlap.
8 Conclusion and Future Work
In this paper, we identified arbitrary rectilinear polygons—ragged
blocks, or rocks—as a building block for rendering nested text vi-
sualizations, and detailed a family of algorithms for computing
compact rock layouts. On a set of benchmark source code files, we
showed that the layouts produced by our implementation, Rocks ,
are significantly more compact than traditional layout techniques
involving only rectangles, or boxes. This result is a step towards
more seamlessly integrating text-based and structure-based edit-
ing into IDEs for programming. Through a small but suggestive
set of example visualizations involving code and prose, we have
identified opportunities for future code and prose editors to inte-
grate structured text visualizations as a channel for communicating
information to users.
There are many natural ways to build on the techniques in this
paper. One is to continue optimizing the implementation of the lay-
out algorithms, to make them more practical for large files. Another
is to investigate avenues for additional simplification, as discussed.
Furthermore, while the algorithms themselves are generally lan-
guage agnostic (which is how we could easily evaluate on bench-
marks written in different languages), a more full-featured editor
for a specific language will require fine-tuned pre-processing to
convert syntax trees in that language to appropriately configured
layout trees. For example, the general front-end parser we currently
use in Rocks generates Haskell syntax trees with unnecessary—and
sometimes incorrect—parse trees; these would need to be refined
before subsequent fine-grained styling and rendering.On a related note, while algorithm L2bSsupports the insertion of
pins to enforce column constraints, it may be impractical to expect
users to manually write them. Instead, general language-specific
conventions could be used to inform default pin constraints—for
example, definitions within a particular block could be implicitly
constrained for alignment, whereas definitions across top-level
blocks would not.
Another direction for future work is to reexamine existing text
visualizations in light of the new layout capabilities, and to imagine
new ones that they enable. Compared to the well-studied realm
of program visualizations, the space of non-code text visualiza-
tions is rather less explored. Perhaps this is because prose is not
as highly structured as code, and because users do not expect the
same degree of richness from “common” tools such as word proces-
sors and web browsers as from coding tools. But we also believe
the lack of expressive layout techniques has been a limiting factor
for experimentation in user interfaces for text-heavy productivity
tools.
Ultimately, the motivation for this work is to enable new text
visualizations that users find helpful as they read and write a variety
of documents. So, in addition to the quantitative benchmarking and
design exploration we have pursued to evaluate our techniques so
far, it will be crucial to understand how different tradeoffs—between
compactness, smoothness, and the use of various decorations—are
perceived by users in different usage scenarios.
References
Leif Andersen, Michael Ballantyne, and Matthias Felleisen. 2020. Adding Interactive
Visual Syntax to Textual Code. Object Oriented Programming Systems Languages &
Applications OOPSLA (2020). https://doi.org/10.1145/3428290
Tom Beckmann, Stefan Ramson, Patrick Rein, and Robert Hirschfeld. 2020. Visual
Design for a Tree-Oriented Projectional Editor. In International Conference on the Art,
Science, and Engineering of Programming . https://doi.org/10.1145/3397537.3397560
Tom Beckmann, Patrick Rein, Stefan Ramson, Joana Bergsiek, and Robert Hirschfeld.
2023. Structured Editing for All: Deriving Usable Structured Editors from Grammars.
InConference on Human Factors in Computing Systems (CHI) . https://doi.org/10.
1145/3544548.3580785
Andrew Blinn. 2019. Fructure: A Structured Editing Engine in Racket. RacketCon.
https://www.youtube.com/watch?v=CnbVCNIh1NA
Sam Cohen and Ravi Chugh. 2025. Code Style Sheets: CSS for Code. Proceedings
of the ACM on Programming Languages (PACMPL), Issue OOPSLA1 (2025). https:
//doi.org/10.1145/3720421
Martin Erwig and Bernd Meyer. 1995. Heterogeneous Visual Languages: Integrating
Visual and Textual Programming. Symposium on Visual Languages (VL) (1995).
https://api.semanticscholar.org/CorpusID:17004705
Han L. Han, Miguel A. Renom, Wendy E. Mackay, and Michel Beaudouin-Lafon. 2020.
Textlets: Supporting Constraints and Consistency in Text Documents. In Conference
on Human Factors in Computing Systems (CHI) . https://doi.org/10.1145/3313831.
3376804
Brian Harvey, Daniel D. Garcia, Tiffany Barnes, Nathaniel Titterton, Daniel Armendariz,
Luke Segars, Eugene Lemon, Sean Morris, and Josh Paley. 2013. SNAP! (Build Your
Own Blocks). In Technical Symposium on Computer Science Education (SIGCSE TS) .
https://doi.org/10.1145/2445196.2445507
Brian Hempel, Justin Lubin, Grace Lu, and Ravi Chugh. 2018. Deuce: A Lightweight
User Interface for Structured Editing. In International Conference on Software Engi-
neering (ICSE) . https://doi.org/10.1145/3180155.3180165
JetBrains. 2011–2024. MPS (Meta Programming System). https://en.wikipedia.org/
wiki/JetBrains_MPS
Donald E. Knuth and Michael F. Plass. 1981. Breaking paragraphs into lines. Software:
Practice and Experience (1981). https://doi.org/10.1002/spe.4380111102
Amy J. Ko and Brad A. Myers. 2006. Barista: An Implementation Framework for En-
abling New Tools, Interaction Techniques and Views in Code Editors. In Conference
on Human Factors in Computing Systems (CHI) . https://doi.org/10.1145/1124772.
1124831
Mikaël Mayer, Gustavo Soares, Maxim Grechkin, Vu Le, Mark Marron, Oleksandr
Polozov, Rishabh Singh, Benjamin Zorn, and Sumit Gulwani. 2015. User Interaction
Models for Disambiguation in Programming by Example. In Symposium on User
Interface Software & Technology (UIST) . https://doi.org/10.1145/2807442.2807459

Sam Cohen and Ravi Chugh
Robert C. Miller and Brad A. Myers. 2002. Multiple Selections in Smart Text Editing.
InInternational Conference on Intelligent User Interfaces (IUI) . https://doi.org/10.
1145/502716.502734
David Moon, Andrew Blinn, and Cyrus Omar. 2023. Gradual Structure Editing with
Obligations. In Symposium on Visual Languages and Human-Centric Computing
(VL/HCC) . https://doi.org/10.1109/VL-HCC57772.2023.00016
Mozilla. 2025. Learn CSS: The Box Model. https://developer.mozilla.org/en-US/docs/
Learn/CSS/Building_blocks/The_box_model;https://developer.mozilla.org/en-
US/docs/Web/CSS/CSS_box_model/Introduction_to_the_CSS_box_modelhttps:
//developer.mozilla.org/en-US/docs/Learn_web_development/Core/Styling_
basics/Box_model
National Instruments. 2024. Laboratory Virtual Instrument Engineering Workbench
(LabVIEW). https://www.ni.com/labview/
Wode Ni, Joshua Sunshine, Vu Le, Sumit Gulwani, and Titus Barik. 2021. reCode: A
Lightweight Find-and-Replace Interaction in the IDE for Transforming Code by
Example. In Symposium on User Interface Software and Technology (UIST) . https:
//doi.org/10.1145/3472749.3474748
Cyrus Omar, David Moon, Andrew Blinn, Ian Voysey, Nick Collins, and Ravi Chugh.
2021. Filling Typed Holes with Live GUIs. In Programming Language Design and
Implementation (PLDI) . https://doi.org/10.1145/3453483.3454059
Cyrus Omar, YoungSeok Yoon, Thomas D. LaToza, and Brad A. Myers. 2012. Active
Code Completion. In International Conference on Software Engineering (ICSE) .
Thomas Ottmann, Eljas Soisalon-Soininen, and Derick Wood. 1984. On the definition
and computation of rectilinear convex hulls. Information Sciences (1984). https:
//doi.org/10.1016/0020-0255(84)90025-2
Josh Pollock, Catherine Mei, Grace Huang, Elliot Evans, Daniel Jackson, and Arvind
Satyanarayan. 2024. Bluefish: Composing Diagrams with Declarative Relations. In
Symposium on User Interface Software and Technology (UIST) . https://doi.org/10.
1145/3654777.3676465
Mitchel Resnick, John Maloney, Andrés Monroy-Hernández, Natalie Rusk, Evelyn East-
mond, Karen Brennan, Amon Millner, Eric Rosenbaum, Jay Silver, Brian Silverman,
et al.2009. Scratch: Programming for All. Communications of the ACM (CACM)
(2009). https://doi.org/10.1145/1592761.1592779
Zhiyuan Wu, Jiening Li, Kevin Ma, Hita Kambhamettu, and Andrew Head. 2023. FFL:
A Language and Live Runtime for Styling and Labeling Typeset Math Formulas. In
Symposium on User Interface Software and Technology (UIST) . https://doi.org/10.
1145/3586183.3606731
Litao Yan, Alyssa Hwang, Zhiyuan Wu, and Andrew Head. 2024. Ivie: Lightweight
Anchored Explanations of Just-Generated Code. In Conference on Human Factors in
Computing Systems (CHI) . https://doi.org/10.1145/3613904.3642239
Katherine Ye, Wode Ni, Max Krieger, Dor Ma’ayan, Jenna Wise, Jonathan Aldrich,
Joshua Sunshine, and Keenan Crane. 2020. Penrose: From Mathematical Notation
to Beautiful Diagrams. ACM Transactions on Graphics (2020). https://doi.org/10.
1145/3386569.3392375
Ryan Yen, Jiawen Zhu, Sangho Suh, Haijun Xia, and Jian Zhao. 2024. CoLadder:
Manipulating Code Generation via Multi-Level Blocks. In Symposium on User
Interface Software and Technology (UIST) . https://doi.org/10.1145/3654777.3676357
Kuat Yessenov, Shubham Tulsiani, Aditya Menon, Robert C. Miller, Sumit Gulwani,
Butler Lampson, and Adam Kalai. 2013. A Colorful Approach to Text Processing
by Example. In Symposium on User Interface Software and Technology (UIST) . https:
//doi.org/10.1145/2501988.2502040

Ragged Blocks
A Layout with Column Constraints
In this section, we will outline modest changes to algorithm L1P
which will permit columns to remain aligned after layout. The key
insight is this: in L1P, given two regions, call them 𝑎and𝑏,joinH
placed 𝑏directly after 𝑎, translating it by 𝑎’s advance. Instead of
translating immediately, we could impose a constraint on𝑏, permit-
ting its horizontal position to be no less than the position of 𝑎plus
𝑎’s advance. We might assign 𝑏’s horizontal position to a variable,
say𝑝0. Then, using 𝑝0to describe the horizontal position of frag-
ments in other lines has the effect of constraining those fragments
to be horizontally aligned with 𝑏.
In order to model the addition of constrained rectangles, we
introduce a new kind of term into our grammar of layout trees:
𝑡 ::=Atom(𝑥𝑖)|Pinpid(𝑥𝑖)|Wrapid(𝑡,padding)
|JoinV(𝑡, 𝑡)|JoinH(𝑡, 𝑡)
layout ::=[[(stack ,pid,v)]]
APinis a fragment, 𝑥𝑖, with an identifier, pid, which is a name
uniquely identifying the constraint variable associated with the Pin.
We encode that two rectangles are constrained to the same horizon-
tal position by wrapping them in the Pinconstructor, and giving
them the same pid, thus ensuring that their horizontal positions
are determined by the same variable.
The type of layouts has also changed. In algorithm L1P, lines were
the atomic units with which layout was concerned—it was never
necessary to peek inside a line—only to take the union of partial
lines, or translate lines. Not so for algorithm L2a P. The horizontal
position of each individual Stack in the layout is determined by the
result of constraint resolution, and so it is necessary to be able to
iterate through these Stack s one-by-one.5This is reflected in the
type of an L2a Playout, which is a nested list of Stack s, annotated
with the pidof the variable which determines the Stack ’s horizontal
position, and the nominal advance of the Stack , which serves the
same role as the advance in algorithm L1P.
Figure 20 gives an example of the constraints we might construct
for the Abs program. Each constraint takes the form of a linear
inequality, where the first fragment on each line is constrained to
be on the left margin. A stack 𝑠𝑖isat least as far right as the stack
to its left ( 𝑠𝑖−1), plus the advance between 𝑠𝑖−1and𝑠𝑖. By giving
an objective function which attempts to minimize the sum of the
𝑝𝑖s, we may phrase this problem as a linear program, and solve it
automatically during layout.
Once the horizontal position of each Stack has been determined,
each line is merged into a single region using leading𝑅just as in
algorithm L1P. The result of using algorithm L2a Pinstead of algo-
rithm L1Pon the Absprogram yields the result shown in Figure 21.
A single constraint has been introduced, causing the Stack s corre-
sponding to ?and:to be aligned in the finished layout.
5In practice, it is usually the case that only a few Stack s in the layout are constrained,
while the others are laid out exactly like in algorithm L1P. It is possible, then, to assign
a variable to each contiguous group of unconstrained stacks, rather than each stack.
This reduces the number of variables in the constraint problem, and has no effect on
the resulting layout.
s13s14s15s16p12p13p14p15p1
s1s2s3s4s5s6p2p3p4p5p6
s7s8s9s10s11s12p7p8p9p3p10p11𝑝1≥ 0
𝑝2≥𝑝1+advance 𝑠1𝑠2
. . .
𝑝6≥𝑝5+advance 𝑠5𝑠6
𝑝7≥ 0
𝑝8≥𝑝7+advance 𝑠7𝑠8
. . .
𝑝11≥𝑝10+advance 𝑠11𝑠12
𝑝12≥ 0
𝑝13≥𝑝12+advance 𝑠13𝑠14
. . .
𝑝15≥𝑝14+advance 𝑠15𝑠16
Minimizing15∑︁
𝑖=1𝑝𝑖
Figure 20: Example constraint problem.
Figure 21: Example 2aPlayout.

Sam Cohen and Ravi Chugh
B Layout with Automatic Line Breaks
In algorithm L2a P, we showed that adjacent stacks need not be
placed tightly against one another. Indeed, we can use advance𝑅as
a kind of minimum bound on the spacing between two stacks. This
minimum bound information is enough to implement a variety
of text layout algorithms, and not only those which are suited
for layout of pre-formatted text. Algorithm L2bP, as opposed to
algorithms L1PandL2a P, takes unformatted text as input. That is,
text which does not contain explicit newlines. For this reason, we
will refer to the linear stream of Stack s that L2bPtakes as input as
the “tape. ”6This is reflected in the new grammar of layout trees for
L2bP, which dispenses with the JoinV constructor for concatenating
two text layouts vertically:
𝑡 ::=Atom(𝑥𝑖)|Wrapid(𝑡,padding)|JoinH(𝑡, 𝑡)
layout ::=[[(stack ,v)]]
Thus, algorithm L2bP’s job is twofold: it must find good locations
for line breaks, and then having inserted those breaks, it must find
the positions of each rectangle, 𝑥𝑖, in the fashion of algorithms
L1PandL2a P. In order to accomplish the former goal, we have
re-implemented a subset of the algorithm given by Knuth and Plass,
modifying it to use Stack s in place of rectangles, thus allowing our
implementation to support structured text.
Knuth and Plass’ algorithm works by iterating once over the
input tape, adding and removing items from a set, called the feasible
set, describing the set of possible line breaks in the layout. On each
element in the tape (the current point), a new line break can be
added to the set if it is deemed feasible (there exists another line
break in the feasible set which, when paired with the current point,
would make a feasible line), and some breaks can be removed from
the set if they are no longer feasible (there does not exist a line break
in the feasible set which could form a valid line with the current
point). Each line break in the feasible set remembers the history of
line breaks before them, and the cost of this history trace, according
to an objective function. Once the entire tape has been traversed,
the trace with the lowest cost is selected, and the text is broken
according to the breaks in the optimal trace.
6Sometimes called “idiot tape:” a stream of text for layout with no additional instruc-
tions on how it should be laid out.
snsn+1sn+2sn+3sn+4sn+5sn+6sn+7
sn+8 leading snsn+1leading sn+3 sn+4padding sn+7
lineWidth snsn+7sn-1
Figure 22: Example input tape.The advance (equivalent to width in the unstructured algorithm)
of an atomic fragment of text is considered only when determin-
ing if a subset of the input tape consistitutes a feasible line. This
feasibility test compares the “ideal” line width (an input to the al-
gorithm) to the line’s actual width, and calculates a cost based on
the difference between these quantities. Therefore, in order to mod-
ify the algorithm to support structured text, we need only update
the function which calculates the length of a line to consider the
presence of padding between fragments.
We define the following function, lineWidth, which calculates
the length of line beginning at stack 𝑠𝑖and ending at stack 𝑠𝑗.
lineWidth 𝑠𝑖𝑠𝑗=Σ𝑝𝑠𝑖+Σ𝑝𝑠𝑖𝑗+Σ𝑎
where
Stack(Σ𝑝𝑠𝑖,cells𝑠𝑖)=𝑠𝑖,and
Stack(Σ𝑝𝑠𝑗,cells𝑠𝑗)=𝑠𝑗,and
Σ𝑎=∑︁
(𝑎,𝑏)∈Padvance 𝑎 𝑏
where
P=adjacent pairs,(𝑎, 𝑏), in𝑠𝑖, 𝑠𝑖+1, . . . , 𝑠𝑗
Note that special care is taken to include allof the padding
surrounding the first and last Stack s on the candidate line, since
these Stack s cannot possibly engage in space sharing with their left
and right neighbors, respectively. (For example, in Figure 22, Stack
𝑠𝑛cannot share space with Stack 𝑠𝑛−1, and Stack 𝑠𝑛+7cannot share
space with Stack 𝑠𝑛+8, despite the fact that they are both wrapped
by the same Node7).
Once the location of line breaks are known, the input layout tree
may be modified by judiciously inserting vertical concatenation
constructors ( JoinV ) according the results of the search. Then, algo-
rithm L1P, for example, could be used to find the final position of
each rectangle. In our implementation, a separate but similar algo-
rithm is used which fully justifies theStack s on each line, evenly
spacing them so the the leftmost Stack on the line abuts the left
margin, and the rightmost Stack abuts the right margin. (See the
examples in Figure 19, for example.)
7Note, however, that these Stack scould (and will) participate in space sharing as their
lines are merged into a single region.