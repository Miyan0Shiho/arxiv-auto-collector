# Back to the Future: The Evolving Universal Relation Model

**Authors**: Nan Tang, Haixun Wang

**Published**: 2026-10-27 08:30:49

**PDF URL**: [https://arxiv.org/pdf/2610.36833v1](https://arxiv.org/pdf/2610.36833v1)

## Abstract
Document backed analytics need databases that can be built as queries arrive. Enterprises increasingly ask analytical queries over messy document collections such as contracts, invoices, reports, emails, policies, and tickets, where the relevant structure and facts are often not known in advance. RAG can retrieve and summarize supporting evidence, but it does not by itself produce reusable, joinable, governed database state. Schema first construction offers the opposite strength: once a database is built, it supports joins, aggregation, constraints, indexes, optimization, and transactions. Yet for document analytics, fixing the schema and extracting all possible fields upfront is both expensive and brittle, because later queries and new documents often surface concepts that were not modeled before. We propose the Evolving Universal Relation Model (EUR), a query first model for document backed analytics. Users ask natural language queries as if the document corpus exposed one conceptual relation over application concepts. Internally, the DBMS turns each query into controlled database evolution: it updates the logical schema when new concepts are needed, extracts and integrates the facts required to answer the query, and decides what durable database state should remain for future queries. Thus, in EUR, a query is not only evaluated; it also helps construct the database. Document backed analytics becomes query first universal access plus workload-driven database schema and instance evolution.

## Full Text


<!-- PDF content starts -->

Back to the Future: The Evolving Universal Relation Model
— Query First, Database Emerges —
Nan Tang
HKUST (GZ), China
nantang@hkust-gz.edu.cnHaixun Wang
Evenup, USA
haixun@gmail.com
ABSTRACT
Document-backed analytics need databases that can be built as queries
arrive.Enterprises increasingly ask analytical queries over messy
document collections—contracts, invoices, reports, emails, policies,
and tickets—where the relevant structure and facts are often not
known in advance. RAG can retrieve and summarize supporting
evidence, but it does not by itself produce reusable, joinable, gov-
erned database state. Schema-first construction offers the opposite
strength: once a database is built, it supports joins, aggregation,
constraints, indexes, optimization, and transactions. Yet for docu-
ment analytics, fixing the schema and extracting all possible fields
upfront is both expensive and brittle, because later queries and
new documents often surface concepts that were not modeled be-
fore. We propose theEvolving Universal Relation Model(EUR), a
query-first model for document-backed analytics. Users ask natural
language queries as if the document corpus exposed one concep-
tual relation over application concepts. Internally, the DBMS turns
each query into controlled database evolution: it updates the logical
schema when new concepts are needed, extracts and integrates
the facts required to answer the query, and decides what durable
database state should remain for future queries. Thus, in EUR, a
query is not only evaluated; it also helps construct the database.
Document-backed analytics becomes query-first universal access
plus workload-driven database schema and instance evolution.
1 INTRODUCTION
The next enterprise database starts as documents.Traditional
databases start with schemas; document-backed analytics start with
queries. In compliance reporting, contract tracking, policy monitor-
ing, and similar settings, users ask over contracts, invoices, reports,
emails, policies, and tickets before the right schema is known. These
queries are analytical, not merely conversational: they require joins,
aggregations, comparisons, and trends. Yet the needed concepts
and facts are scattered across messy documents and often become
clear only as new queries arrive.
RAG only reads; analytics need databases.Retrieval-augmented
generation (RAG) makes document collections searchable and con-
versational, but it does not provide a database contract. It retrieves
evidence into a prompt and produces a response; it does not, by
itself, leave behind stable identifiers, typed tuples, relationships,
constraints, versions, indexes, or governed provenance. Analytical
queries should not repeatedly reconstruct the same facts from text
snippets. They need reusable database that can be joined, aggre-
gated, checked, maintained, and optimized.
Schema-first construction freezes the design too early.The natu-
ral counter-move is to build the database upfront: declare entities
and relationships, design tables, define mappings, and run extrac-
tors. This restores familiar DBMS benefits, but only if the schema,mappings, and required facts are already known. Classical mod-
els provide the right ideals, but under a schema-first assumption:
Entity-Relationship (ER) [ 4] model captures entities, relationships,
attributes, and constraints, while the relational model [ 5] provides
declarative access and data independence over a declared schema.
What is missing is a contract that preserves these ideals while the
database is still being discovered from documents and queries.
Research gap: database construction should become query-first.
Document-backed analytics need a stable query interface over ap-
plication concepts, while the DBMS is still deciding what tables
exist, where attributes belong, which facts to extract, and what
durable state to keep. The challenge is not better retrieval alone,
nor better upfront extraction alone. It is query-driven database
construction: each query may reveal new concepts, require new
facts, and improve the reusable database state for future queries.
Pay-as-you-go dataspaces already challenged the need for com-
plete semantic integration before useful access [ 7,10]. The question
we target is therefore not whether construction can be incremental,
but how a DBMS should govern it: when a query requires concepts
or facts that are not yet represented, how should the query-facing
vocabulary, logical schema, and materialized instance evolve to-
gether while preserving logical path independence? Document-
backed analytics needs a contract that makes these changes con-
trolled effects of query processing, so that answering a query also
builds reusable database state.
Back to the future: The EUR model.We propose theEvolving
Universal Relation Model(EUR, pronounced “your”): a query-first
revival of the universal relation model (URM) [ 14] for document-
backed analytics. Users pose queries as if the document collection
exposed a single evolving universal relation over application con-
cepts. Internally, the DBMS maintains three coupled levels: an evolv-
ing universal relation that defines the user-facing query interface; a
logical schema that organizes concepts into entity and relationship
tables; and a materialized instance containing extracted, integrated
facts. Each query both returns an answer and triggers a controlled
database evolution. Unlike schema-on-read, this interpretation is
retained as durable database state; unlike schema-first ETL, the
design evolves as the workload reveals what is needed.
2 A MOTIVATING EXAMPLE
Next, we illustrate EUR with a small course-analytics example. The
key point is that a query does not directly create tables. It first
reveals application-level vocabulary:attribute rolesandsemantic
connections. EUR merges these revealed roles and connections into
theevolving universal relation. The DBMS then derives a logical
schema by decomposing this relation according to the semantic
connections and (available) functional dependencies. Finally, the
1
arXiv:2610.36833v1  [cs.DB]  27 Oct 2026

Nan Tang and Haixun Wang
system extracts the document-backed facts needed for the current
queries and materializes them as reusable database state.
Consider an example about course analytics over a document
collection𝐹. Suppose𝐹contains the following document snippet.
Text snippet:In 2026, the CS department in the School of Engineering
offersCS101, titledIntroduction to AI, andCS201, titledDatabase Systems.
CS101has 3 credits, andCS201has 4 credits. Alice Sara is a CS major and
takes bothCS101andCS201.
A workload with two queries arrive over time:
𝑄1: Which CS courses are offered in 2026 by the School of Engineering?
𝑄2: Which of these courses does Alice Sara, a CS major, take, and how
many credits is each course?
For simplicity, let’s assume that initially, the database is empty.
2.1 First Query: Revealing the Course World
Query𝑄1asks for courses, but it also mentions the conditions under
which those courses should be selected: department, school, and
year. The system therefore begins by identifying the attribute roles
and semantic connections needed to interpret the query.
Attribute roles and semantic connections.The query reveals
fiveattribute roles:
𝐶(course):CourseCode
𝑇(title):CourseTitle
𝐷(department):DepartmentName
𝑆(school):SchoolName
𝑌(year):OfferingYear
The query also reveals semantic connections among these roles,
represented aspredicates𝑃over subsets of the attribute roles:
𝑃course(𝐶,𝑇):course𝐶has title𝑇,
𝑃department(𝐷,𝑆):department𝐷belongs to school𝑆,
𝑃offering(𝐷,𝐶,𝑌):department𝐷offers course𝐶in year𝑌.
Universal relation.The revealed roles and semantic connections
are merged into the first query-facinguniversal relation 𝑈1w.r.t.
𝑄1[14].𝑈 1is a conceptual relation, not a stored wide table:
𝑈1(𝐶,𝑇,𝐷,𝑆,𝑌).
𝑈1={(𝑐,𝑡,𝑑,𝑠,𝑦)|𝑃 course(𝑐,𝑡)∧𝑃 department(𝑑,𝑠)∧𝑃 offering(𝑑,𝑐,𝑦)}.
It lets the query refer to (𝐶,𝑇,𝐷,𝑆,𝑌) without exposing how
the DBMS will later organize these roles into tables. Under 𝑈1, the
query is interpreted below, which is independent of any particular
table decomposition:
𝑄1(𝑈1)={(𝑐,𝑡)|(𝑐,𝑡,𝑑,𝑠,𝑦)∈𝑈 1∧𝑑=“CS”
∧𝑠=“Engineering”∧𝑦=2026}.
The query has revealed the application vocabulary needed to
answer it. At this stage, the system knowswhatconcepts exist, but
not yethowthey should be organized into relational tables.
Logical schema.After 𝑈1is established, the DBMS chooses a logi-
cal schema underneath it. The design is not obtained by creating
one table per attribute role. Instead, the DBMS decomposes the uni-
versal relation according to the semantic connections and (available)
functional dependencies as:
<latexit sha1_base64="XHtObO81c3wp+8OFdUxO/LO/a50=">AAACw3icbVFdb9MwFHUCjJHB6OCRF4uKiacqGdPYY6Egsbehrdukpqoc56a16tiRfTOtCvmTvMGvwW4rtK67lqWjc++5n1klhcU4/hOET54+23m++yLae/lq/3Xn4M2V1bXhMORaanOTMQtSKBiiQAk3lQFWZhKus/nA+69vwVih1SUuKhiXbKpEIThDR006f9MMpkI1yLJaMtM2v+TytVE68zmjFOEOs6IZuIoWBjqHlh7STfbS192mBwZygW2abuSyzn2RxImPP1NodF5z3wtFTb+cOfLTY4KjleAbc426cenFwiKU1lHH98JB5f8niSadbtyLl0a3QbIGXbK280nnd5prXpegkEtm7SiJKxw3zKDgbr4orS1UjM/ZFEYOKlaCHTfLG7T0g2NyWmjjvkK6ZO8rGlZauygzF1kynNmHPk8+5hvVWJyOG6GqGkHxVaGiln5d/qA0FwY4yoUDjBvheqV8xgzj6M7ul5A8HHkbXB31kpPeyc/jbv/reh275B15Tz6ShHwmffKDnJMh4UE/KAIdVOH3cB6aEFehYbDWvCUbFrb/AKGf2wU=</latexit>CourseCodeCourseTitleCourseCreditCS101Introduction to AI3CS201Database Systems4
<latexit sha1_base64="ZW6Xgp7qMPz0jS6A/Uszv2HCxcY=">AAACoHicbVFdb9MwFHXC1whfBR4RkrUKxFOVTNPgcVAeNgnEpq3bRFNVN85Na82xI/sGUYX8Lv4Hb/wbnLZC67ZrWTo651zfD2eVko7i+G8Q3rl77/6DrYfRo8dPnj7rPX9x5kxtBY6EUcZeZOBQSY0jkqTworIIZabwPLscdvr5D7ROGn1KiwonJcy0LKQA8tS09zvNcCZ1Q5DVCmzb/FL+tFE6716MUsKflBXN0NdzODQ5tvwt32RPu6ptmm7kOK+eJHHS2Q81WZPXoqvIyfCPh7eZd1bmz+Bb8QPxk4UjLN0VK+r8f5/TXj8exMvgN0GyBn22jqNp70+aG1GXqEkocG6cxBVNGrAkhW8/SmuHFYhLmOHYQw0lukmzXHDL33gm54Wx/mriS/ZqRgOlc4sy884SaO6uax15mzauqfgwaaSuakItVoWKWnVb6n6L59KiILXwAISVvlcu5mBBkP/TyC8huT7yTXC2M0j2BnvHu/39T+t1bLFXbJu9Ywl7z/bZATtiIyaC18Ew+BJ8DbfDg/BbeLyyhsE65yXbiPD7Py8czm8=</latexit>CourseCodeCourseTitleCS101Introduction to AICS201Database Systems
<latexit sha1_base64="Ybc34efdyF60Fml7aidtLL3l3lI=">AAACb3icbVFNT9tAEF27lNK00EAPPYDQqlERXCK7qqBHBK3UEwKl+ZDiKFpvxsmK9a61O0ZEJtf+wN76H3rhH7AOAeWDWVl6evNm5+1znElhMQj+ef6rtdfrbzbeVt6939z6UN3eaVmdGw5NrqU2nZhZkEJBEwVK6GQGWBpLaMfX52W/fQPGCq1+4ziDXsqGSiSCM3RUv/onimEoVIEsziUzk+JOujOpRKPyxkqEcItxUvyAjBlMQeEFS2FCD+hTp8FHWsspG0ULYzYpzhul9KdyGwCMUMM5CajB89ZKv1oL6sG06CoIZ6BGZnXZr/6NBprnpSMumbXdMMiwVziTgktw/nPrLPNrNoSug8r5s71imteEfnHMgCbauE8hnbLzEwVLrR2nsVOmDEd2uVeSL/W6OSbfe4VQWY6g+OOiJJcUNS3DpwNhgKMcO8C4Ec4r5SNmGEf3i8oQwuUnr4LW13p4XD+++lY7PZvFsUF2yWdySEJyQk7JL3JJmoST/96Ot+vteff+J3/fp49S35vNfCQL5R89ABJDvfA=</latexit>DepartmentNameSchoolNameCSEngineering
<latexit sha1_base64="1IwX8ysg0Eer1mS6wWT6roOJRX0=">AAACwXichVFLT+MwEHayPMuruxy5WFQgTlVSoe6eEKJ74MRDUB5qqspxJq2F40T2ZEWV7Z/kBP8GJxREAYmxLH355puJ55swk8Kg5z057o+5+YXFpeXayura+kb9568rk+aaQ5enMtU3ITMghYIuCpRwk2lgSSjhOrzrlPnrf6CNSNUljjPoJ2yoRCw4Q0sN6o9BCEOhCmRhLpmeFP9ldSa1YFT2rAUI9xjGxV/ImMYEFJ6wBCZ0l75mOvYtBjppNMOexjFooYa3YLvSIJjpZ2zVxZu6+vI9vyRaXqv9vbj1pRhU9DZHbVBveE2vCvoZ+FPQINM4G9QfgijleTkhl8yYnu9l2C/s0IJLsH7kxlrA79gQehYq64LpF9UGJnTHMhGNU22vQlqx7ysKlhgzTkKrTBiOzMdcSX6V6+UY/+kXQmU5guIvP4pzSTGl5TppJDRwlGMLGNfCvpXyEdOMo116aYL/ceTP4KrV9NvN9vl+4/BoascS2SLbZI/45Dc5JMfkjHQJdw6cyEkc5XZc4WaufpG6zrRmk8yEWzwDfarZiQ==</latexit>DepartmentNameCourseCodeO↵eringYearCSCS1012026CSCS2012026<latexit sha1_base64="Ybc34efdyF60Fml7aidtLL3l3lI=">AAACb3icbVFNT9tAEF27lNK00EAPPYDQqlERXCK7qqBHBK3UEwKl+ZDiKFpvxsmK9a61O0ZEJtf+wN76H3rhH7AOAeWDWVl6evNm5+1znElhMQj+ef6rtdfrbzbeVt6939z6UN3eaVmdGw5NrqU2nZhZkEJBEwVK6GQGWBpLaMfX52W/fQPGCq1+4ziDXsqGSiSCM3RUv/onimEoVIEsziUzk+JOujOpRKPyxkqEcItxUvyAjBlMQeEFS2FCD+hTp8FHWsspG0ULYzYpzhul9KdyGwCMUMM5CajB89ZKv1oL6sG06CoIZ6BGZnXZr/6NBprnpSMumbXdMMiwVziTgktw/nPrLPNrNoSug8r5s71imteEfnHMgCbauE8hnbLzEwVLrR2nsVOmDEd2uVeSL/W6OSbfe4VQWY6g+OOiJJcUNS3DpwNhgKMcO8C4Ec4r5SNmGEf3i8oQwuUnr4LW13p4XD+++lY7PZvFsUF2yWdySEJyQk7JL3JJmoST/96Ot+vteff+J3/fp49S35vNfCQL5R89ABJDvfA=</latexit>DepartmentNameSchoolNameCSEngineering
<latexit sha1_base64="1IwX8ysg0Eer1mS6wWT6roOJRX0=">AAACwXichVFLT+MwEHayPMuruxy5WFQgTlVSoe6eEKJ74MRDUB5qqspxJq2F40T2ZEWV7Z/kBP8GJxREAYmxLH355puJ55swk8Kg5z057o+5+YXFpeXayura+kb9568rk+aaQ5enMtU3ITMghYIuCpRwk2lgSSjhOrzrlPnrf6CNSNUljjPoJ2yoRCw4Q0sN6o9BCEOhCmRhLpmeFP9ldSa1YFT2rAUI9xjGxV/ImMYEFJ6wBCZ0l75mOvYtBjppNMOexjFooYa3YLvSIJjpZ2zVxZu6+vI9vyRaXqv9vbj1pRhU9DZHbVBveE2vCvoZ+FPQINM4G9QfgijleTkhl8yYnu9l2C/s0IJLsH7kxlrA79gQehYq64LpF9UGJnTHMhGNU22vQlqx7ysKlhgzTkKrTBiOzMdcSX6V6+UY/+kXQmU5guIvP4pzSTGl5TppJDRwlGMLGNfCvpXyEdOMo116aYL/ceTP4KrV9NvN9vl+4/BoascS2SLbZI/45Dc5JMfkjHQJdw6cyEkc5XZc4WaufpG6zrRmk8yEWzwDfarZiQ==</latexit>DepartmentNameCourseCodeO↵eringYearCSCS1012026CSCS2012026<latexit sha1_base64="h7lyj5A4vxJnS91EKAZwpjn2PH4=">AAACZXicbVHPb9MwFHayASVj0G2ICwesVSBOVTJNg+NgFy6godJtUlNVL85La+Y4kf2CVmX5J3fjyoV/A6frNuh4lqVP3/d++XNSKmkpDH96/tr6g4ePOo+DjSebT591t7ZPbFEZgUNRqMKcJWBRSY1DkqTwrDQIeaLwNDk/avXTH2isLPQ3mpc4zmGqZSYFkKMm3cs4wanUNUFSKTBNfancaYJ41nYMYsILSrJ6QFWKmr5Ajg1/w1foz/C9ME0c31R9UFIgH4ABl3s0uBNi1OntqCCYdHthP1wEvw+iJeixZRxPuldxWogqdzOFAmtHUVjSuAZDUih0W1cWSxDnMMWRg9qta8f1wqWGv3ZMyrPCuKuJL9i/K2rIrZ3nicvMgWZ2VWvJ/2mjirL341rqsiLU4npQVilOBW8t56k0KEjNHQBhpNuVi5kzR5D7mNaEaPXJ98HJXj866B983e8dflza0WEv2S57yyL2jh2yT+yYDZlgv7yOt+Vte7/9Tf+5/+I61feWNTvsn/Bf/QH3FLiI</latexit>StudentNameStudentMajorAlice SaraCS
<latexit sha1_base64="rfMu6hWb+DgciOyfHtwpplitlmc=">AAACk3ichVFNT9tAEF2bQmmg4LbqqZdVI1BPkR2hUKkHQtJDL62oQgApjqL1epysWK+t3TFq5PoP8XO48W9Y56MqUKmzWunpzZvdmTdRLoVB37933I0Xm1svt181dnZf7+17b95emKzQHIY8k5m+ipgBKRQMUaCEq1wDSyMJl9F1v85f3oA2IlPnOM9hnLKpEongDC018W7DCKZClciiQjJdlb+lPVUjnNUvNkKEXxgl5QCLGBT+YClU9JCu6b5tw0A/i6EKw3XNqRQc6IBptlYaqxwEfvB/UfuRKAQV/+msMfGafstfBH0OghVoklWcTby7MM54kdrGuWTGjAI/x3HJNAouwc5YGMgZv2ZTGFmo7HBmXC48reiBZWKaZNpehXTB/l1RstSYeRpZZcpwZp7mavJfuVGByedxKVReICi+/CgpJMWM1guisdDAUc4tYFwL2yvlM+sTR7vG2oTg6cjPwUW7FXRanZ9HzW5vZcc2+UA+kk8kIMekS76RMzIk3PGcjnPidN337he3535dSl1nVfOOPAr3+wPt+Mga</latexit>StudentNameCourseCodeAlice SaraCS101Alice SaraCS201Course  (entity)Department  (entity)Offers  (relationship)Course  (entity)Department  (entity)Offers  (relationship)Student  (entity)Takes  (relationship)(b) After Q1(c) After Q2(a) InitializationEvolve (Q1)Evolve (Q2)<latexit sha1_base64="HkCDkcLB2QXsFTxTXv6taglUEYM=">AAAB8HicbVBNS8NAEN3Ur1q/qh69LBbBU0lEquCl4MVjBfshbSib7aRdupuE3YlQQn+FFw+KePXnePPfuG1z0NYHA4/3ZpiZFyRSGHTdb6ewtr6xuVXcLu3s7u0flA+PWiZONYcmj2WsOwEzIEUETRQooZNoYCqQ0A7GtzO//QTaiDh6wEkCvmLDSISCM7TSYw9UghMD2C9X3Ko7B10lXk4qJEejX/7qDWKeKoiQS2ZM13MT9DOmUXAJ01IvNZAwPmZD6FoaMQXGz+YHT+mZVQY0jLWtCOlc/T2RMWXMRAW2UzEcmWVvJv7ndVMMr/1MREmKEPHFojCVFGM6+54OhAaOcmIJ41rYWykfMc042oxKNgRv+eVV0rqoerVq7f6yUr/J4yiSE3JKzolHrkid3JEGaRJOFHkmr+TN0c6L8+58LFoLTj5zTP7A+fwBOlSQrw==</latexit>→I1I2I0Figure 1: Sample database evolution guided by queries.
𝐸1={Course(CourseCode,CourseTitle),
Department(DepartmentName,SchoolName)},
𝑅1={Offers(DepartmentName,CourseCode,OfferingYear)}.
Here,CourseandDepartmentare entity tables, whileOffersis
a relationship table. The attributeOfferingYearis placed inOffers
because it describes a department-course offering.
This separation is why EUR keeps the universal relation distinct
from the logical schema. 𝑈1is theuser-facingquery interface: it
presents concepts and predicates without exposing tables or join
paths.𝑆1=(𝐸 1,𝑅1)is theDBMS-facingdesign: it decomposes those
concepts into entity and relationship tables for better management.
Materialized instance.The logical schema 𝑆1specifies what the
DBMS can store, but the database is still empty. The system there-
fore identifies the materialization gap: the facts required to answer
𝑄1that are missing from the initial database instance 𝐼0=∅. It
then extracts and integrates those facts from the document snippet,
producing the materialized instance𝐼 1, as shown in Figure 1(b).
2.2 Second Query: Reusing and Extending State
Query𝑄2asks which of the courses identified by 𝑄1Alice Sara
takes, and how many credits each course has. The query therefore
reuses part of the database produced by 𝑄1: the course role 𝐶and
the course-offering scope already materialized in 𝐼1. At the same
time, it reveals new information about students, enrollments, and
course credits. As before, the system first interprets the query as
attribute roles and semantic connections.
Newattribute roles and semantic connections.The query 𝑄2
reuses course role 𝐶from𝑈1and reveals three new attribute roles:
𝑁(student):StudentName
𝑀(major):StudentMajor
𝐾(credits):CourseCredits
The new roles 𝑁,𝑀, and𝐾are additional attributes over which
the second query is interpreted.
Query𝑄 2also reveals new semantic connections as predicates:
𝑃student(𝑁,𝑀):student𝑁has major𝑀,
𝑃takes(𝑁,𝐶):student𝑁takes course𝐶,
𝑃credits(𝐶,𝐾):course𝐶has𝐾credits.
Thus,𝑄2extends the application vocabulary of 𝑄1from courses
and offerings to students, enrollments, and credits.
2

Back to the Future: The Evolving Universal Relation Model
Evolvinguniversal relation.The newly revealed roles and predi-
cates are merged with the existing universal relation. The resulting
universal relation𝑈 2w.r.t.(𝑄 1,𝑄2)extends𝑈 1(𝐶,𝑇,𝐷,𝑆,𝑌)as:
𝑈2(𝐶,𝑇,𝐷,𝑆,𝑌,𝑁,𝑀,𝐾).
𝑈2={(𝑐,𝑡,𝑑,𝑠,𝑦,𝑛,𝑚,𝑘)|𝑃 course(𝑐,𝑡)∧𝑃 department(𝑑,𝑠)
∧𝑃offering(𝑑,𝑐,𝑦)∧𝑃 student(𝑛,𝑚)∧𝑃 takes(𝑛,𝑐)∧𝑃 credits(𝑐,𝑘)}.
The universal relation is again the query-facing interpretation of
the workload. The user asks over the attributes 𝑁,𝑀,𝐶,𝐾 together
with the previously revealed course-offering attributes. The DBMS
hides whether those roles are implemented by extending an existing
table, creating a new entity table, or adding a relationship table.
Evolvinglogical schema.After 𝑈2is established, the DBMS ex-
tends the logical schema underneath it. The design reuses the entity
and relationship tables from 𝑆1and adds only what is needed for
the new query, annotated by 𝑋.
𝐸2={Course(cname,title, credits),Department(dname,school),
Student(sname,major)},
𝑅2={Offers(dname,cname,year), Takes(sname,cname)}.
This shows how schema evolution is driven by the query but
mediated through the universal relation. 𝑈2manages the expanded
query-facing attributes and predicates, while 𝑆2=(𝐸 2,𝑅2)is one
DBMS-chosen decomposition into entity and relationship tables.
Evolvingmaterialized instance.The updated logical schema
𝑆2=(𝐸 2,𝑅2)says what can be stored, but not all needed facts
have been materialized yet. The database can reuse the course,
department, and offering facts already extracted for 𝑄1. It only needs
to extract the facts newly required by 𝑄2: course credits, student
record, and the enrollment facts. Given the above requirement, the
system then extracts required facts as𝐼 2shown in Figure 1(c).
3 THE EUR MODEL
Classical Universal Relation Model (URM) assumes that the uni-
versal relation is defined upfront by its predicates and dependen-
cies [ 14]. Our proposedEvolving Univeral Model(EUR) keeps the
one-relation interface, but changes how that relation is built: in-
stead of requiring the full design before querying, the database
evolves as queries arrive over the document collection. We now
make this precise.
Evolution step.The EUR model captures application-bounded,
query-driven evolution for document-backed databases. Given a
coherent application A, a document collection 𝐹1, and an evolving
workload(𝑄1,...,𝑄𝑛), EUR treats each workload increment 𝑄𝑖as
both a query to answer and a controlled database evolution step:
𝐷𝑖−1𝑄𝑖,𝐹;𝐷𝑖.
The applicationAis the boundary of evolution: it defines the
recurring intents, answer forms, governance rules, and evidence
standards that make change controlled rather than open-ended. A
workload increment 𝑄𝑖may be a natural-language query, a batch
of related queries, or an emerging query pattern. It is interpreted
over application concepts, not over the tables that happen to exist.
1For simplicity, we first treat 𝐹as fixed. In practice, EUR also allows document changes,
denotedΔ𝐹, which trigger new document-to-database extraction steps.
Reveal query rolesOrganize schemaMaterialize factsUniversal RelationEntities andRelationshipsEntity Sets and Relationship SetsWorkload increments {Q1, Q2, … Qn}Document collection FControlled Evolution
<latexit sha1_base64="ruGLSdfI+KZJ/GdgEHuMetNHf1o=">AAACDnicbVC7SgNBFJ2NrxhfUUubxRCw0LArEi2DBrFMwDwgCcvs5CYZMruzzNwVwpIvsPFXbCwUsbW282+cPApNPDBwOOdc7tzjR4JrdJxvK7Wyura+kd7MbG3v7O5l9w/qWsaKQY1JIVXTpxoED6GGHAU0IwU08AU0/OHNxG88gNJchvc4iqAT0H7Ie5xRNJKXzZe9hJ+540xbTmKASdXjp7fjpC2AdjXKcabscS+bcwrOFPYyceckR+aoeNmvdleyOIAQmaBat1wnwk5CFXImwGyLNUSUDWkfWoaGNADdSabnjO28Ubp2TyrzQrSn6u+JhAZajwLfJAOKA73oTcT/vFaMvatOwsMoRgjZbFEvFjZKe9KN3eUKGIqRIZQpbv5qswFVlKGpJmNKcBdPXib184JbLBSrF7nS9byONDkix+SEuOSSlMgdqZAaYeSRPJNX8mY9WS/Wu/Uxi6as+cwh+QPr8weLPJvR</latexit>Di→1Qi,F↭DiFigure 2: Query-driven database evolution.
Database state.After processing𝑄 1,...,𝑄𝑖, the database state is:
𝐷𝑖=(𝑈𝑖,𝑆𝑖,𝐼𝑖).
Here,𝑈𝑖is the evolving universal relation: the user-facing concep-
tual relation. 𝑆𝑖=(𝐸𝑖,𝑅𝑖)is the evolving logical design, organized
into entity and relationship schemas. 𝐼𝑖is the materialized instance
extracted and integrated from documents.
Three coupled levels.As shown in Figure 2, each transition
𝐷𝑖−1𝑄𝑖,𝐹;𝐷𝑖is implemented through three coupled levels: (1)Re-
veal:expose new attribute roles, concepts, predicates, and semantic
connections, thereby evolving 𝑈𝑖−1to𝑈𝑖; (2)Organize:place the
revealed concepts into entity and relationship schemas, thereby
evolving𝑆𝑖−1to𝑆𝑖; and (3)Materialize:extract, reconcile, validate,
and store document-backed facts, thereby evolving𝐼 𝑖−1to𝐼𝑖.
Thus, users continue to ask over a conceptual universal relation,
while the DBMS evolves the universal relation, logical design, and
materialized instance underneath.
Answer plus durable evolution.Each step has two outcomes:
an answer to the current query and an updated durable database
state𝐷𝑖. Existing Text-to-SQL [ 11] techniques can help produce
the answer once the needed schema and facts are available. The
distinctive role of EUR is to decide what state the query should
induce when the needed roles, schemas, or facts are missing.
Latent limit state.For application A, document collection 𝐹, and
long-running workload 𝑊∞=(𝑄 1,𝑄2,...) , the transition sequence
𝐷0𝑄1,𝐹;𝐷 1𝑄2,𝐹;···;𝐷∞
can be viewed as revealing finite approximations to an eventual
application-bounded latent state 𝐷∞=(𝑈∞,𝑆∞,𝐼∞). Here,𝑈∞de-
notes the latent universal relation: the semantic closure of the
attribute roles and connections that the workload would eventually
make relevant under A. Each finite state 𝐷𝑖is the governed ap-
proximation reached after the prefix (𝑄1,...,𝑄𝑖), and each query
decides which part of the latent state is worth revealing, organizing,
and materializing now.
Starting assumptions.An EUR database may start from either
an initial database or an empty database. In production, applica-
tions often provide partial schemas, mappings, extracted facts, or
coverage metadata upfront; this initializes some of𝑈 0,𝑆0, and𝐼 0.
3

Nan Tang and Haixun Wang
4 REVEAL, ORGANIZE, MATERIALIZE
As shown in Figure below, each query is handled as a controlled
evolution step with three parts. First, EUR reveals the concepts
needed to understand the query. Second, it organizes those concepts
into entity and relationship schemas. Third, it materializes the
missing document-backed facts needed to answer the query.
Reveal
Δ𝑈𝑖=Reveal(𝑄 𝑖,𝑈𝑖−1)
𝑈𝑖−1Δ𝑈𝑖;𝑈𝑖Organize
Δ𝑆𝑖=Organize(𝑆 𝑖−1,𝑈𝑖,Δ𝑈𝑖)
𝑆𝑖−1Δ𝑆𝑖;𝑆𝑖Materialize
Δ𝐼𝑖=Extract(𝐹,Ω 𝑖)
𝐼𝑖−1Δ𝐼𝑖;𝐼𝑖
4.1 Reveal: The Universal Relation Grows
At step𝑖,𝑈𝑖is the conceptual relation exposed to users. It records
the query-facing vocabulary revealed so far: attribute roles that
can be returned, filtered, grouped, or aggregated, and semantic
connections that explain how those roles belong together. It is not
a stored wide table and not the logical schema.
Query-revealed roles and connections.When 𝑄𝑖arrives, EUR
identifies additional roles and connections needed to interpret it:
Δ𝑈𝑖=Reveal(𝑄 𝑖,𝑈𝑖−1).
The delta may introduce missing roles and connections, or reuse
and refine concepts already present in𝑈 𝑖−1.
Universal relation transition.The revealed delta is semantically
merged into the accumulated universal relation:
𝑈𝑖=Merge(𝑈 𝑖−1,Δ𝑈𝑖)=⇒𝑈 𝑖−1Δ𝑈𝑖;𝑈𝑖.
The merge resolves aliases, reused concepts, compatible predicates,
refined meanings, and provenance-related roles under the applica-
tion interpretation policy. The result 𝑈𝑖is the updated query-facing
contract; storage decisions are deferred to the logical-schema step.
4.2 Organize: Roles Become Relations
The universal relation is the application-facing query contract. The
logical schema is the DBMS-facing organization of that contract.
As𝑈𝑖grows, EUR places revealed roles and semantic connections
into entity and relationship schemas.
Incremental normalization.In traditional databases, normaliza-
tion decomposes a known wide relation into smaller tables around
entities and relationships. In EUR, this organization is incremental,
because the universal relation and the evidence behind it arrive
over time. Semantic connections in 𝑈𝑖, together with available func-
tional dependencies, guide whether a revealed role becomes an
entity attribute, a relationship attribute, a key/foreign-key role, or
part of a new relationship schema.
Given𝑆𝑖−1, the updated universal relation 𝑈𝑖, and the newly
revealed deltaΔ𝑈 𝑖, EUR computes the schema delta:
Δ𝑆𝑖=(Δ𝐸𝑖,Δ𝑅𝑖)=Organize(𝑆 𝑖−1,𝑈𝑖,Δ𝑈𝑖).
It then applies this delta to obtain the updated logical schema:
𝑆𝑖=(𝐸𝑖,𝑅𝑖)=Merge(𝑆 𝑖−1,Δ𝑆𝑖)=⇒𝑆 𝑖−1Δ𝑆𝑖;𝑆𝑖.
Here,𝐸𝑖is the set of entity-table schemas and 𝑅𝑖is the set of
relationship-table schemas after step 𝑖. In the monotonic version
considered here, EUR only extends the schema by introducing enti-
ties, attributes, or relationships. More general evolution may reviseearlier choices, such as moving attributes across tables or splitting
and merging schemas; such changes require schema mappings and
instance migration across versions.
4.3 Materialize: Gaps Become Facts
After𝑈𝑖determines what users can ask and 𝑆𝑖=(𝐸𝑖,𝑅𝑖)determines
how those roles and connections are organized, the instance 𝐼𝑖must
contain enough document-backed facts to support 𝑄𝑖. Materializa-
tion is the execution side of evolution: the query exposes what is
missing, and the system turns that gap into extracted, integrated,
and reusable state.
Materialization gaps.Given the evolved schema and the previous
instance, EUR derives a materialization gap Ω𝑖and passes it to the
document-to-database extraction layer:
Ω𝑖=MaterializationGap(𝑄 𝑖,Δ𝑆𝑖,𝐼𝑖−1),
Δ𝐼𝑖=Extract(𝐹,Ω 𝑖),
𝐼𝑖=Merge(𝐼 𝑖−1,Δ𝐼𝑖)=⇒𝐼 𝑖−1Δ𝐼𝑖;𝐼𝑖.
Here,Ω𝑖specifies the missing role values, entity tuples, relationship
tuples, predicate regions, or evidence obligations needed to answer
𝑄𝑖under the evolved universal relation and schema. The instance
deltaΔ𝐼𝑖contains newly extracted and reconciled facts over 𝑆𝑖,
together with the evidence and provenance metadata needed to
justify them.
Materialization boundary and policy.EUR decides what the
query reveals and how it changes 𝑈𝑖and𝑆𝑖; the document-to-
database layer decides how to close the resulting gap in 𝐼𝑖. DataMo-
saic [ 15] studies how new queries and document updates trigger
extraction, reconciliation, validation, and integration of facts.
Two materialization policies bracket the cost-reuse tradeoff.
Aschema-completepolicy extracts all facts for newly revealed
schema fragments, maximizing future reuse but risking over-
extraction. Aquery-guidedpolicy extracts only what is needed
to answer𝑄𝑖while leaving behind useful reusable facts. Detailed
extraction planning is left to the document-to-database layer.
5 DATABASE PRINCIPLES, REWIRED
EUR does not discard classical database principles; it changes when
they must operate.Documents provide the evidence; workloads
provide the design signal. A DBMS must connect the two while
preserving the principles that still matter: identity, logical path
independence, physical data independence, optimization, prove-
nance, constraints, and transactions. In EUR, the contract changes
from managing a mostly fixed database to managing the transition
𝐷𝑖−1𝑄𝑖,𝐹;𝐷𝑖as part of query processing. The universal relation, the
logical entity/relationship design, the materialized instance, and the
coverage state must evolve with the workload, while users continue
to ask over a stable conceptual interface.
5.1 What Databases Already Got Right
Database systems already provide much of what document-backed
analytics lacks. Chen’s ER model gives a language for stable enti-
ties, relationships, attributes, and constraints [ 4]. Codd’s relational
model separates logical access from physical representation and
access paths [ 5]. Ullman’s Universal Relation Model (URM) adds
4

Back to the Future: The Evolving Universal Relation Model
the user-facing ideal of logical path independence: users query at-
tributes and predicates without choosing stored relations or join
paths [ 14]. Together, these ideas express the contract that EUR
inherits: users should ask about application concepts, while the
DBMS manages logical design and physical access underneath.
This separation is exactly what document-backed applications
need, but under a different timing assumption. Recent work ar-
gues that DBMSs should elevate ER concepts as first-class abstrac-
tions [ 6], which aligns with EUR: document analytics needs stable
things, relationships, identifiers, constraints, and provenance, not
merely text chunks, prompt contexts, or flat extracted tables. Rela-
tional systems also provide the execution discipline needed once
state exists. Cost-based optimization [ 12], query answering using
views and data integration [ 8], and provenance management [ 3] are
all directly relevant to EUR. Text-to-SQL [ 11] techniques can help
answer the current query once the needed schema and facts are
available. The distinctive problem in EUR is what happens when
they are not available: the query must also induce universal-relation
fragments, logical schema deltas, materialization gaps, extraction
tasks, and durable state for future use.
Relation to pay-as-you-go.Dataspaces and PAYGO established
that useful access need not wait for complete semantic integra-
tion, and that integration effort should be invested incrementally
as needs emerge [ 7,10]. EUR shares this philosophy but targets
a specific database-model question. Whereas PAYGO addresses
web-scale heterogeneity through schema clusters, approximate
mappings, and query routing, EUR targets application-bounded
document analytics through an evolving universal relation. Each
workload increment may jointly extend the query-facing vocabu-
lary, reorganize its logical representation, and materialize missing
facts. These changes are coupled effects of query processing, not
independent improvements to source integration. EUR thus spe-
cializes the pay-as-you-go philosophy into a model and systems
agenda for governing the database state that queries leave behind.
5.2 Why Schema-First Breaks
Classical DBMS contracts are schema-first. A catalog records re-
lations after they are declared; an optimizer chooses plans over
existing tables and indexes; storage assumes known layouts; mainte-
nance begins from base data already inside the system. EUR changes
this order. A workload increment may reveal a new attribute, add an
entity type, extend an existing entity, introduce a relationship table,
or require facts that have not yet been extracted from documents.
The user should not have to follow these changes. Logical path
independence means that the user continues to ask over the con-
ceptual application domain, while the DBMS decides how concepts
map to evolving tables, joins, stored facts, and coverage state.
The individual actions are not exotic. SQL already supports table
creation, schema alteration, tuple insertion, and merge. EUR there-
fore does not require a new storage substrate merely to represent
evolving tables. The challenge is control: today these operations
are treated as occasional administrative changes or ordinary trans-
actional writes, not as planned effects of query processing. Current
DBMSs are not optimized for workloads in which queries repeatedly
trigger small schema changes, document extraction jobs, new rela-
tionship tables, coverage updates, and new physical access paths.Thus familiar components must be reinterpreted. The catalog
must become an evolution catalog: it must record not only schemas,
but also universal-relation roles, aliases, workload-induced scopes,
schema versions, document mappings, materialization coverage,
provenance, and confidence. Storage engines must support schemas
whose attributes and relationships are discovered gradually. Record
layouts, column files, compression dictionaries, statistics, and meta-
data generally assume that columns are declared first and changed
infrequently. In EUR, adding attributes, creating relationship tables,
backfilling columns, and reorganizing stored state become normal
execution events rather than rare administrative operations.
Indexing exposes the same mismatch. Classical indexes such as
B+-trees are excellent for range predicates over stable attributes,
but they are not designed for attributes, tuples, and relation-
ships discovered on demand. Creating, updating, and discarding
indexes over newly extracted columns may cost more than the
query they support. The optimizer must therefore choose among
reuse, natural-language-to-query compilation over existing state,
universal-relation evolution, logical schema evolution, extraction,
validation, indexing, and over-materialization. Its cost model must
account not only for disk I/O and CPU, but also for extraction cost,
uncertainty, coverage gaps, and expected future reuse.
5.3 Evolution as the New DBMS Primitive
In EUR, a query should not merely consume an existing database;
it should help shape the database that ought to exist. Each query
exposes a small part of the application world, and the DBMS turns
that exposure into lasting conceptual, logical, and materialized
state. The user continues to ask over stable application concepts,
while the system quietly lets the universal relation, schema, facts,
coverage, and access structures grow with the workload.
This vision also repositions document extraction. The document-
to-database layer is not an offline ETL stage, but the mechanism
by which missing evidence becomes governed database state [ 15].
A query is complete only when it returns an answer and leaves
behind reusable, validated, and accountable state for future queries.
6 THE STACK THAT EVOLVES
We now turn the model contract into a systems agenda: what DBMS
stack should manage when the universal relation, logical schema,
and materialized instance are all evolving?
Storage for sparse, evolving coverage.Logical path indepen-
dence is useful only if the physical layer can absorb evolution with-
out exposing it to users. Row-oriented storage is a poor fit when
attributes are added frequently and coverage is partial: adding an
attribute may require changing layouts, rewriting records, refresh-
ing statistics, and distinguishing true nulls from values not yet
extracted. Columnar storage is a more natural starting point: newly
revealed attributes can be appended as columns, missing values can
remain sparse, and queries can scan only the attributes in scope.
The columnar designs of C-Store [ 13] and MonetDB [ 2] provide
useful foundations. The open problem is to adapt column storage
toevolving coverage: columns may be materialized only for selected
entities, documents, or workload scopes, and backfilled later with
provenance links and coverage metadata.
5

Nan Tang and Haixun Wang
Indexes that follow the workload.Hash indexes remain useful
for point lookups and key-based joins, and B+-trees remain attrac-
tive for range predicates over stable attributes. But EUR creates a
different regime: attributes, tuples, and relationships are discovered
on demand, and a newly extracted column may be useful for only
a few queries. Building, updating, and discarding indexes may cost
more than the query they support. Packed-memory-array layouts,
as used in cache-oblivious indexing [ 1], reduce reorganization cost
under insertions, while database cracking [ 9] incrementally reorga-
nizes columns as queries arrive. EUR needs adaptive indexes that
track workload-induced scopes, partial coverage, universal-relation
predicates, and expected reuse as part of the evolution plan.
Optimization as evolution planning.The optimizer cannot
be only an access-path selector over fixed base tables. It must
choose among answering over existing state, compiling natural
language to queries, extending the universal relation, evolving the
logical schema, extracting missing facts, building indexes, and over-
materializing for reuse. Optimization therefore becomes a joint
problem of query planning and evolution planning: the plan de-
cides not only how to compute the answer, but also what durable
state should remain afterward. The cost model must include doc-
ument retrieval, LLM calls, extraction failures, validation, entity
resolution, index maintenance, schema migration, coverage updates,
and expected reuse, in addition to I/O and CPU. The objective is no
longer cheapest current execution; it is cheapest useful evolution.
Normalization when meanings arrive late.In classical design,
many-to-many relationships normally become relationship tables,
while one-to-one and many-to-one relationships are often repre-
sented as attributes or foreign keys. In EUR, the cardinality and
meaning of a relationship may not be known when it first appears
in the universal relation. A system may initially materialize rela-
tionships uniformly as relationship tables; later workloads may
reveal that some should collapse into entity attributes or foreign
keys, while others should become richer relationship entities. The
research problem is safe normalization and denormalization over
partially materialized data while preserving provenance, coverage,
schema versions, and previously answered-query semantics.
Continuous cleaning and constraints.Data quality problems
are amplified when facts are extracted incrementally from hetero-
geneous evidence. Extracted tuples may be duplicated, incomplete,
contradictory, or expressed in incompatible units and formats. Nor-
malization, transformation, entity resolution, and constraint check-
ing therefore become continuous operations rather than one-time
cleaning steps. Constraints must distinguish database violations
from evidence uncertainty: a key conflict may indicate a duplicate,
an extraction error, an ambiguous entity, or genuinely conflicting
evidence. EUR needs constraints that govern both schema evolution
and materialized facts while preserving provenance and evidence.
Extraction as a physical operator.Extraction should not be an
offline ETL script placed before the database. It should be a first-
class physical operator whose input is a materialization gap and
whose output is typed tuples, provenance, confidence, coverage
updates, and failures [ 15]. This operator must support projection
pushdown, predicate pushdown, batching, entity resolution, val-
idation, and incremental maintenance. Treating extraction as anoperator gives the optimizer something concrete to cost, reorder,
cache, and combine with relational operators, while making missing
evidence, conflicting evidence, and low-confidence values explicit
outputs rather than hidden prompt artifacts.
Transactions over data and evolution.Transaction manage-
ment must be revisited. In EUR, many writes may run concurrently:
universal-relation changes add roles, schema-evolution actions add
attributes or relationship tables, extraction jobs add tuples, valida-
tion jobs repair values, and user queries may read partially material-
ized state. The system needs isolation levels for evolving databases:
should a query see freshly extracted tuples before validation, should
related universal-relation, schema, and tuple deltas commit atomi-
cally, and should failed extraction calls roll back or leave partial ev-
idence? Without such semantics, the same natural-language work-
load may produce non-repeatable answers, inconsistent coverage,
and irreproducible schema states.
Semantic operators for multimodal evidence.EUR needs se-
mantic operators that treat interpretation itself as part of query
processing. Document collections mix text, tables, forms, figures,
screenshots, and images; a query may require reading a clause,
locating a table cell, comparing a chart, or linking a visual region
to an entity. These operations should become typed operators with
inputs, outputs, costs, confidence, provenance, and failure modes,
so the optimizer can combine them with relational operators and
the system can materialize their results as reusable state.
7 CONCLUSION
RAG made documents searchable; EUR asks the database question:
what durable, governed state should each query leave behind? The
answer is not a schema-first replica of the corpus, but a query-
shaped database whose universal relation, logical schema, material-
ized facts, coverage, indexes, and constraints evolve together while
preserving logical path independence for users. The agenda is clear:
make evolution—universal-relation deltas, schema deltas, extraction
deltas, coverage deltas, validation, indexing, and transactions—a
first-class DBMS operation.
REFERENCES
[1]Michael A. Bender, Erik D. Demaine, and Martin Farach-Colton. 2005. Cache-
Oblivious B-Trees.SIAM J. Comput.35, 2 (2005), 341–358.
[2]Peter A. Boncz, Martin L. Kersten, and Stefan Manegold. 2008. Breaking the
memory wall in MonetDB.Commun. ACM51, 12 (2008), 77–85.
[3] Peter Buneman, Sanjeev Khanna, and Wang-Chiew Tan. 2001. Why and Where:
A Characterization of Data Provenance. InICDT.
[4]Peter Pin-Shan Chen. 1976. The Entity-Relationship Model: Toward a Unified
View of Data.ACM Transactions on Database Systems1, 1 (1976), 9–36.
[5]E. F. Codd. 1970. A Relational Model of Data for Large Shared Data Banks.
Commun. ACM13, 6 (1970), 377–387.
[6]Amol Deshpande. 2025. Beyond Relations: A Case for Elevating to the
Entity-Relationship Abstraction. https://www.cidrdb.org/cidr2025/papers/p11-
deshpande.pdf. CIDR 2025.
[7] Michael J. Franklin, Alon Y. Halevy, and David Maier. 2005. From databases to
dataspaces: a new abstraction for information management.SIGMOD Rec.34, 4
(2005), 27–33.
[8]Alon Y. Halevy. 2001. Answering Queries Using Views: A Survey.The VLDB
Journal10, 4 (2001), 270–294.
[9] Stratos Idreos, Martin L. Kersten, and Stefan Manegold. 2007. Database Cracking.
InCIDR.
[10] Shawn R. Jeffery, Michael J. Franklin, and Alon Y. Halevy. 2008. Pay-as-you-
go User Feedback for Dataspace Systems. InProceedings of the ACM SIGMOD
International Conference on Management of Data (SIGMOD ’08). Association for
Computing Machinery, 847–860.
6

Back to the Future: The Evolving Universal Relation Model
[11] Xinyu Liu, Shuyu Shen, Boyan Li, Peixian Ma, Runzhi Jiang, Yuxin Zhang, Ju
Fan, Guoliang Li, Nan Tang, and Yuyu Luo. 2025. A Survey of Text-to-SQL in
the Era of LLMs: Where Are We, and Where Are We Going?IEEE Trans. Knowl.
Data Eng.37, 10 (2025), 5735–5754.
[12] P. Griffiths Selinger, M. M. Astrahan, D. D. Chamberlin, R. A. Lorie, and T. G.
Price. 1979. Access Path Selection in a Relational Database Management System.
InSIGMOD.[13] Michael Stonebraker, Daniel J. Abadi, Adam Batkin, Xuedong Chen, Mitch Cher-
niack, Miguel Ferreira, Edmond Lau, Amerson Lin, Samuel R. Madden, Elizabeth
O’Neil, Patrick E. O’Neil, Alexander Rasin, Nga Tran, and Stanley B. Zdonik.
2005. C-Store: A Column-Oriented DBMS. InVLDB.
[14] Jeffrey D. Ullman. 1982. The U. R. Strikes Back. InPODS.
[15] Zhengxuan Zhang, Zhuowen Liang, Jiazhuo Chen, Haixun Wang, and Nan Tang.
2026. Document-to-Database: Extraction Meets Relational Semantics.PVLDB.
(2026).
7