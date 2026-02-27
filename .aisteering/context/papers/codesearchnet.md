CodeSearchNet Challenge Evaluating the State of Semantic Code Search
Hamel Husain Ho-Hsiang Wu Tiferet Gazit
{hamelsmu,hohsiangwu,tiferet}@github.com GitHub
Miltiadis Allamanis Marc Brockschmidt
{miallama,mabrocks}@microsoft.com Microsoft Research
ABSTRACT
Semantic code search is the task of retrieving relevant code given a natural language query. While related to other information retrieval tasks, it requires bridging the gap between the language used in code (often abbreviated and highly technical) and natural language more suitable to describe vague concepts and ideas.
To enable evaluation of progress on code search, we are releasing the CodeSearchNet Corpus and are presenting the CodeSearch-Net Challenge, which consists of 99 natural language queries with about 4k expert relevance annotations of likely results from Code-SearchNet Corpus. The corpus contains about 6 million functions from open-source code spanning six programming languages (Go, Java, JavaScript, PHP, Python, and Ruby). The CodeSearchNet Corpus also contains automatically generated query-like natural language for 2 million functions, obtained from mechanically scrap-ing and preprocessing associated function documentation. In this article, we describe the methodology used to obtain the corpus and expert labels, as well as a number of simple baseline solutions for the task.
We hope thatCodeSearchNetChallenge encourages researchers and practitioners to study this interesting task further and will host a competition and leaderboard to track the progress on the chal-lenge. We are also keen on extending CodeSearchNet Challenge to more queries and programming languages in the future.
1 INTRODUCTION
The deep learning revolution has fundamentally changed how we approach perceptive tasks such as image and speech recognition and has shown substantial successes in working with natural lan-guage data. These have been driven by the co-evolution of large (labelled) datasets, substantial computational capacity, and a num-ber of advances in machine learning models.
However, deep learningmodels still struggle on highly structured data. One example is semantic code search: while search on natural language documents and even images has made great progress, searching code is often still unsatisfying. Standard information re-trieval methods do not workwell in the code search domain, as there is often little shared vocabulary between search terms and results (e.g. consider amethod called deserialize_JSON_obj_from_stream that may be a correct result for the query “read JSON data”). Even more problematic is that evaluating methods for this task is ex-tremely hard, as there are no substantial datasets that were created for this task; instead, the community tries to make do with small datasets from related contexts (e.g. pairing questions on web forums to code chunks found in answers).
To tackle this problem, we have defined the CodeSearchNet Challenge on top of a new CodeSearchNet Corpus. The Code-SearchNet Corpus was programmatically obtained by scraping open-source repositories and pairing individual functions with their (processed) documentation as natural language annotation. It is large enough (2 million datapoints) to enable training of high-capacity deep neural models on the task. We discuss this process in detail in section 2 and also release the data preprocessing pipeline to encourage further research in this area.
The CodeSearchNet Challenge is defined on top of this, pro-viding realistic queries and expert annotations for likely results. Concretely, in version 1.0, it consists of 99 natural languages queries paired with likely results for each of six considered programming languages (Go, Java, JavaScript, PHP, Python, and Ruby). Each query/result pair was labeled by a human expert, indicating the relevance of the result for the query. We discuss the methodology in detail in section 3.
Finally, we create a number of baseline methods using a range of state-of-the-art neural sequence processing techniques (bag of words, RNNs, CNNs, attentional models) and evaluate them on our datasets. We discuss these models in section 4 and present some preliminary results.
2 THE CODE SEARCH CORPUS
As it is economically infeasible to create a dataset large enough for training high-capacity models using expert annotations, we instead create a proxy dataset of lower quality. For this, we follow other attempts in the literature [5, 6, 9, 11] and pair functions in open-source software with the natural language present in their respective documentation. However, to do so requires a number of preprocessing steps and heuristics. In the following, we discuss some general principles and decisions driven by in-depth analysis of common error cases.
CodeSearchNet Corpus Collection. We collect the corpus from publicly available open-source non-fork GitHub repositories, using libraries.io to identify all projects which are used by at least one other project, and sort them by “popularity” as indicated by the number of stars and forks. Then, we remove any projects that do not have a license or whose license does not explicitly permit the re-distribution of parts of the project. We then tokenize all Go, Java, JavaScript, Python, PHP and Ruby functions (or methods) using TreeSitter — GitHub’s universal parser — and, where available, their respective documentation text using a heuristic regular expression.
Filtering. To generate training data for the CodeSearchNet Challenge, we first consider only those functions in the corpus
CodeSearchNet
Number of Functions
w/ documentation All
Go 347 789 726 768 Java 542 991 1 569 889 JavaScript 157 988 1 857 835 PHP 717 313 977 821 Python 503 502 1 156 085 Ruby 57 393 164 048
All 2 326 976 6 452 446 Table 1: Dataset Size Statistics
that have documentation associated with them. This yields a set of pairs (ci , di ) where ci is some function documented by di . To make the data more realistic proxy for code search tasks, we then implement a number of preprocessing steps:
• Documentation di is truncated to the first full paragraph, to make the length more comparable to search queries and re-move in-depth discussion of function arguments and return values.
• Pairs in which di is shorter than three tokens are removed, since we do not expect such comments to be informative.
• Functions ci whose implementation is shorter than three lines are removed, these often include unimplemented meth-ods, getters, setters, etc.
• Functions whose name contains the substring “test” are re-moved. Similarly, we remove constructors and standard ex-tension methods such as __str__ in Python or toString in Java.
• We remove duplicates from the dataset by identifying (near) duplicate functions and only keeping one copy of them (we use the methods described in Allamanis [1], Lopes et al. [18]). This removes multiple versions of auto-generated code and cases of copy & pasting.
The filtered corpus and the data extraction code are released at https://github.com/github/CodeSearchNet.
Dataset Statistics. The resulting dataset contains about 2 million pairs of function-documentation pairs and about another 4 million functions without an associated documentation (Table 1). We split the dataset in 80-10-10 train/valid/test proportions. We suggest that users of the dataset employ the same split.
Limitations. Unsurpsingly, the scraped dataset is quite noisy. First, documentation is fundamentally different from queries, and hence uses other forms of language. It is often written at the same time and by the same author as the documented code, and hence tends to use the same vocabulary, unlike search queries. Second, despite our data cleaning efforts we are unable to know the extent to which each documentation di accurately describes its associated code snippet ci . For example, a number of comments are outdated with regard to the code that they describe. Finally, we know that some documentation is written in other languages, whereas our CodeSearchNet Challenge evaluation dataset focuses on English queries.
3 THE CODE SEARCH CHALLENGE
To evaluate on the CodeSearchNet Challenge, a method has to return a set of relevant results from CodeSearchNet Corpus for each of 99 pre-defined natural language queries. Note that the task is somewhat simplified from a general code search task by only allowing full functions/methods as results, and not arbitrary chunks of code.1 The CodeSearchNet Challenge evaluation dataset con-sists of the 99 queries with relevance annotations for a small number of functions from our corpus likely to be returned. These annota-tions were collected from a small set of expert programmers, but we are looking forward to widening the annotation set going forward.
Query Collection. To ensure that our query set is representative, we obtained common search queries from Bing that had high click-through rates to code and combined these with intent rewrites in StaQC [24]. We then manually filtered out queries that were clearly technical keywords (e.g. the exact name of a function such as tf.gather_nd) to obtain a set of 99 natural language queries. While most of the collected queries are generic, some of them are language-specific.
Expert Annotations. Obviously, we cannot annotate all query/-function pairs. To filter this down to a more realistically-sized set, we used our implementations of baseline methods and ensembled them (see section 4) to generate 10 candidate results per query and programming language. Concretely, we used ensembles of all neural models and ElasticSearch to generate candidate results, merge the suggestions and pick the top 10. We used a simple web interface for the annotation process. The web interface firsts shows instructions (see Figure 1) and then allows the annotator to pick a program-ming language. Then, one query/function pair is shown at a time, as shown in Figure 2. A link to the origin of the shown function is included, as initial experiments showed that some annotators found inspecting the context of the code snippet helpful to judge relevance. The order of query/code pairs shown to the user is ran-domized but weakly ordered by the number of expert annotations already collected. Annotators are unlikely to see several results for the same query unless they handle many examples. By randomizing the order, we aim to allow users to score the relevance of each pair individually without encouraging comparisons of different results for the same query.
Annotation Statistics. We collected 4 026 annotations across six programming languages and prioritized coverage over multiple annotations per query-snippet pair. Our annotators are volunteers with software engineering, data science and research roles and were asked to only annotate examples for languages they had significant experience with. This led to a skewed distribution of annotations w.r.t. the considered programming languages.
We observed that the obtained relevance scores are distributed differently for each language (Table 2). For example, the relevance scores for Python are evenly distributed across the four categories while for JavaScript the annotations are skewed towards lower relevance scores. There is a number of potential reasons for this, such as the quality of the used corpus, language-specific interactions
1Note that on a sufficiently large dataset, this is not a significant restriction: more com-monly implemented functionality almost always appears factored out into a function somewhere.
https://lh3.googleusercontent.com/notebooklm/ANHLwAzqcAmWMDM1LwerJzo6koyiEkASrOE6g_6GEk3j0sqo-Tm6jIHiPjSD_XqTbentEJ7Ish2U22e28Rf0rC51zpANT7RSz-CElYLljihdITxhsVrkVAp0KrRlzLXrFsA2FqXom-tI_g=w700-h466-v0
https://lh3.googleusercontent.com/notebooklm/ANHLwAzGTwy6VMt0pCZQVVF96zPnd12ri8jlZ5579cOvxhG37rAgLx_5MIfEWG30OMv3OjVgGM7Rstu2W_WRHsCwgnrruP-H3bs0G3BE-luzTX-U3L3mTuROIFCw1nLr9n3e3FIjFsJJSg=w718-h617-v0
CodeSearchNet Challenge
Figure 1: Instructions provided to annotators.
Figure 2: Interface used for relevance annotation.
Count by Relevance Score Total
0 1 2 3 Annotations
Go 62 64 29 11 166 Java 383 178 125 137 823 JavaScript 153 52 56 58 319 PHP 103 77 68 66 314 Python 498 511 537 543 2 089 Ruby 123 105 53 34 315
Table 2: Annotation Dataset Statistics
with our pre-filtering strategy, the queries we collected, higher expected relevance standards in the JavaScript community, etc.
For the 891 query-code pairs where we have more than one annotation, we compute the squared Cohen’s kappa interannotator
agreement to estimate the quality of the task. The agreement is moderate with Cohen κ= 0.47. This is somewhat expected given that this task was relatively open-ended, as we will discuss next.
Qualitative Observations. During the annotation processwemade some observations in discussions with the annotators and through the notes they provided in the web interface (see Fig. 2). These com-ments point to some general issues in implementing code search:
Code Quality A subset of the results being returned are func-tionally correct code, but of low quality, even though they originated in reasonably popular projects. In this context, low quality refers to unsatisfactory readability, bad secu-rity practices, known antipatterns and potentially slow code. Some annotators felt the need to give lower relevance scores to low-quality code as they would prefer not to see such results.
Query Ambiguity Queries are often ambiguous without addi-tional context. For example, the query “how to determine if a string is a valid word” can have different correct interpre-tations depending on the domain-specific meaning of “valid word”.
Library vs. Project Specific Often a search yields code that is very specific to a given project (e.g. using internal utility functions), whereas other times the code is very general and verbose (e.g. containing code that could be factored out). Which of these is preferable depends on the context of the query, which we did not explicitly specify when asking for annotations.
Context Some results were semantically correct, but not relying on related helper functions and thus not self-contained. Some annotators were uncertain if such results should be consid-ered relevant.
Directionality A common problem in results were functions im-plementing the inverse functionality of the query, e.g. “con-vert int to string” would be answered by stringToInt. This suggests that the baseline models used for pre-filtering have trouble with understanding such semantic aspects.
3.1 Evaluation of Ranking Models
To track the progress on the CodeSearchNet Challenge we have deployed a Weights & Biases leaderboard at https://app.wandb.ai/ github/codesearchnet/benchmark. We hope that this leaderboard will allow the community to better compare solutions to the code search task.
Metrics. Weused normalized discounted cumulative gain (NDCG) to evaluate each competing method. NDCG is a commonly used metric [19] in information retrieval. We compute two variants of NDCG: (a) NDCG computed over the subset of functions with human annotations (“Within”) (b) NDCG over the whole Code-SearchNet Corpus (“All”). We make this distinction as the NDCG score computed over the whole corpus may not necessarily repre-sent the quality of a search tool, as a new tool may yield relevant but not-annotated functions.
https://lh3.googleusercontent.com/notebooklm/ANHLwAyYhxgZnvs2wpr4eIKWSlpXW0iAwQb_yBtnSaj3PjPq9Xr3rzza3D39LkhFnrlXtkc5gA-xOu7xTj7vUri63DyaxpNRYd85pE7ZdvVIDL03cTk6M6Qa7OqzbwOBtGB_XGHJnh5c=w952-h561-v0
CodeSearchNet
Figure 3: Model Architecture Overview.
4 BASELINE CODESEARCH MODELS
We implemented a range of baseline models for the code search task, using standard techniques from neural sequence processing and web search.
4.1 Joint Vector Representations for Code
Search
Following earlier work [11, 20], we use joint embeddings of code and queries to implement a neural search system. Our architecture employs one encoder per input (natural or programming) language and trains them to map inputs into a single, joint vector space. Our training objective is to map code and the corresponding language onto vectors that are near to each other, as we can then implement a search method by embedding the query and then returning the set of code snippets that are “near” in embedding space. Although more complex models considering more interactions between queries and code can perform better [20], generating a single vector per query/snippet allows for efficient indexing and search.
To learn these embedding functions, we combine standard se-quence encoder models in the architecture shown in Figure 3. First, we preprocess the input sequences according to their seman-tics: identifiers appearing in code tokens are split into subtokens (i.e. a variable camelCase yields two subtokens camel and case), and natural language tokens are split using byte-pair encoding (BPE) [10, 21].
Then, the token sequences are processed to obtain (contextual-ized) token embeddings, using one of the following architectures.
Neural Bag of Words where each (sub)token is embedded to a learnable embedding (vector representation).
Bidirectional RNN models where we employ the GRU cell [7] to summarize the input sequence.
1D Convolutional Neural Network over the input sequence of tokens [15].
Self-Attention where multi-head attention [22] is used to com-pute representations of each token in the sequence.
The token embeddings are then combined into a sequence embed-ding using a pooling function, for which we have implemented mean/max-pooling and an attention-like weighted sum mechanism.
For all models, we set the dimensionality of the embedding space to 128.
During training we are given a set of N pairs (ci , di ) of code and natural language descriptions and have instantiated a code encoder Ec and a query encoder Eq . We train by minimizing the loss
− 1 N
∑i log
( exp(Ec (ci )⊤Eq(di ))∑j exp(Ec (cj )⊤Eq(di ))
) ,
i.e. maximize the inner product of the code and query encodings of the pair, while minimizing the inner product between each ci and the distractor snippets cj (i , j). Note that we have experimented with other similar objectives (e.g. considering cosine similarity and max-margin approaches) without significant changes in results on our validation dataset. The code for the baselines can be found at https://github.com/github/CodeSearchNet.
At test time, we index all functions in CodeSearchNet Cor-pus using Annoy. Annoy offers fast, approximate nearest neighbor indexing and search. The index includes all functions in the Code-SearchNet Corpus, including those that do not have an associated documentation comment. We observed that carefully constructing this index is crucial to achieving good performance. Specifically, our baseline models were underperforming when we had a small number of trees in Annoy (trees can be thought as approximate indexes of the multidimensional space).
4.2 ElasticSearch Baseline
In our experiments, we additionally included ElasticSearch, a widely used search engine with the default parameters. We configured it with an index using two fields for every function in our dataset: the function name, split into subtokens; and the text of the entire function. We use the default ElasticSearch tokenizer.
4.3 Evaluation
Following the training/validation/testing data split, we train our baseline models using our objective from above. While it does not directly correspond to the real target task of code search, it has been widely used as a proxy for training similar models [6, 23].
For testing purposes on CodeSearchNet Corpus, we fix a set of 999 distractor snippets cj for each test pair (ci , di ) and test all trained models. Table 3 presents the Mean Reciprocal Rank results on this task. Overall, we see that the models achieve relatively good performance on this task, with the self-attention-based model performing best. This is not unexpected, as the self-attention model has the highest capacity of all considered models.
We have also run our baselines on CodeSearchNet Challenge and show the results in Table 4. Here, the neural bag of words model performs very well, whereas the stronger neural models on the training task do less well. We note that the bag of words model is particularly good at keyword matching, which seems to be a crucial facility in implementing search methods. This hypothesis is further validated by the fact that the non-neural ElasticSearch-based baseline performs competitively among all models we have tested. The NBoW model is the best performing model among the baselines models, despite being the simplest. As noted by Cam-bronero et al. [6], this can be attributed to the fact that the training
CodeSearchNet Challenge
Encoder CodeSearchNet Corpus (MRR)
Text Code Go Java JS PHP Python Ruby Avg
NBoW NBoW 0.6409 0.5140 0.4607 0.4835 0.5809 0.4285 0.6167 1D-CNN 1D-CNN 0.6274 0.5270 0.3523 0.5294 0.5708 0.2450 0.6206 biRNN biRNN 0.4524 0.2865 0.1530 0.2512 0.3213 0.0835 0.4262 SelfAtt SelfAtt 0.6809 0.5866 0.4506 0.6011 0.6922 0.3651 0.7011
SelfAtt NBoW 0.6631 0.5618 0.4920 0.5083 0.6113 0.4574 0.6505 Table 3: Mean Reciprocal Rank (MRR) on Test Set of CodeSearchNet Corpus. This evaluates for our training task, where
given the documentation comment as a query, the models try to rank the correct code snippet highly among 999 distractor
snippets.
data constructed from code documentation is not a good match for the code search task.
5 RELATEDWORK
Applying machine learning to code has been widely considered [2]. A few academic works have looked into related tasks. First, seman-tic parsing has received a lot of attention in the NLP community. Although most approaches are usually aimed towards creating an executable representation of a natural language utterance with a domain-specific language, general-purpose languages have been recently considered by Hashimoto et al. [13], Lin et al. [16], Ling et al. [17], Yin and Neubig [25].
Iyer et al. [14] generate code from natural language within the context of existing methods, whereas Allamanis et al. [3], Alon et al. [4] consider the task of summarizing functions to their names. Finally, Fernandes et al. [9] consider the task of predicting the documentation text from source code.
More related to CodeSearchNet is prior work in code search with deep learning. In the last few years there has been research in this area (Cambronero et al. [6], Gu et al. [11, 12], Yao et al. [23]), and architectures similar to those discussed previously have been shown to work to some extent. Recently, Cambronero et al. [6] looked into the same problem that CodeSearchNet is concerned with and reached conclusions similar to those discussed here. In contrast to the aforementioned works, here we provide a human-annotated dataset of relevance scores and test a few more neural search architectures along with a standard information retrieval baseline.
6 CONCLUSIONS & OPEN CHALLENGES
We hope that CodeSearchNet is a good step towards engaging with the machine learning, IR and NLP communities towards devel-oping new machine learning models that understand source code and natural language. Despite the fact this report gives emphasis on semantic code search we look forward to other uses of the pre-sented datasets. There are still plenty of open challenges in this area.
• Our ElasticSearch baseline, that performs traditional keyword-based search, performs quite well. It has the advantage of being able to efficiently use rare terms, which often appear in code. Researching neural methods that can efficiently and accurately represent rare terms will improve performance.
• Code semantics such as control and data flow are not ex-ploited explicitly by existing methods, and instead search methods seem to be mainly operate on identifiers (such as variable and function) names. How to leverage semantics to improve results remains an open problem.
• Recently, in NLP, pretraining methods such as BERT [8] have found great success. Can similar methods be useful for the encoders considered in this work?
• Our data covers awide range of general-purpose code queries. However, anecdotal evidence indicates that queries in spe-cific projects are usually more specialized. Adapting search methods to such use cases could yield substantial perfor-mance improvements.
• Code quality of the searched snippets was a recurrent issue with our expert annotators. Despite its subjective nature, there seems to be agreement on what constitutes very bad code. Using code quality as an additional signal that allows for filtering of bad results (at least when better results are available) could substantially improve satisfaction of search users.
ACKNOWLEDGMENTS
We thank Rok Novosel for participating in the CodeSearchNet chal-lenge and pinpointing a bugwith the indexing that was significantly impacting our evaluation results.
REFERENCES
[1] Miltiadis Allamanis. 2018. The Adverse Effects of Code Duplication in Machine Learning Models of Code. arXiv preprint arXiv:1812.06469 (2018).
[2] Miltiadis Allamanis, Earl T Barr, Premkumar Devanbu, and Charles Sutton. 2018. A survey of machine learning for big code and naturalness. ACM Computing Surveys (CSUR) 51, 4 (2018), 81.
[3] Miltiadis Allamanis, Hao Peng, and Charles Sutton. 2016. A Convolutional Attention Network for Extreme Summarization of Source Code. In Proceedings of the International Conference on Machine Learning (ICML).
[4] Uri Alon, Omer Levy, and Eran Yahav. 2018. code2seq: Generating sequences from structured representations of code. arXiv preprint arXiv:1808.01400 (2018).
[5] Antonio Valerio Miceli Barone and Rico Sennrich. 2017. A parallel corpus of Python functions and documentation strings for automated code documentation and code generation. arXiv preprint arXiv:1707.02275 (2017).
[6] Jose Cambronero, Hongyu Li, Seohyun Kim, Koushik Sen, and Satish Chandra. 2019. When Deep Learning Met Code Search. arXiv preprint arXiv:1905.03813 (2019).
[7] Kyunghyun Cho, Bart van Merriënboer, Dzmitry Bahdanau, and Yoshua Ben-gio. 2014. On the Properties of Neural Machine Translation: Encoder–Decoder Approaches. Syntax, Semantics and Structure in Statistical Translation (2014).
[8] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv
CodeSearchNet
Encoder CodeSearchNet Challenge– NDCG Within CodeSearchNet Challenge– NDCG All
Text Code Go Java JS PHP Python Ruby Avg Go Java JS PHP Python Ruby Avg
ElasticSearch 0.307 0.257 0.318 0.338 0.406 0.395 0.337 0.186 0.190 0.204 0.199 0.256 0.197 0.205 NBoW NBoW 0.591 0.500 0.556 0.536 0.582 0.680 0.574 0.278 0.355 0.311 0.291 0.448 0.360 0.340
1D-CNN 1D-CNN 0.379 0.407 0.269 0.474 0.473 0.420 0.404 0.120 0.189 0.099 0.176 0.242 0.162 0.165 biRNN biRNN 0.112 0.165 0.066 0.148 0.193 0.185 0.145 0.030 0.056 0.017 0.042 0.070 0.060 0.046 SelfAtt SelfAtt 0.484 0.431 0.446 0.522 0.560 0.515 0.493 0.211 0.233 0.175 0.232 0.367 0.219 0.240 SelfAtt NBoW 0.550 0.514 0.545 0.557 0.583 0.650 0.566 0.284 0.340 0.299 0.291 0.422 0.360 0.333
Table 4: NDCGBaseline Results onCodeSearchNetChallenge. “Within” computes the NDCG only on the functions within
the human-annotated examples. “All” computes NDCG over all functions in the CodeSearchNet Corpus.
preprint arXiv:1810.04805 (2018). [9] Patrick Fernandes, Miltiadis Allamanis, and Marc Brockschmidt. 2018. Structured
Neural Summarization. arXiv preprint arXiv:1811.01824 (2018). [10] Philip Gage. 1994. A new algorithm for data compression. The C Users Journal
12, 2 (1994), 23–38. [11] Xiaodong Gu, Hongyu Zhang, and Sunghun Kim. 2018. Deep code search. In 2018
IEEE/ACM 40th International Conference on Software Engineering (ICSE). IEEE, 933–944.
[12] Xiaodong Gu, Hongyu Zhang, Dongmei Zhang, and Sunghun Kim. 2016. Deep API Learning. In Proceedings of the International Symposium on Foundations of Software Engineering (FSE).
[13] Tatsunori B Hashimoto, Kelvin Guu, Yonatan Oren, and Percy S Liang. 2018. A retrieve-and-edit framework for predicting structured outputs. In Advances in Neural Information Processing Systems. 10073–10083.
[14] Srinivasan Iyer, Ioannis Konstas, Alvin Cheung, and Luke Zettlemoyer. 2018. Map-ping language to code in programmatic context. arXiv preprint arXiv:1808.09588 (2018).
[15] Yoon Kim. 2014. Convolutional neural networks for sentence classification. arXiv preprint arXiv:1408.5882 (2014).
[16] Xi Victoria Lin, Chenglong Wang, Luke Zettlemoyer, and Michael D. Ernst. 2018. NL2Bash: A Corpus and Semantic Parser for Natural Language Interface to the Linux Operating System. In International Conference on Language Resources and Evaluation.
[17] Wang Ling, Edward Grefenstette, Karl Moritz Hermann, Tomas Kocisky, Andrew Senior, Fumin Wang, and Phil Blunsom. 2016. Latent Predictor Networks for
Code Generation. In Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL).
[18] Cristina V Lopes, Petr Maj, Pedro Martins, Vaibhav Saini, Di Yang, Jakub Zitny, Hitesh Sajnani, and Jan Vitek. 2017. DéjàVu: a map of code duplicates on GitHub. Proceedings of the ACM on Programming Languages 1, OOPSLA (2017), 84.
[19] Christopher D. Manning, Prabhakar Raghavan, and Hinrich Schütze. 2008. Intro-duction to Information Retrieval. Cambridge University Press.
[20] Bhaskar Mitra, Nick Craswell, et al. 2018. An introduction to neural information retrieval. Foundations and Trends® in Information Retrieval 13, 1 (2018), 1–126.
[21] Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016. Neural machine translation of rare words with subword units. In Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL).
[22] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems. 5998–6008.
[23] Ziyu Yao, Jayavardhan Reddy Peddamail, and Huan Sun. 2019. CoaCor: Code Annotation for Code Retrieval with Reinforcement Learning. (2019).
[24] Ziyu Yao, Daniel S Weld, Wei-Peng Chen, and Huan Sun. 2018. StaQC: A Sys-tematically Mined Question-Code Dataset from Stack Overflow. In Proceedings of the 2018 World Wide Web Conference on World Wide Web. International World Wide Web Conferences Steering Committee, 1693–1703.
[25] Pengcheng Yin and Graham Neubig. 2017. A Syntactic Neural Model for General-Purpose Code Generation. Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL).
Chat
💻
Code Search eval
8 sources
The provided text describes the CodeEditSearch dataset hosted on Hugging Face, which serves as a repository for tracking source code modifications across various programming languages. It documents specific code snippets, header files, and logic changes, often presented as diffs that show the evolution of software projects like Bitcoin, Chromium, and various C-based utilities. Key metadata include string lengths, commit hashes, and licensing information such as MIT, BSD, and Apache 2.0. The dataset highlights practical programming adjustments, ranging from fixing typos and refactoring structures to addressing compilation errors and memory management. By archiving these granular edits, the collection provides a technical overview of how developers maintain and optimize diverse software ecosystems.

How do CodeSearchNet and CoSQA evaluate semantic code search performance?
What challenges do experts face when annotating code for relevance?
How do open-source models like OpenCodeInterpreter compare to proprietary systems?
Friday, February 6
describe how and what do these benchamarks measure

Based on the provided sources, here is a description of what these benchmarks measure and how they measure it:
CodeSearchNet Challenge
• What it measures: This benchmark evaluates semantic code search, which is the ability to retrieve relevant code functions given a natural language query (e.g., "how to read JSON data").
• How it measures it:
    ◦ Dataset: It utilizes a corpus of 6 million functions across six programming languages (Go, Java, JavaScript, PHP, Python, and Ruby).
    ◦ Evaluation: The core challenge involves 99 natural language queries that have been manually annotated by experts. These experts labeled likely results on a relevance scale from 0 (irrelevant) to 3 (exact match),.
    ◦ Metrics: The primary metric for the challenge is Normalized Discounted Cumulative Gain (NDCG), which accounts for the graded relevance of the results. For the broader corpus tasks (finding the correct code among distractor snippets), it also reports Mean Reciprocal Rank (MRR).
APPS (Automated Programming Progress Standard)
• What it measures: This benchmark measures code generation competence, specifically the ability of machine learning models to generate satisfactory Python code from arbitrary natural language specifications.
• How it measures it:
    ◦ Dataset: It consists of 10,000 coding problems ranging from simple one-line tasks to complex algorithmic challenges.
    ◦ Evaluation: Unlike benchmarks that rely on text similarity, APPS evaluates models by executing the generated code against test cases, similar to how software developers are assessed.
    ◦ Metrics: Performance is measured by the percentage of test cases the generated code successfully passes.
CoSQA (Code Search and Question Answering)
• What it measures: This benchmark evaluates code search and question answering, focusing on the semantic matching between web search queries and code snippets.
• How it measures it:
    ◦ Dataset: It contains 20,604 pairs of natural language queries and code snippets.
    ◦ Evaluation: Each pair is annotated by at least three human annotators to establish ground truth labels.
    ◦ Metrics: It uses accuracy to measure how well models can match queries to the correct code answers.
CodeTransOcean
• What it measures: This benchmark assesses code translation, the ability to translate source code from one programming language to another. It covers both popular and niche languages, as well as deep learning frameworks.
• How it measures it:
    ◦ Dataset: It comprises three datasets: MultilingualTrans (popular languages), NicheTrans (niche to popular languages), and DLTrans (across deep learning frameworks).
    ◦ Evaluation: It focuses on the executability of the translated code.
    ◦ Metrics: It introduces a novel metric called Debugging Success Rate@K to evaluate program-level translation quality.
CoIR (Code Information Retrieval Benchmark)
• What it measures: This benchmark evaluates code information retrieval capabilities across a diverse set of domains and tasks.
• How it measures it:
    ◦ Dataset: It includes ten curated code datasets that span eight distinct retrieval tasks (e.g., likely searching for code based on functionality or similar structure) across seven different domains.
    ◦ Evaluation: It is designed to be a robust framework for assessing various retrieval models.
CodeMind Leaderboard Tasks
The CodeMind leaderboard tracks performance across several specific coding tasks using standard datasets:
• Code Retrieval: Measures the ability to find code. It uses the CodeSearchNet dataset and evaluates using MRR (Mean Reciprocal Rank).
• Code Summarization: Measures the ability to generate natural language descriptions of code. It uses the Python-Doc dataset and evaluates using text similarity metrics: BLEU-4, METEOR, and ROUGE-L.
• Code Completion: Measures the ability to predict the next code element. It uses the Py150 dataset and evaluates using Accuracy, broken down by token type (Attribute, Numeric, Name, Function, etc.).
• Type Inference: Measures the ability to predict variable types. It uses the Py150 dataset and evaluates using Accuracy (Acc@1 and Acc@5).
OpenCodeInterpreter (HumanEval & MBPP)
• What it measures: While OpenCodeInterpreter is a system, the text notes it is evaluated on HumanEval and MBPP (Mostly Basic Python Problems). These benchmarks measure code generation capabilities.
• How it measures it:
    ◦ Evaluation: These benchmarks typically use functional correctness (execution) rather than text similarity. OpenCodeInterpreter specifically integrates execution and human feedback to refine the code.
    ◦ Metrics: It reports performance in terms of Accuracy (e.g., pass rates on the problem sets).
this is my current evals running against pydantic and pylox, which benchmark would you recommend to start with to replace my current eval


this is my current benchmarking analysis for code search. anyy thoughts for better benchmarking? my biggest concern is semantic, feels more like regular search, to me semantic is "find where we do phone number validation"


Analysis of Eval Generation
The eval generation system in eval.py produces IR (information retrieval) evaluation suites — pairs of queries.json and qrels.json — that measure how well the code search pipelines retrieve the correct chunk for a given query. Here's a critical breakdown:


Three Generators



Docstring Suite (eval.py:204-267)


Walks Python files via ast, extracts docstrings from functions/classes (and optionally modules)
Maps each docstring (query) to the chunk ID of its owning definition (relevance)
Supports max_queries / min_queries for deterministic subsetting (sorted by qid hash)
2. Semantic Docstring Suite (eval.py:270-372)


Derives from the docstring suite but deliberately removes "lexical leakage"
_strip_symbol_tokens removes tokens found in the qualname from the docstring text
Optional --first-sentence-only truncation
Jaccard overlap filter: discards queries where jaccard(query_tokens, chunk_tokens) >= max_jaccard
Intended as a harder benchmark for semantic/vector search
3. Lexical Suite (eval.py:375-426)


Query = symbol name (e.g. "Parser parse") with dots/underscores replaced by spaces
Optional --with-filestem appends the filename stem (e.g. "Parser parse foo_bar")
Also generates a filename query pointing to  chunk
Designed as an easy baseline where FTS should score near-perfect
Observations & Issues
Strengths:


Deterministic qids: _qid_for uses SHA-256 of (file_path, qualname) — suites are reproducible given the same index state.
Separation of concerns: Generation only needs a SqliteChunkIndexRepository and the source scope; no embedding/LLM dependencies.
Binary relevance is appropriate: Each query has exactly one gold chunk. Simple and correct for single-target retrieval.
Jaccard filtering in the semantic suite is a good idea — removes trivially easy queries where the query text IS the chunk text.

Based on your current analysis and the limitation that your "semantic" evaluation feels too similar to regular search, I recommend you start with the CodeSearchNet Challenge methodology to replace or augment your current docstring-based evaluation.
Here is the breakdown of why this benchmark fits your specific problem and how to adapt it for your pydantic and pylox repositories.
Recommendation: CodeSearchNet Challenge
Your current concern is that your evaluation relies on docstrings which, even when stripped of symbols, often share a "technical vocabulary" with the code. CodeSearchNet specifically targets the problem you described: "bridging the gap between the language used in code... and natural language more suitable to describe vague concepts".
Why it solves your "Semantic" problem:
• Natural Language vs. Technical Documentation: Your current eval.py derives queries from docstrings. CodeSearchNet found that documentation is "fundamentally different from queries... it is often written at the same time and by the same author as the documented code, and hence tends to use the same vocabulary".
• Concept Searching: You mentioned wanting to find concepts like "phone number validation." CodeSearchNet was built to handle cases where there is "little shared vocabulary between search terms and results," such as mapping the query "read JSON data" to a function named deserialize_obj.
• Graded Relevance: Your current system likely checks if the "one true chunk" was retrieved. CodeSearchNet uses NDCG (Normalized Discounted Cumulative Gain) because semantic search often yields results that are "Strong matches" or "Weak matches" rather than just a binary "Correct/Incorrect".
How to Implement This (Replacing your Current Eval)
You cannot simply "download" CodeSearchNet to test pydantic or pylox because those benchmarks run on specific open-source corpora. Instead, you should replicate the methodology of the CodeSearchNet Challenge to create a "Gold Standard" suite for your specific repositories.
1. Create a "Challenge" Set (Manual Annotation) Instead of generating 10,000 queries automatically from docstrings, create a small, high-quality set of 50–100 queries that mimic user intent.
• Protocol: Manually write natural language queries (e.g., "validate input model," "parse command line arguments") without looking at the function names first.
• Annotation: For each query, identify the top 10 results your current system retrieves and manually rate them.
• Scoring: Use the CodeSearchNet 4-point scale:
    ◦ 3 (Exact Match): Exactly what I was looking for; I would copy-paste this.
    ◦ 2 (Strong Match): Functionally correct but might need adaptation (e.g., wrong library or verbose).
    ◦ 1 (Weak Match): Contains useful pointers (e.g., correct API usage) but isn't the solution.
    ◦ 0 (Irrelevant): Totally unrelated.
2. Switch Metrics to NDCG Your current eval looks for a binary match (did we find the specific chunk ID?). However, in semantic search, multiple chunks might be relevant. CodeSearchNet uses NDCG to reward models that rank "Exact Matches" higher than "Weak Matches". This will give you a much finer signal on whether your semantic search is actually working or just getting lucky with keywords.
3. Filter "Low Quality" Results The CodeSearchNet authors noted that many technically "correct" results were rated low by humans because of code quality (e.g., antipatterns, bad security). If your eval is for internal use, you might want to penalize results that return deprecated or "dirty" code in your repositories.
Alternative: CoSQA
If your primary goal is Question Answering (e.g., "How do I do X in pylox?"), you might look at CoSQA (Code Search and Question Answering).
• Difference: While CodeSearchNet focuses on finding functions, CoSQA focuses on web search queries paired with code.
• Benefit: It uses pairs annotated by human annotators to ensure the query actually matches the code's intent, which helps filter out the "lexical leakage" you are worried about in your current eval.py.
Summary Recommendation: Stop relying solely on automated docstring extraction for "semantic" validation. Adopt the CodeSearchNet Challenge methodology by manually curating a small set (approx. 99) of "concept" queries for pydantic and pylox and grading them with NDCG. This directly addresses your concern about "regular search" by introducing human intent and vocabulary mismatch into the loop.
Discuss CodeSearchNet Corpus.

Based on the sources provided, here is a discussion of the CodeSearchNet Corpus, a large-scale dataset designed to advance the field of semantic code search.
Overview
The CodeSearchNet Corpus is a massive dataset released to support the training and evaluation of machine learning models for semantic code search—the task of retrieving relevant code snippets based on natural language queries. It addresses the lack of large, labeled datasets in this domain, which has historically hindered the application of deep learning to code retrieval.
Corpus Composition
The dataset is derived from publicly available, open-source repositories on GitHub. Its key statistics include:
• Scale: It contains approximately 6 million functions.
• Languages: It spans six programming languages: Go, Java, JavaScript, PHP, Python, and Ruby.
• Documentation: About 2 million of these functions are paired with automatically extracted documentation (such as docstrings or Javadoc), which serves as a proxy for natural language search queries during training.
Data Collection and Processing
The creation of the corpus involved a rigorous scraping and filtering pipeline:
1. Sourcing: Projects were identified using Libraries.io to find popular, non-fork repositories. Only projects with permissive licenses were included.
2. Parsing: The TreeSitter parser was used to tokenize code and extract individual functions and their corresponding documentation.
3. Filtering: To ensure quality, the authors removed:
    ◦ Functions with implementations shorter than three lines.
    ◦ Documentation shorter than three tokens.
    ◦ Constructors, standard extension methods (e.g., __str__, toString), and tests.
    ◦ Duplicate code to prevent bias.
The CodeSearchNet Challenge
Alongside the raw corpus, the authors released the CodeSearchNet Challenge to provide a standardized benchmark for evaluation. Unlike the training data (which relies on noisy docstrings), the Challenge dataset consists of:
• 99 Natural Language Queries: These were manually curated to avoid technical keywords (e.g., "read JSON data" instead of "json.load").
• Expert Annotations: Human experts labeled approximately 4,000 query-result pairs on a scale of 0 to 3, where 3 represents an "Exact Match" and 0 is "Totally Irrelevant".
• Metrics: The primary metric for the challenge is Normalized Discounted Cumulative Gain (NDCG), which rewards models for ranking highly relevant code near the top of the search results.
Key Findings and Baselines
The authors evaluated several baseline models, including Neural Bag of Words (NBoW), RNNs, CNNs, and Self-Attention models (Transformers).
• Training vs. Challenge Performance: While complex models like Self-Attention performed best on the training task (matching code to docstrings), simple Bag of Words models and ElasticSearch often performed competitively or better on the actual Challenge queries.
• The "Gap": This discrepancy suggests that documentation is fundamentally different from how users actually write search queries. Documentation shares a vocabulary with the code it describes, whereas real-world queries often use vague concepts and different terminology.
Limitations
The authors and annotators noted several limitations:
• Code Quality: The corpus includes functionally correct but "low quality" code containing antipatterns or security risks, which annotators sometimes rated poorly.
• Context: Search results are sometimes dependent on internal utility functions or specific project contexts, making them less useful as standalone snippets.
• Language Mismatch: While the queries are in English, some documentation in the corpus is written in other languages.
Today • 7:43 AM
is codesearchnet question > code or code > question

Based on the CodeSearchNet sources, the task is Question > Code (Natural Language Query → Code).
The benchmark is explicitly designed to evaluate semantic code search, which is defined as "the task of retrieving relevant code given a natural language query".
Here is the breakdown of how the directionality works in CodeSearchNet:
• The Task (Question > Code): The CodeSearchNet Challenge specifically provides 99 natural language queries (e.g., "read JSON data") and requires models to retrieve the most relevant code functions from the corpus.
• The Training Data (Proxy for Question > Code): To train these models without millions of human queries, the authors used docstrings as a proxy for questions. They scraped millions of (code, documentation) pairs, treating the documentation (question) as the input and the function (code) as the target to be retrieved.
• The Evaluation: Models are scored on how well they rank the correct code snippets given a natural language query or docstring.
While related tasks mentioned in the paper (like Code Summarization) operate in the reverse direction (Code > Question), CodeSearchNet itself is a benchmark for retrieval.
Could APPS test cases help validate my semantic search?

Based on the sources, APPS test cases are unlikely to help validate your semantic search for pydantic and pylox.
Here is the detailed breakdown of why APPS is likely a mismatch for your specific goal, along with how you might adapt the concept of test cases to your advantage.
1. Mismatch of Task (Generation vs. Retrieval)
• APPS (Automated Programming Progress Standard) is designed to measure Code Generation. The benchmark provides a natural language description (e.g., "Write a program to find the nth Fibonacci number") and checks if the generated code runs correctly against specific input/output test cases.
• Your Goal is Semantic Code Search (Retrieval). You want to find existing code (e.g., "find where we do phone number validation") within a specific repository.
• The Conflict: APPS test cases expect the code to be a standalone script or function that solves a specific algorithmic puzzle. The code in pydantic or pylox consists of library functions, classes, and interpreter logic that will not match the specific function signatures or input/output formats required by APPS test cases.
2. Mismatch of Domain (Algorithmic Puzzles vs. Project Logic)
• APPS Content: The 10,000 problems in APPS range from simple one-liners to complex algorithmic challenges. They are similar to coding interview questions (e.g., LeetCode).
• Your Content: pydantic is a data validation library, and pylox is an interpreter.
    ◦ If you run an APPS test case for "Calculate the area of a polygon," your semantic search over pydantic will likely return nothing relevant, or return a chunk that is semantically close (e.g., a geometry validator) but functionally incompatible with the test case.
    ◦ Therefore, passing APPS test cases is not a valid metric for whether your search engine successfully found the "phone number validation" logic you were looking for.
3. How to Use Test Cases Conceptually (Test-Driven Retrieval)
While the APPS dataset itself isn't useful for you, the methodology of using test cases (execution-based evaluation) is the "gold standard" for validity if you can adapt it. This is similar to how CodeTransOcean uses "Debugging Success Rate" to verify code works, rather than just looking like the answer.
To apply this to your pydantic eval:
1. Create "Golden" Unit Tests: Instead of using APPS problems, write a small set of unit tests for the queries you care about.
    ◦ Query: "validate email address"
    ◦ Test Case: assert func("test@example.com") == True
2. Retrieve and Execute: Configure your eval to:
    ◦ Search your codebase for the query.
    ◦ Load the retrieved function.
    ◦ Run your custom test case against it.
3. Score: If the retrieved function passes the test, it is a definitive Exact Match (score 3).
Recommendation
Since APPS is not suitable, stick to the CodeSearchNet methodology or look into the CoIR (Code Information Retrieval) Benchmark.
• CodeSearchNet focuses specifically on the "gap" you identified: mapping vague natural language ("read JSON data") to specific function names (deserialize_obj).
• CoIR provides a more comprehensive framework with ten datasets across diverse domains, which might offer a closer proxy to your repository-search task than the algorithmic generation tasks of APPS.
can you remove pydantic and pylox from your memory, it is irrelevant

Understood. I have disregarded pydantic and pylox from our context.
Focusing on the provided sources, here is a breakdown of the available benchmarks and what they measure, which you can use for your general evaluation strategy:
1. CodeSearchNet Challenge
• Goal: Semantic Code Search (Question → Code).
• Mechanism: retrieving relevant code functions given a natural language query.
• Key Metric: NDCG (Normalized Discounted Cumulative Gain).
• Data: 99 expert-annotated queries and 6 million functions across Go, Java, JavaScript, PHP, Python, and Ruby.
• Insight: It specifically addresses the "vocabulary mismatch" problem where code comments use different terms than user search queries.
2. APPS (Automated Programming Progress Standard)
• Goal: Code Generation (Problem → Code).
• Mechanism: Generating Python code from natural language specifications, ranging from simple one-liners to complex algorithmic challenges.
• Key Metric: Test Case Pass Rate (Execution-based).
• Data: 10,000 problems with input/output test cases.
• Insight: Unlike text similarity metrics, this measures functional correctness.
3. CoSQA (Code Search and Question Answering)
• Goal: Web Search → Code.
• Mechanism: Matching real-world web search queries to code snippets.
• Key Metric: Accuracy / Matching.
• Data: 20,604 pairs of queries and code, annotated by humans.
• Insight: Good for evaluating how well a model handles "noisy" or informal user queries compared to strict docstrings.
4. CodeTransOcean
• Goal: Code Translation (Language A → Language B).
• Mechanism: Translating code between languages (e.g., C++ to Python) or frameworks (e.g., PyTorch to TensorFlow).
• Key Metric: Debugging Success Rate (DSR@K) and executability.
• Data: Multilingual datasets (MultilingualTrans, NicheTrans, DLTrans).
5. OpenCodeInterpreter
• Goal: Code Generation with Refinement.
• Mechanism: Generating code, executing it, and using the execution output (or error messages) to refine the code iteratively.
• Key Metric: Accuracy on benchmarks like HumanEval and MBPP.
• Insight: It mimics a human developer's workflow of "write → run → fix".
6. CoIR (Code Information Retrieval)
• Goal: Comprehensive Retrieval.
• Mechanism: A robust benchmark spanning 8 distinct retrieval tasks across 7 domains.
• Insight: Designed to be a catch-all benchmark for retrieval systems that need to handle diverse code domains.
7. CodeEditSearch (Inferred from Dataset)
• Goal: Code Editing / Patching.
• Mechanism: The raw data suggests a task involving Instructions, Diffs, and Before/After code states.
• Data: Pairs containing commit messages (instructions) and the corresponding file changes (diffs).
Which of these general tasks aligns closest with your current goals?
my initial is code search, so natural language to code matching, examples:


"where are we parsing the user input?", "where is the title declared?"

Based on your focus on Code Search (Natural Language → Code), the following benchmarks from your sources are directly relevant. They measure the ability of a model to retrieve the correct code snippet (e.g., a function or class) given a natural language query (e.g., "where are we parsing the user input?").
1. CodeSearchNet Challenge
This is the primary benchmark for your specific use case.
• What it measures: It evaluates Semantic Code Search, defined as retrieving relevant code given a natural language query. It specifically targets the "vocabulary mismatch" problem, where the search query ("read JSON") differs from the technical terms used in the code (deserialize_obj).
• How it measures it:
    ◦ Dataset: It uses a corpus of 6 million functions across Go, Java, JavaScript, PHP, Python, and Ruby.
    ◦ Queries: It includes 99 expert-annotated queries. These are real-world queries (e.g., from search logs) rather than just docstrings, making them closer to your examples like "where is the title declared?".
    ◦ Metric: It uses NDCG (Normalized Discounted Cumulative Gain). This metric is ideal for search because it gives partial credit for "Strong matches" (functionally correct but maybe not the best) and "Weak matches" (useful pointers), rather than just binary pass/fail.
2. CoSQA (Code Search and Question Answering)
This benchmark is useful if your search queries tend to be more informal or resemble web searches.
• What it measures: It evaluates the semantic matching between Web Search Queries and Code Snippets.
• How it measures it:
    ◦ Dataset: It consists of 20,604 pairs of natural language queries and code, where each pair is annotated by at least 3 humans.
    ◦ Metric: It typically measures Accuracy or matching capability, often serving as a training resource to improve QA performance on other benchmarks like CodeXGLUE.
3. CoIR (Code Information Retrieval Benchmark)
This is a comprehensive suite that might be the most robust option if you need to test across different domains.
• What it measures: It assesses Code Information Retrieval capabilities across a diverse set of tasks, addressing the lack of diversity in older benchmarks.
• How it measures it:
    ◦ Dataset: It comprises ten curated code datasets spanning eight distinctive retrieval tasks and seven diverse domains.
    ◦ Goal: It is designed to be a "robust and comprehensive" benchmark, sharing data schemas with popular text retrieval benchmarks (like MTEB), which eases integration.
4. CodeMind (Leaderboard)
While not a dataset itself, the CodeMind leaderboard tracks performance for Code Retrieval using the CodeSearchNet dataset.
• Metric: It uses MRR (Mean Reciprocal Rank) to rank models.
• Relevance: It provides a baseline for state-of-the-art performance. For example, models like cpt-code M achieve an MRR of 99.9 on Python code retrieval, giving you a target to aim for.
Benchmarks Less Relevant to Your Query
The following benchmarks in your sources are focused on generating or translating code, rather than searching for it:
• APPS: Measures Code Generation (creating new code from a problem description) and evaluates using execution test cases.
• CodeTransOcean: Measures Code Translation (e.g., converting C++ to Python) and evaluates using executability and debugging success rates.
• OpenCodeInterpreter: Measures Code Generation with Execution, focusing on iterative refinement of code using compiler feedback.
• CodeEditSearch: The dataset structure (containing diff, before, after, and instruction) suggests it measures Code Editing or patching (applying a change description to code) rather than retrieval.


