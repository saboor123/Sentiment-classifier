# Sentiment-Analysis-Twitter


Microblogging today has become a very popular communication tool among Internet users. Millions of messages are appearing daily in popular web-sites that provide services for microblogging such as Twitter, Tumblr, Facebook. Authors of those messages write about their life, share opinions on variety of topics and discuss current issues. Because of a free format of messages and an easy accessibility
of microblogging platforms, Internet users tend to shift from traditional communication tools (such as traditional blogs or mailing lists) to microblogging services. As more and more users post about products and services they use, or express their political and religious views, microblogging web-sites become valuable sources of people’s opinions and sentiments. Such data can be efficiently used
for marketing or social studies.[1]


### 1.2 Characteristic features of Tweets 

From the perspective of Sentiment
Analysis, we discuss a few characteristics of Twitter:

**Length of a Tweet**
     The maximum length of a Twitter message is 140 characters. This means that we can practically consider a tweet to be a single sentence, void of complex grammatical constructs. This is a vast difference from traditional subjects of Sentiment Analysis, such as movie reviews. 
     
**Language used**
     Twitter is used via a variety of media including SMS and mobile phone apps. Because of this and the 140-character limit, language used in Tweets tend be more colloquial, and filled with slang and misspellings. Use of hashtags also gained popularity on Twitter and is a primary feature in any given tweet. Our analysis shows that there are approximately 1-2 hashtags per tweet, as shown in Table 3 . 
     
**Data availability**
     Another difference is the magnitude of data available. With the Twitter API, it is easy to collect millions of tweets for training. There also exist a few datasets that have automatically and manually labelled the tweets [2] [3]. 
     
**Domain of topics**
     People often post about their likes and dislikes on social media. These are not al concentrated around one topic. This makes twitter a unique place to model a generic classifier as opposed to domain specific classifiers that could be build datasets such as movie reviews. 
     
    

##  3  Approach

We use different feature sets and machine learning classifiers to determine
the best combination for sentiment analysis of twitter. We also experiment
with various pre-processing steps like - punctuations, emoticons, twitter
specific terms and stemming. We investigated the following features -
unigrams, bigrams, trigrams and negation detection. We finally train our
classifier using various machine-learning algorithms - Naive Bayes, Decision
Trees and Maximum Entropy.


We use a modularized approach with feature extractor and classification
algorithm as two independent components. This enables us to experiment with
different options for each component.

###  3.1  Datasets

One of the major challenges in Sentiment Analysis of Twitter is to collect a
labelled dataset. Researchers have made public the following datasets for
training and testing classifiers.

####  3.1.1  Twitter Sentiment Corpus

This is a collection of 5513 tweets collected for four different topics,
namely, Apple, Google, Microsoft, Twitter It is collected and hand-classified
by Sanders Analytics LLC. Each entry in the corpus contains, Tweet id,
Topic and a Sentiment label. We use Twitter-Python library to enrich this data
by downloading data like Tweet text, Creation Date, Creator etc. for every
Tweet id. Each Tweet is hand classified by an American male into the following
four categories. For the purpose of our experiments, we consider Irrelevant
and Neutral to be the same class. Illustration of Tweets in this corpus is
show in Table 1 .

- **Positive**
     For showing positive sentiment towards the topic
     
- **Positive**
     For showing no or mixed or weak sentiments towards the topic
     
- **Negative**
     For showing negative sentiment towards the topic
     
- **Irrelevant**
     For non English text or off-topic comments
     

<div style="text-align:center">
<table border="1">
<tr><td align="left">Class </td><td align="right">Count </td><td width="0">Example </td></tr>
<tr><td align="left">neg </td><td align="right">529 </td><td width="0">#Skype often crashing: #microsoft, what are you doing? </td></tr>
<tr><td align="left">neu </td><td align="right">3770 </td><td width="0">How #Google Ventures Chooses Which Startups Get Its $200
                Million http://t.co/FCWXoUd8 via @mashbusiness @mashable </td></tr>
<tr><td align="left">pos </td><td align="right">483 </td><td width="0">Now all @Apple has to do is get swype on the iphone and
                it will be crack. Iphone that is </td></tr></table>


<div class="p"><!----></div>
<div style="text-align:center">Table 1: Twitter Sentiment Corpus</div>
<a id="tab:TSC">
</a>
</div>

####  3.1.2  Stanford Twitter

This corpus of tweets, developed by Sanford’s Natural Language processing
research group, is publically available. The training set is collected by
querying Twitter API for happy emoticons like "`:)`" and sad emoticons like
"`:(`" and labelling them positive or negative. The emoticons were then
stripped and Re-Tweets and duplicates removed. It also contains around 500
tweets manually collected and labelled for testing purposes. We randomly
sample and use 5000 tweets from this dataset. An example of Tweets in this
corpus are shown in Table 2 .

<div style="text-align:center">
<table border="1">
<tr><td align="left">Class </td><td align="right">Count </td><td width="0">Example </td></tr>
<tr><td align="left">neg </td><td align="right">2501 </td><td width="0">Playing after the others thanks to TV scheduling may well allow us to know what's go on, but it makes things look bad on Saturday nights  </td></tr>
<tr><td align="left">pos </td><td align="right">2499 </td><td width="0">@francescazurlo HAHA!!! how long have you been singing that song now? It has to be at least a day. i think you're wildly entertaining!  </td></tr></table>


<div class="p"><!----></div>
<div style="text-align:center">Table 2: Stanford Corpus</div>
<a id="tab:STAN">
</a>
</div>

###  3.2  Pre Processing

User-generated content on the web is seldom present in a form usable for
learning. It becomes important to normalize the text by applying a series of
pre-processing steps. We have applied an extensive set of pre-processing steps
to decrease the size of the feature set to make it suitable for learning
algorithms. Figure 2 illustrates various features seen in micro-blogging.
Table 3 illustrates the frequency of these features per tweet, cut by
datasets. We also give a brief description of pre-processing steps taken.

####  3.2.1  Hashtags

A hashtag is a word or an un-spaced phrase prefixed with the hash symbol (#).
These are used to both naming subjects and phrases that are currently in
trending topics. For example, #iPad, #news

Regular Expression: `#(\w+)`

Replace Expression: `HASH_\1`

####  3.2.2  Handles

Every Twitter user has a unique username. Any thing directed towards that user
can be indicated be writing their username preceded by ‘@’. Thus, these are
like proper nouns. For example, @Apple

Regular Expression: `@(\w+)`

Replace Expression: `HNDL_\1`

####  3.2.3  URLs

Users often share hyperlinks in their tweets. Twitter shortens them using its
in-house URL shortening service, like http://t.co/FCWXoUd8 - such links also
enables Twitter to alert users if the link leads out of its domain. From the
point of view of text classification, a particular URL is not important.
However, presence of a URL can be an important feature. Regular expression for
detecting a URL is fairly complex because of different types of URLs that can
be there, but because of Twitter’s shortening service, we can use a relatively
simple regular expression.

Regular Expression: `(http|https|ftp)://[a-zA-Z0-9\\./]+`

Replace Expression: `URL`

####  3.2.4  Emoticons

Use of emoticons is very prevalent throughout the web, more so on micro-
blogging sites. We identify the following emoticons and replace them with a
single word. Table 4 lists the emoticons we are currently detecting. All other
emoticons would be ignored.

<div style="text-align:center"> 
<table border="1">
<tr><td colspan="1" align="center">Emoticons </td><td colspan="6" align="center">Examples </td></tr>
<tr><td align="left"><tt>EMOT_SMILEY</tt>   </td><td align="left"><tt>:-)</tt>  </td><td align="left"><tt>:)</tt>   </td><td align="left"><tt>(:</tt>   </td><td align="left"><tt>(-:</tt>  </td><td align="left"><tt></tt>     </td><td align="left"><tt></tt> </td></tr>
<tr><td align="left"><tt>EMOT_LAUGH</tt>    </td><td align="left"><tt>:-D</tt>  </td><td align="left"><tt>:D</tt>   </td><td align="left"><tt>X-D</tt>  </td><td align="left"><tt>XD</tt>   </td><td align="left"><tt>xD</tt>   </td><td align="left"><tt></tt> </td></tr>
<tr><td align="left"><tt>EMOT_LOVE</tt>     </td><td align="left"><tt>&lt;3</tt>    </td><td align="left"><tt>:*</tt>   </td><td align="left"><tt></tt>     </td><td align="left"><tt></tt>     </td><td align="left"><tt></tt>     </td><td align="left"><tt></tt> </td></tr>
<tr><td align="left"><tt>EMOT_WINK</tt>     </td><td align="left"><tt>;-)</tt>  </td><td align="left"><tt>;)</tt>   </td><td align="left"><tt>;-D</tt>  </td><td align="left"><tt>;D</tt>   </td><td align="left"><tt>(;</tt>   </td><td align="left"><tt>(-;</tt> </td></tr>
<tr><td align="left"><tt>EMOT_FROWN</tt>    </td><td align="left"><tt>:-(</tt>  </td><td align="left"><tt>:(</tt>   </td><td align="left"><tt>(:</tt>   </td><td align="left"><tt>(-:</tt>  </td><td align="left"><tt></tt>     </td><td align="left"><tt></tt> </td></tr>
<tr><td align="left"><tt>EMOT_CRY</tt>  </td><td align="left"><tt>:,(</tt>  </td><td align="left"><tt>:'(</tt>  </td><td align="left"><tt>:"(</tt>  </td><td align="left"><tt>:((</tt>  </td><td align="left"><tt></tt>     </td><td align="left"><tt></tt> </td></tr></table>


<div style="text-align:center">Table 4: List of Emoticons</div>
<a id="tab:emot">
</a>
</div>

####  3.2.5  Punctuations

Although not all Punctuations are important from the point of view of
classification but some of these, like question mark, exclamation mark can
also provide information about the sentiments of the text. We replace every
word boundary by a list of relevant punctuations present at that point. Table
5 lists the punctuations currently identified. We also remove any single
quotes that might exist in the text.

<div style="text-align:center"> 
<table border="1">
<tr><td colspan="1" align="center">Punctuations </td><td colspan="2" align="center">Examples </td></tr>
<tr><td align="left"><tt>PUNC_DOT</tt> </td><td align="left"><tt>.</tt> </td><td align="left"><tt></tt> </td></tr>
<tr><td align="left"><tt>PUNC_EXCL</tt> </td><td align="left"><tt>!</tt> </td><td align="left"><tt>¡</tt> </td></tr>
<tr><td align="left"><tt>PUNC_QUES</tt> </td><td align="left"><tt>?</tt> </td><td align="left"><tt>¿</tt> </td></tr>
<tr><td align="left"><tt>PUNC_ELLP</tt> </td><td align="left"><tt>...</tt> </td><td align="left"><tt>…</tt> </td></tr></table>


<div style="text-align:center">Table 5: List of Punctuations</div>
<a id="tab:punc">
</a>
</div>

####  3.2.6  Repeating Characters

People often use repeating characters while using colloquial language, like
"I’m in a hurrryyyyy", "We won, yaaayyyyy!" As our final pre-processing step,
we replace characters repeating more than twice as two characters.

Regular Expression: `(.)\1{1,}`

Replace Expression: `\1\1`

###  3.3  Stemming Algorithms

All stemming algorithms are of the following major types – affix removing,
statistical and mixed. The first kind, Affix removal stemmer, is the most
basic one. These apply a set of transformation rules to each word in an
attempt to cut off commonly known prefixes and / or suffixes [8]. A trivial
stemming algorithm would be to truncate words at N-th symbol. But this
obviously is not well suited for practical purposes.

J.B. Lovins described first stemming algorithm in 1968. It defines 294
endings, each linked to one of 29 conditions, plus 35 transformation rules.
For a word being stemmed, an ending with a satisfying condition is found and
removed. Another famous stemmer used extensively is described in the next
section.

####  3.3.1  Porter Stemmer

Martin Porter wrote a stemmer that was published in July 1980. This stemmer
was very widely used and became and remains the de facto standard algorithm
used for English stemming. It offers excellent trade-off between speed,
readability, and accuracy. It uses a set of around 60 rules applied in 6
successive steps [9]. An important feature to note is that it doesn’t involve
recursion. The steps in the algorithm are described in Table 7 .

<div style="text-align:center">
<table border="1">
<tr><td align="right">1.    </td><td align="left">Gets rid of plurals and -ed or -ing suffixes </td></tr>
<tr><td align="right">2.    </td><td align="left">Turns terminal y to i when there is another vowel in the stem￼ </td></tr>
<tr><td align="right">3.    </td><td align="left">Maps double suffixes to single ones: -ization, -ational, etc. </td></tr>
<tr><td align="right">4.    </td><td align="left">Deals with suffixes, -full, -ness etc. </td></tr>
<tr><td align="right">5.￼   </td><td align="left">Takes off -ant, -ence, etc. </td></tr>
<tr><td align="right">6.    </td><td align="left">Removes a final –e </td></tr></table>


<div style="text-align:center">Table 7: Porter Stemmer Steps</div>
<a id="tab:porter">
</a>
</div>

####  3.3.2  Lemmatization

Lemmatization is the process of normalizing a word rather than just finding
its stem. In the process, a suffix may not only be removed, but may also be
substituted with a different one. It may also involve first determining the
part-of-speech for a word and then applying normalization rules. It might also
involve dictionary look-up. For example, verb ‘saw’ would be lemmatized to
‘see’ and the noun ‘saw’ will remain ‘saw’. For our purpose of classifying
text, stemming should suffice.

###  3.4  Features

A wide variety of features can be used to build a classifier for tweets. The
most widely used and basic feature set is word n-grams. However, there's a lot
of domain specific information present in tweets that can also be used for
classifying them. We have experimented with two sets of features:

####  3.4.1  Unigrams

Unigrams are the simplest features that can be used for text classification. A
Tweet can be represented by a multiset of words present in it. We, however,
have used the presence of unigrams in a tweet as a feature set. Presence of a
word is more important than how many times it is repeated. Pang et al. found
that presence of unigrams yields better results than repetition [1]. This also
helps us to avoid having to scale the data, which can considerably decrease
training time [2]. Figure 3 illustrated the cumulative distribution of words
in our dataset.

![Figure](http://i.imgur.com/o5QZwnn.png)

Figure 3: Cumulative Frequency Plot for 50 Most Frequent Unigrams

We also observe that the unigrams nicely follow Zipf’s law. It states that in
a corpus of natural language, the frequency of any word is inversely
proportional to its rank in the frequency table. Figure 4 is a plot of log
frequency versus log rank of our dataset. A linear trendline fits well with
the data.

####  3.4.2  N-grams

N-gram refers to an n-long sequence of words. Probabilistic Language Models
based on Unigrams, Bigrams and Trigrams can be successfully used to predict
the next word given a current context of words. In the domain of sentiment
analysis, the performance of N-grams is unclear. According to Pang et al.,
some researchers report that unigrams alone are better than bigrams for
classification movie reviews, while some others report that bigrams and
trigrams yield better product-review polarity classification [1].

As the order of the n-grams increases, they tend to be more and more sparse.
Based on our experiments, we find that number of bigrams and trigrams increase
much more rapidly than the number of unigrams with the number of Tweets.
Figure 4 shows the number of n-grams versus number of Tweets. We can observe
that bigrams and trigrams increase almost linearly where as unigrams are
increasing logarithmically.

![Figure](http://i.imgur.com/j0TyDow.png)

Figure 4: Number of n-grams vs. Number of Tweets

Because higher order n-grams are sparsely populated, we decide to trim off the
n-grams that are not seen more than once in the training corpus, because
chances are that these n-grams are not good indicators of sentiments. After
the filtering out non-repeating n-grams, we see that the number of n-grams is
considerably decreased and equals the order of unigrams, as shown in Figure 5
.

![Figure](http://i.imgur.com/JZZ5OPI.png)

Figure 5: Number of repeating n-grams vs. Number of Tweets



####  3.4.3  Negation Handling

The need negation detection in sentiment analysis can be illustrated by the
difference in the meaning of the phrases, "This is good" vs. "This is not
good" However, the negations occurring in natural language are seldom so
simple. Handling the negation consists of two tasks – Detection of explicit
negation cues and the scope of negation of these words.

Councill et al. look at whether negation detection is useful for sentiment
analysis and also to what extent is it possible to determine the exact scope
of a negation in the text [7]. They describe a method for negation detection
based on Left and Right Distances of a token to the nearest explicit negation
cue.

#### Scope of Negation

Words immediately preceding and following the negation cues are the most
negative and the words that come farther away do not lie in the scope of
negation of such cues. We define left and right negativity of a word as the
chances that meaning of that word is actually the opposite. Left negativity
depends on the closest negation cue on the left and similarly for Right
negativity. Figure 7 illustrates the left and right negativity of words in a
tweet.

![Figure](http://i.imgur.com/QhtlwPb.png)

Figure 7: Scope of Negation

##  4  Experimentation

We train 90% of our data using different combinations of features and test
them on the remaining 10%. We take the features in the following combinations
- only unigrams, unigrams + filtered bigrams and trigrams, unigrams +
negation, unigrams + filtered bigrams and trigrams + negation. We then train
classifiers using different classification algorithms - Naive Bayes Classifier
and Maximum Entropy Classifier.

The task of classification of a tweet can be done in two steps - first,
classifying "neutral" (or "subjective") vs. "objective" tweets and second,
classifying objective tweets into "positive" vs. "negative" tweets. We also
trained 2 step classifiers. The accuracies for each of these configuration are
shown in Figure 8 , we discuss these in detail below.

![Figure](http://i.imgur.com/TfYr9Se.png)

Figure 8: Accuracy for Naive Bayes Classifier

###  4.1  Naive Bayes

Naive Bayes classifier is the simplest and the fastest classifier. Many
researchers [2], [4] claim to have gotten best results using this classifier.

For a given tweet, if we need to find the label for it, we find the
probabilities of all the labels, given that feature and then select the label
with maximum probability.

The results from training the Naive Bayes classifier are shown below in Figure
8 . The accuracy of Unigrams is the lowest at 79.67%. The accuracy increases
if we also use Negation detection (81.66%) or higher order n-grams (86.68%).
We see that if we use both Negation detection and higher order n-grams, the
accuracy is marginally less than just using higher order n-grams (85.92%). We
can also note that accuracies for double step classifier are lesser than those
for corresponding single step.

We have also shown Precision versus Recall values for Naive Bayes classifier
corresponding to different classes – Negative, Neutral and Positive in Figure
9 . The solid markers show the P-R values for single step classifier and
hollow markers show the affect of using double step classifier. Different
points are for different feature sets. We can see that both precision as well
as recall values are higher for single step than that for double step.

![Figure](http://i.imgur.com/h2IReTP.png)

Figure 9: Precision vs. Recall for Naive Bayes Classifier

##  5  Future Work

**Investigating Support Vector Machines**
     Several papers have discussed the results using Support Vector Machines (SVMs) also. The next step would be to test our approach on SVMs. However, Go, Bhayani and Huang have reported that SVMs do not increase the accuracy [2]. 
     
**Building a classifier for Hindi tweets**
     There are many users on Twitter that use primarily Hindi language. The approach discussed here can be used to create a Hindi language sentiment classifier. 
     
**Improving Results using Semantics Analysis**
     Understanding the role of the nouns being talked about can help us better classify a given tweet. For example, "Skype often crashing: microsoft, what are you doing?" Here Skype is a product and Microsoft is a company. We can use semantic labellers to achieve this. Such an approach is discussed by Saif, He and Alani [6]. 

##  6  Conclusion

We create a sentiment classifier for twitter using labelled
data sets. We also investigate the relevance of using a double step classifier
and negation detection for the purpose of sentiment analysis.

Our baseline classifier that uses just the unigrams achieves an accuracy of
around 80.00%. Accuracy of the classifier increases if we use negation
detection or introduce bigrams and trigrams. Thus we can conclude that both
Negation Detection and higher order n-grams are useful for the purpose of text
classification. However, if we use both n-grams and negation detection, the
accuracy falls marginally. We also note that Single step classifiers out
perform double step classifiers. In general, Naive Bayes Classifier performs
better than Maximum Entropy Classifier.

We achieve the best accuracy of 86.68% in the case of Unigrams + Bigrams +
Trigrams, trained on Naive Bayes Classifier.



## References
[1] Pak, Alexander, and Patrick Paroubek. "Twitter as a Corpus for Sentiment Analysis and Opinion Mining." LREc. Vol. 10. 2010.

[2] Alec Go, Richa Bhayani, and Lei Huang. Twitter sentiment classification using distant supervision. _Processing_, pages 1-6, 2009.

[3] Niek Sanders. Twitter sentiment corpus. http://www.sananalytics.com/lab/twitter-sentiment/. Sanders Analytics.

[4] Alexander Pak and Patrick Paroubek. Twitter as a corpus for sentiment analysis and opinion mining. volume 2010, pages 1320-1326, 2010. 

[5] Efthymios Kouloumpis, Theresa Wilson, and Johanna Moore. Twitter sentiment analysis: The good the bad and the omg! _ICWSM_, 11:pages 538-541, 2011. 

[6] Hassan Saif, Yulan He, and Harith Alani. Semantic sentiment analysis of twitter. In _The Semantic Web-ISWC 2012_, pages 508-524. Springer, 2012. 

[7] Isaac G Councill, Ryan McDonald, and Leonid Velikovich. What's great and what's not: learning to classify the scope of negation for improved sentiment analysis. In _Proceedings of the workshop on negation and speculation in natural language processing_, pages 51-59. Association for Computational Linguistics, 2010. 

[8] Ilia Smirnov. Overview of stemming algorithms. _Mechanical Translation_, 2008. 

[9] Martin F Porter. An algorithm for suffix stripping. _Program: electronic library and information systems_, 40(3):pages 211-218, 2006. 

[10] Balakrishnan Gokulakrishnan, P Priyanthan, T Ragavan, N Prasath, and A Perera. Opinion mining and sentiment analysis on a twitter data stream. In _Advances in ICT for Emerging Regions (ICTer), 2012 International Conference on. IEEE_, 2012. 

[11] John Ross Quinlan. _C4. 5: programs for machine learning_, volume 1\. Morgan kaufmann, 1993. 

[12] Steven Bird, Ewan Klein, and Edward Loper. _Natural language processing with Python_. " O'Reilly Media, Inc.", 2009.

