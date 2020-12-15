# Ham-Spam-Naive-Bayesian-Classifier
The goal of this exercise is to create a basic spam detector using Naive Bayes classifier. We will
use the attached data set spam.csv. The data set contains a collection of SMS messages that
are labeled as either Spam or Ham. We think of each message as a set of tokens that appear
in the message and we assume that the probability of obtaining a set of tokens {t1, · · · , tk}
given a label y can be factorized as p({t1, t2, · · · , tk}|y) = p(t1|y) · p(t2|y)· · · p(tk|y).

Note that we split data set into training and test sets. You should train your classifier on
the training set and report your results for the test set. When classifying a new message,
we may see tokens that have not been found in the training set. To avoid 0 counts, we will
do additive smoothing by assuming that each unseen token has a default count of α = 0.1.
Also, instead of multiplying probabilities in p(t1|y)· p(t2|y)· · · p(tk|y) it is better to work with
log probabilities which can be added. 
