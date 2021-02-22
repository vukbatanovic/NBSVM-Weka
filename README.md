# NBSVM-Weka - a Java implementation of the multiclass NBSVM classifier for Weka
NBSVM is an algorithm, originally designed for binary text/sentiment classification, which combines the Multinomial Naive Bayes (MNB) classifier with the Support Vector Machine (SVM).
It does so through the element-wise multiplication of standard feature vectors by the positive class/negative class ratios of MNB log-counts. Such vectors are then used as inputs for the SVM classifier.

This implementation extends the [original algorithm](http://github.com/sidaw/nbsvm) to support multiclass classification using the one-vs-all approach.
When dealing with *N* classes the classifier calculates *N* distinct ratios of MNB log-counts by separately considering each class as 'positive' and all the other classes taken together as 'negative'.
These ratios are then combined with the input feature vectors of *N* separate discriminative classifiers.

## Dependencies
This package relies on the [LibLINEAR library](http://www.csie.ntu.edu.tw/~cjlin/liblinear/) and its [Weka wrapper class](http://github.com/bwaldvogel/liblinear-weka). The LibLINEAR Weka package must be installed in order for NBSVM-Weka to function.
Since it uses LibLINEAR, logistic regression can easily be used as the discriminative classifier instead of SVM (thereby maximizing the log-likelihood instead of the SVM margin).

Certain SVMTypes in the LibLINEAR package which are not compatible with (this implementation of) NBSVM, such as Support Vector Regression models or Crammer and Singer's multiclass classification algorithm, have been disabled.
However, for the sake of consistency, the numbering of options used in LibLINEAR has been preserved.

## Installation
NBSVM-Weka can be installed as an unofficial plug-in module within Weka.
To do so, download the [NBSVM-Weka package](https://github.com/vukbatanovic/NBSVM-Weka/releases/download/v1.0.1/NBSVM-Weka_1.0.1.zip).
Open the Weka package manager and use the "Unofficial - File/URL" option to select and install NBSVM-Weka.
After restarting Weka, the list of available classifiers (within the functions category) will contain the NBSVM option.

## Usage
The classifier can be used either through the Weka GUI or through the command line interface.
Either way, the configuration options are very similar to the options of LibLINEAR's wrapper for Weka.
One difference is that NBSVM-Weka is not compatible with Support Vector Regression and Crammer and Singer's multiclass classification algorithm, making those SVMTypes disabled.

In addition, NBSVM-Weka allows the user to specify the Laplace smoothing parameter alpha (*default: 1.0*) and the interpolation parameter beta (*default: 0.25*).
These parameters can be set in the command line options as -A <double> (for Laplace smoothing) and -X <double> (for interpolation). 

## References
If you wish to use the NBSVM classifier in your paper or project, please cite the original paper:

**[Baselines and Bigrams: Simple, Good Sentiment and Topic Classification](http://nlp.stanford.edu/pubs/sidaw12_simple_sentiment.pdf)**, Sida Wang, Christopher D. Manning, in Proceedings of the 50th Annual Meeting of the Association for Computational Linguistics (ACL 2012), pp. 90–94, Jeju Island, South Korea (2012).

This Weka package was introduced in the following paper:

**[Reliable Baselines for Sentiment Analysis in Resource-Limited Languages: The Serbian Movie Review Dataset](http://www.lrec-conf.org/proceedings/lrec2016/pdf/284_Paper.pdf)**, Vuk Batanović, Boško Nikolić, Milan Milosavljević, in Proceedings of the 10th International Conference on Language Resources and Evaluation (LREC 2016), pp. 2688-2696, Portorož, Slovenia (2016).

## Additional Documentation
All methods contain extensive documentation and comments.
If you have any questions about the classifier's functioning, please review the supplied javadoc documentation, the source code, and the papers listed above.
If no answer can be found, feel free to contact me at vuk.batanovic / at / ic.etf.bg.ac.rs

## License
GNU General Public License 3.0 (GNU GPL 3.0)
