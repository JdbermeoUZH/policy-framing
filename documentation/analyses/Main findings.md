# Main findings so far

## Structure

* Articles in the different datasets are of different sizes:
    * English dataset tends to have articles around 1000 tokens, with the majority being under 3000-3500.
    * Italian articles have around 500 tokens, with the majority under 1500.
    * Russian articles have around 250 tokens, with the majority under 1500-2000.
 * Articles seem to come from fringe rather unreliable sources of news. This is probably because they reflect a specific frame and bias more strongly.
 * Structure/parsing is overall stable, with russian seeming to have the least number of edge cases. Articles in english sometimes have the wrong paragraph order or parsed ads or links to other articles within the document.
 * A couple of exact duplicates where found in the english dataset.
    * No other exact duplicates were detected at the other 4 units of analysis (title_and_5_sentences, title_and_10_sentences, title_and_first_paragraph, title_and_first_sentence_each_paragraph).
    * Fuzzy matching for possible duplicates is pending, but is probably more of an edge case for English given the size of the dataset.
    * 


## Labels

* Overall, we do not see strong correlations between labels in each of the three datasets. That is a good sign that labels are not ambiguous and are in fact distinct.
    * **Correlations between frames are capped between -0.3 and 0.3**
 * All labels are present in all three datasets
 * Lowest relative frequency of a label is around (~5%) and the highest is around (50%)
 * The frequencies are a bit varied across languages, there is not a clear prevalent label across the three languages.
 * Number of labels per document also has a different behaviour per dataset. Each one has a different skew and spread. Russian is skewed towards 1-3 frames, english 3-4, and italian 2-5.  

 * **There is no chance to treat this as a powerset multiclass classification problem unless we break up the powerset into small groups of labels**
   * Out of $2^{14}$ (16,384) combinations, all the datasets combined have 680. That is, we do not have 15,704 of them.
   * 73.6% of the combinations we do have show only once.
   * As labels have a low correlation between them, we probably cannot break them up to cleanly.
 * After visualizing co-occurences of the labels in a graph, in effect **there does not seem to be a subgroup or community of labels**. 

## Base Performance 

 * Using the highest ammount of text we can seems to work the best for the classical classifiers accross all datasets. 

 * By looking at the confusion matrix of each label for each of the datasets, we can also see that the models are not overfitting so strongly that they **only** predict the majority classes. However, while they are not just predicting the majority, there is much room for improvement in the prediction of the minority classes. **Probably via upsampling of minority examples**.

 * Many of the gradient based models are underfitting (XGBoost, Logistic regressions with regularization) as the number of examples is very small, resulting in that the number of iterations must be much higher to reach a better local optimum. 

 * On the other hand, models like RandomForest and SVC are overfitting, as evidenced by their large train/test gaps and their stronger tendency not to predict minority examples.

 * The baseline and best 'f1-micro' performances so far are:
   * English: Baseline (0.39 $\pm$ 0.1), and LinearSVC (0.65 $\pm$ 0.02)
   * Italian: Baseline (0.35 $\pm$ 0.2), and ComplementNaiveBayes (0.53 $\pm$ 0.03)
   * French: Baseline (0.30 $\pm$ 0.2), and LinearSVC (0.48 $\pm$ 0.04) 
   * Polish: Baseline (0.40 $\pm$ 0.2), and LogisticRegression (0.61 $\pm$ 0.02)
   * Russian: Baseline (0.27 $\pm$ 0.2), and LogisticRegression (0.45 $\pm$ 0.02)
   * German: Baseline (0.38 $\pm$ 0.2), and LogisticRegression (0.54 $\pm$ 0.03) 
