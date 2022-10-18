# Main findings so far

## Structure

* Articles in the different datasets are of different sizes:
    * English dataset tends to have articles around 1000 tokens, with the majority being under 3000-3500.
    * Italian articles have around 500 tokens, with the majority under 1500.
    * Russian articles have around 250 tokens, with the majority under 1500-2000.
 * Articles seem to come from fringe rather unreliable sources of news. This is probably because they reflect a specific frame and bias more strongly.
 * Structure/parsing is overall stable, with russian seeming to have the least number of edge cases. Articles in english sometimes have the wrong paragraph order or parsed ads or links to other articles within the document.
 * A couple of exact duplicates where found in the english dataset.


## Labels

* Overall, we do not see strong correlations between labels in each of the three datasets. That is a good sign that labels are not ambiguous and are in fact distinct.
    * **Correlations between frames are capped between -0.3 and 0.3**
 * All labels are present in all three datasets
 * Lowest relative frequency of a label is around (~5%) and the highest is around (50%)
 * The frequencies are a bit varied across languages, there is not a clear prevalent label across the three languages.
 * Number of labels per document also has a different behaviour per dataset. Each one has a different skew and spread. Russian is skewed towards 1-3 frames, english 3-4, and italian 2-5.  


## Base Performance
