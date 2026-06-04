**By Canyon Crutchfield and Nate McNeill**

---

## Introduction

In this project we explore a dataset provided by Spotify of songs from dozens of genres, with each song having audio features such as energy, danceability, valence, loudness, and speechiness. I honed our focus on 5 distinct and important musical genres — **urban** (hip-hop, R&B), **metal**, **electronic** (EDM, house, dubstep, techno), **acoustic** (folk, singer-songwriter), and **latin** (reggaeton, salsa, reggae) — giving us 18,000 tracks total after cleaning.

Our central question is: **Can we predict whether a song is explicit based on its audio features?**

This matters because the relationship between a song's audio features and its lyrical content is obscure. If you can accurately predict explicit songs given the base audio features you can automate content moderation and musical recommendations without needing to sift through the actual songs lyrics

The dataset contains **18,000 rows** after cleaning. With the following relevant columns:

| Column | Description |
|---|---|
| `explicit` | Boolean — whether the track is marked explicit on Spotify |
| `energy` | Measures intensity, from 0.0 to 1.0 |
| `loudness` | Overall loudness in decibels (typically –60 to 0 dB) |
| `speechiness` | Presence of spoken words, higher values indicate more speech-like content |
| `danceability` | How suitable a track is for dancing based on rhythm and beat |
| `valence` | Musical positivity, high = happy, low = sad/angry |
| `acousticness` | Confidence that the track is acoustic, from 0.0 to 1.0 |
| `instrumentalness` | Predicts whether a track has no vocals |
| `liveness` | Detects the presence of a live audience |
| `genre_category` | Our derived broad genre label (urban, metal, electronic, acoustic, latin) |

---

## Data Cleaning and Exploratory Data Analysis

### Data Cleaning

1. **Dropped a corrupted row** this row had many corrupted values, in `artists`, `album_name`, and `track_name`. It was the only row with missing values across these columns.
2. **Dropped the `Unnamed: 0` column**, useless column that did not add any new information, resulted from loading it from the CSV.
3. **Extracted the year from `release_date`** every song had year as the first 4 values in the date column, but not every entry had months or days. So we sliced the first 4 from each entry.
4. **Consolidated the dozens of raw genres into 5 broad categories** using a dictionary to map it. We chose urban, metal, electronic, acoustic, and latin. These genres were chosen because of the distinct audio features of each genre. Urban has high speechiness and danceability, metal has extremely high energy and low valence, acoustic has low energy and high acousticness, latin has high valence and danceability, and electronic has high energy with near-zero acousticness. These genre cover a large variety of audio features, and they were some of our personal favorite genres to listen to.
   
This is the markdown of the first rows of the cleaned dataframe with the relevant columns:

| track_name                 | artists                | genre_category   | explicit   |   energy |   loudness |   speechiness |   danceability |   valence |
|:---------------------------|:-----------------------|:-----------------|:-----------|---------:|-----------:|--------------:|---------------:|----------:|
| Comedy                     | Gen Hoshino            | acoustic         | False      |   0.461  |     -6.746 |        0.143  |          0.676 |     0.715 |
| Ghost - Acoustic           | Ben Woodward           | acoustic         | False      |   0.166  |    -17.235 |        0.0763 |          0.42  |     0.267 |
| To Begin Again             | Ingrid Michaelson;ZAYN | acoustic         | False      |   0.359  |     -9.734 |        0.0557 |          0.438 |     0.12  |
| Can't Help Falling In Love | Kina Grannis           | acoustic         | False      |   0.0596 |    -18.515 |        0.0363 |          0.266 |     0.143 |
| Hold On                    | Chord Overstreet       | acoustic         | False      |   0.443  |     -9.681 |        0.0526 |          0.618 |     0.167 |

### Univariate Analysis

<iframe
  src="assets/fig_energy_box.html"
  width="100%"
  height="500"
  frameborder="0"
></iframe>

This box plot compares the energy distributions of explicit and non-explicit tracks. Explicit tracks have a noticeably higher median energy (0.825) compared to non-explicit tracks (0.766), along with the entire box plot being shifted upward. We found this very intriguing so we further explore this relationship later on in the project

### Bivariate Analysis

<iframe
  src="assets/fig_valence_hist.html"
  width="100%"
  height="500"
  frameborder="0"
></iframe>

This overlaid histogram shows the conditional distribution of valence across the five genre categories. The distributions are strikingly different in shape — metal is heavily right-skewed(sadder songs), while latin skews strongly left (more upbeat songs). This confirms that genre is a meaningful feature for our classifier, as it captures distinct emotional and sonic profiles.

### Interesting Aggregates

The table below shows mean audio features broken down by genre. It reveals the "sonic fingerprint" of each category:

| genre_category   |   danceability |   energy |   valence |   loudness |   acousticness |
|:-----------------|---------------:|---------:|----------:|-----------:|---------------:|
| acoustic         |          0.557 |    0.472 |     0.443 |     -9.42  |          0.532 |
| electronic       |          0.63  |    0.754 |     0.388 |     -5.921 |          0.104 |
| latin            |          0.724 |    0.729 |     0.684 |     -5.616 |          0.248 |
| metal            |          0.396 |    0.887 |     0.312 |     -5.504 |          0.019 |
| urban            |          0.675 |    0.66  |     0.592 |     -6.761 |          0.282 |

Metal stands out with the highest energy and lowest valence by a lot. Latin and urban lead in danceability. Acoustic tracks are the quietest and most acoustic. These observations show the unique sound portfolio for each genre that we selected, and demonstrate how genre is an important predictor of audio features.

---

## Assessment of Missingness

### NMAR Analysis

For the missingness our cleaned dataframe only has 1 column with missing values and that is `tempo`. In our data frame, tempo is missing 2918 values out of the 18000 entries. So now we need to determine that cause of the missing values. We believe that `tempo` could potentially be NMAR, we know that Spotify uses algorithms to detect BPM so irregular beats could slip by the `tempo` measurement taken by Spotify. To truly know we would need to know if the detection failed on the song or simply wasn't run. Nonetheless, we will run tests to see if it's MAR or MCAR on any columns.

### Missingness Dependency

Analysis of missingness of `tempo` on other columns

**Tempo missingness IS dependent on `energy` (MAR):**
A permutation test using difference in means yielded a p-value of 0.0. Out of the 500 random permutation trials that we ran, none produced a difference as extreme as observed. Tracks with missing tempo tend to have higher energy on average, hinting that high energy with more unique and complex beats have higher rates of missingness.

**Tempo missingness IS dependent on `explicit` (MAR):**

<iframe
  src="assets/fig_missingness_explicit.html"
  width="100%"
  height="500"
  frameborder="0"
></iframe>

Using TVD as the test statistic due to `explicit` being a categorical column, the permutation test p-value was 0.004. This is significant enough to conclude that missing values in `tempo` are dependent on `explicit`.

**Tempo missingness is NOT dependent on `duration_ms` :**
The permutation p-value was 0.468 — the observed difference in mean duration between tempo-missing and tempo-present tracks is well within what chance alone could produce. Song length does not appear to influence whether tempo is recorded.

---

## Hypothesis Testing

**Question: Are explicit tracks more energetic on average than non-explicit tracks?**

- **Null Hypothesis:** Explicit and non-explicit tracks have the same mean energy level. Any observed difference is due to chance.
- **Alternative Hypothesis:** Explicit tracks have higher mean energy on average than non-explicit tracks.
- **Test Statistic:** Difference in mean energy (explicit − non-explicit)
- **Significance Level:** α = 0.01

We used a one-tailed permutation test with 500 trials. The observed difference in means was **+0.069** (explicit mean: 0.788, non-explicit mean: 0.719).

<iframe
  src="assets/fig_hypothesis.html"
  width="100%"
  height="500"
  frameborder="0"
></iframe>

The p-value was 0.0. This is statistically significant, so we **reject the null hypothesis**. The difference in means as our test statistic is the obvious choice here since energy is a continuous variable and we have a hypothesis with direction. We use a permutation test to simulate under the null.

However, this result does not hold uniformly across all genres. Latin and urban tracks show the opposite trend. Non-explicit tracks are slightly more energetic in both cases. Suggesting that the overall relationship between explicitness and energy is largely driven by genres like metal and acoustic where the difference is most pronounced.

---

## Framing a Prediction Problem

**Prediction Problem:** Given the audio features of a Spotify track, predict whether a song is explicit.

This is a **Binary Classification** prediction model. The response variable is `explicit` this is what we are trying to predict given the audio features of a song, the features that are going to be used are `energy`, `danceability`, `valence`, `loudness`, `acousticness`, `instrumentalness`, `tempo`, `speechiness`, and `liveness`. All of the features will be available to us before we predict the explicit label. 

For our evaluation metric of the model we chose to use the **F1-Score of the model**. This is because the data set is significantly weighted to non explicit songs so a naive model that just guesses `False` would result in 87% accuracy. This is why F1-score is more appropriate — it emphasizes the importance of getting the `True` values correct.

---

## Baseline Model

Our baseline model is a **Logistic Regression** trained on two quantitative features:

- `energy` — quantitative continuous
- `loudness` — quantitative continuous

Both features were standardized using `StandardScaler` within a single `sklearn` Pipeline. No categorical features were included. We used `class_weight='balanced'` to account for the label imbalance.

| Split | F1-Score |
|---|---|
| Train | 0.2506 |
| Test | 0.2398 |

The train and test F1 scores of 0.2506 and 0.2398 are close to each other, suggesting the model generalizes without overfitting. However, this is not a good baseline model — an F1 of ~0.24 means the model fails to identify the vast majority of explicit tracks correctly. This is expected given that we are only using two features, and energy and loudness alone do not capture enough of the signal needed to distinguish explicit from non-explicit songs across diverse genres. The baseline serves as a useful lower bound to measure improvement against in the final model.

We could potentially add more features to the model or we could also switch the model to a random forest to capture non linear relationships between variables. We think that random forest would be a good idea along with feature engineering to generate a more optimized classifying model.

---

## Final Model

### Added Features

We included all remaining audio features (danceability, valence, acousticness, instrumentalness, liveness) and one-hot encoded genre_category.

These features combine to describe the sound profile of the track. `acousticness` and `instrumentalness` are meaningful — high acoustic and instrumental songs don't have as much speech in them, and the genres that they fall into have lower rates of explicit language. 

`valence` captures emotional tone. Explicit tracks tend toward aggression or raw emotion rather than positivity. 

`danceability` and `liveness` add context about the performance setting and rhythm structure, which correlate with genre norms around language.

`genre_category` (OHE): The genre of a song is highly cultural, with different cultures having differing norms around explicit language. An urban track uses language that would never appear in an acoustic folk song. That difference in culture leads to different rates of explicit language in the songs, making it one of the most predictive features we could include. We need to use one-hot encoding because `genre_category` has no inherent order.

### Feature Engineering

Feature engineering 2 columns

`energy_x_loudness` — songs that are both high in energy and loudness prompt a much stronger signal of explicitness. Easier to capture these together than to compare the relation of either one on their own

`speechiness` applying QuantileTransformer this distribution is very skewed because most songs have low speechiness other than spoken word/rap songs. High speechiness values allows a lot more opportunity for explicit words to be used because there are higher levels of words used. For example, EDM songs have a very low speechiness value because there are not many lyrics or vocals involved, so therefore they have a lesser chance of using explicit language. The QuantileTransformer is used so that the model can better distinguish the crowded low-value speechiness values.

Hyperparameters to Tune
We plan to tune two hyperparameters for a RandomForestClassifier, we chose this model to capture the non-linear relationships between features and to deal with the imbalance between explicit and non-explicit songs

n_estimators: Number of trees within the forest. The more trees a forest has the more it reduces variance by averaging over more independent trials, with diminishing returns. We iterate over [100, 200, 250] to find the optimal level.

max_depth: controls the depth of the tree, how deep the nodes stretch. Shallow trees underfit the data while deep trees overfit, so lower and higher variances. We loop through [5, 10, 15, 20] to find the best depth for the forest.

**Best parameters found: `n_estimators=200`, `max_depth=20`**

| Split | F1-Score |
|---|---|
| Baseline Test | 0.2398 |
| Final Model Train | ~0.98 |
| Final Model Test | ~0.64 |
| Improvement | **+0.40** |

The final model improved test F1 by roughly 40 percentage points over the baseline. The high train F1 relative to test F1 shows some overfitting at max_depth=20, but the test performance is still a large improvement.

---

## Fairness Analysis

**Groups:** Metal tracks (Group X) vs. Non-metal tracks (Group Y)  
**Metric:** Precision, of all tracks the model predicts as explicit, what fraction actually are?

We chose this comparison because metal is the genre with the most training examples and the lowest real-world explicit rate, so we were curious whether the model had learned genre-specific biases.

- **Null Hypothesis:** Our model is fair. Its precision for metal and non-metal tracks is roughly the same; any observed difference is due to random chance.
- **Alternative Hypothesis:** Our model is unfair. Its precision differs between metal and non-metal tracks.
- **Test Statistic:** Precision(metal) − Precision(non-metal)
- **Significance Level:** α = 0.01
- **Test Type:** Two-tailed permutation test with 1,000 trials

<iframe
  src="assets/fig_fairness.html"
  width="100%"
  height="500"
  frameborder="0"
></iframe>

The permutation p-value was **0.0**, this is statistically significant so we **reject the null hypothesis**. Our model's precision differs significantly between metal and non-metal tracks. It is far less effective at predicting the explicitness of metal vs non-metal songs. This fairness disparity likely stems from the fact that metal has a very low base rate of explicitness, making it harder for the model to achieve high precision on that group.
