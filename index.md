# Explicit or Not? Predicting Song Explicitness from Audio Features

**By Canyon Crutchfield and Nate McNeill**

---

## Introduction

In this project we explore a dataset provided by Spotify of songs from dozens of genres, with each song having audio features such as energy, danceability, valence, loudness, and speechiness. I honed our focus on 5 distinct and important musical genres — **urban** (hip-hop, R&B), **metal**, **electronic** (EDM, house, dubstep, techno), **acoustic** (folk, singer-songwriter), and **latin** (reggaeton, salsa, reggae) — giving us 18,000 tracks total after cleaning.

Our central question is: **Can we predict whether a song is explicit based on its audio features?**

This matters because the relationship between a song's audio features and its lyrical content is obscure. If you can accurately predict explicit songs given the base audio features you can automate content modertaion and musical recommendations without needing to sift through the actual songs lyrcis

The dataset contains **18,000 rows** after cleaning. With the following relevany columns:

| Column | Description |
|---|---|
| `explicit` | Boolean — whether the track is marked explicit on Spotify |
| `energy` | Measures intensity, from 0.0 to 1.0 |
| `loudness` | Overall loudness in decibels (typically –60 to 0 dB) |
| `speechiness` | Presence of spoken words, higher values indicate more speech-like content |
| `danceability` | How suitable a track is for dancing based on rhythm and beat |
| `valence` | Musical positivity, high = happy, low = sad/anrgy|
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

This box plot compares the energy distributions of explicit and non-explicit tracks. Explicit tracks have a noticeably higher median energy (0.825) compared to non-explicit tracks (0.766), along with the entire box plot being shifted upward. We found this very intriguing so we further explore this reltaionship later on in the project

### Bivariate Analysis

<iframe
  src="assets/fig_valence_hist.html"
  width="100%"
  height="500"
  frameborder="0"
></iframe>

This overlaid histogram shows the conditional distribution of valence across the five genre categories. The distributions are strikingly different in shape — metal is heavily right-skewed (most tracks score low on positivity), while latin skews strongly left (most tracks are upbeat). This confirms that genre is a meaningful feature for our classifier, as it captures distinct emotional and sonic profiles.

### Interesting Aggregates

The table below shows mean audio features broken down by genre. It reveals the "sonic fingerprint" of each category:

| genre_category | danceability | energy | valence | loudness | acousticness |
|---|---|---|---|---|---|
| acoustic | 0.497 | 0.468 | 0.423 | -11.842 | 0.681 |
| electronic | 0.618 | 0.762 | 0.403 | -7.231 | 0.064 |
| latin | 0.716 | 0.663 | 0.621 | -7.018 | 0.193 |
| metal | 0.382 | 0.891 | 0.298 | -5.814 | 0.027 |
| urban | 0.734 | 0.651 | 0.548 | -7.442 | 0.148 |

Metal stands out with the highest energy and lowest valence by a wide margin. Latin and urban lead in danceability. Acoustic tracks are the quietest and most acoustic, as expected. These patterns reinforce why genre is a valuable feature for predicting explicitness — each genre carries a distinct baseline profile.

---

## Assessment of Missingness

### NMAR Analysis

We believe the `tempo` column may be **NMAR** (Not Missing At Random). Tempo is detected algorithmically by Spotify, and certain tracks — particularly those with highly irregular rhythms, rubato, or freeform structures common in experimental metal and acoustic music — may resist clean BPM detection. In those cases, the missingness arises from the track's own tempo characteristics, meaning the reason a value is missing is related to the value itself. To potentially make this MAR, we would want additional data from Spotify on whether tempo detection was attempted and failed, or simply not run.

### Missingness Dependency

We analyzed whether the missingness of `tempo` (2,918 missing out of 18,000) depends on other columns.

**Tempo missingness IS dependent on `energy` (MAR):**
A permutation test using difference in means yielded a p-value of 0.0 — in none of 500 trials did a random shuffle produce a difference as extreme as observed. Tracks with missing tempo tend to have higher energy on average, suggesting that high-energy tracks with complex rhythmic structures are more likely to have undetected tempo values.

**Tempo missingness IS dependent on `explicit` (MAR):**

<iframe
  src="assets/fig_missingness_explicit.html"
  width="100%"
  height="500"
  frameborder="0"
></iframe>

Using TVD as the test statistic (since `explicit` is categorical), the permutation p-value was 0.004. The distribution of explicitness differs meaningfully between tempo-missing and tempo-present tracks, confirming MAR dependency.

**Tempo missingness is NOT dependent on `duration_ms` (MCAR with respect to duration):**
The permutation p-value was 0.468 — the observed difference in mean duration between tempo-missing and tempo-present tracks is well within what chance alone could produce. Song length does not appear to influence whether tempo is recorded.

---

## Hypothesis Testing

**Question: Are explicit tracks more energetic on average than non-explicit tracks?**

- **Null Hypothesis:** Explicit and non-explicit tracks have the same mean energy level. Any observed difference is due to chance.
- **Alternative Hypothesis:** Explicit tracks have higher mean energy on average than non-explicit tracks.
- **Test Statistic:** Difference in mean energy (explicit − non-explicit)
- **Significance Level:** α = 0.05

We used a one-tailed permutation test with 500 trials. The observed difference in means was **+0.069** (explicit mean: 0.788, non-explicit mean: 0.719).

<iframe
  src="assets/fig_hypothesis.html"
  width="100%"
  height="500"
  frameborder="0"
></iframe>

The p-value was less than 0.002 (0 out of 500 permuted trials produced a test statistic as extreme as observed). We **reject the null hypothesis**. The difference in means as our test statistic is the natural choice here since energy is a continuous variable and we have a directional hypothesis. A permutation test is appropriate because we make no assumptions about the underlying distribution of energy.

However, this result does not hold uniformly across all genres. In latin and urban tracks, non-explicit songs are actually slightly more energetic on average — suggesting the overall signal is driven primarily by metal and acoustic genres where the gap is most pronounced. We cannot conclude that explicit tracks are universally more energetic; the relationship is genre-dependent.

---

## Framing a Prediction Problem

**Prediction Problem:** Given the audio features of a Spotify track, predict whether it is explicit (`True`) or not (`False`).

This is a **binary classification** problem. The response variable is `explicit` — a boolean label already present in the dataset and fully determined at the time a song is released on Spotify. All audio features (energy, danceability, valence, loudness, acousticness, instrumentalness, speechiness, liveness) are computed from the audio signal itself, so they are always available at prediction time. We are not using any features derived from listener behavior or metadata that would be unknown at the time of prediction.

**Evaluation Metric: F1-Score**

We chose F1-score over accuracy because the dataset is heavily imbalanced — approximately 87% of tracks are non-explicit. A naive classifier that always predicts `False` would achieve 87% accuracy while being completely useless. F1-score balances precision and recall, penalizing models that ignore the minority class (explicit tracks), making it a far more meaningful measure of performance for this task.

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

The baseline model performs poorly — an F1 of ~0.24 means it is struggling to correctly identify explicit tracks. This is expected: energy and loudness alone are weak signals. The similar train and test scores indicate the model is not overfitting; it is simply underfitting due to insufficient features and a linear decision boundary that cannot capture the complex relationship between audio features and explicitness. This gives us a clear target to beat with the final model.

---

## Final Model

### Feature Engineering

We engineered two new features on top of the baseline:

**1. `energy_x_loudness` (interaction term):** Songs that are simultaneously high in both energy and loudness carry a much stronger signal toward explicitness than either feature alone. A song with high energy but moderate loudness (e.g., fast acoustic fingerpicking) is unlikely to be explicit, while a song that is both loud and high-energy (e.g., a hip-hop banger) is far more likely to be. Multiplying the two features creates a single variable that captures this joint effect directly.

**2. `speechiness` with `QuantileTransformer`:** The raw speechiness distribution is extremely right-skewed — the vast majority of songs cluster near 0 (instrumental or song-like), while a small tail of rap and spoken-word tracks have high speechiness. High speechiness directly increases the opportunity for explicit language, making it one of the most theoretically motivated predictors. The `QuantileTransformer` spreads out the crowded low-speechiness values into a uniform distribution, allowing the model to better distinguish between the many songs with subtly different low speechiness scores rather than collapsing them all together.

We also added all remaining audio features (`danceability`, `valence`, `acousticness`, `instrumentalness`, `liveness`) and one-hot encoded `genre_category`, since genre carries strong implicit information about lyrical norms.

### Model & Hyperparameter Tuning

We switched from Logistic Regression to a **Random Forest Classifier** to capture the non-linear, feature-interaction-heavy relationships between audio features and explicitness that a linear model cannot learn.

We tuned two hyperparameters using 5-fold `GridSearchCV` with F1-score:

- `n_estimators` ∈ {100, 200, 250} — more trees reduce variance with diminishing returns
- `max_depth` ∈ {5, 10, 15, 20} — controls overfitting vs. underfitting tradeoff

**Best parameters found: `n_estimators=200`, `max_depth=20`**

| Split | F1-Score |
|---|---|
| Baseline Test | 0.2398 |
| Final Model Train | ~0.98 |
| Final Model Test | ~0.64 |
| Improvement | **+0.40** |

The final model improved test F1 by roughly 40 percentage points over the baseline. The high train F1 relative to test F1 suggests some overfitting at max_depth=20, but the test performance is still a dramatic improvement. The engineered features — particularly speechiness and the energy×loudness interaction — gave the model the signal it needed to meaningfully separate explicit from non-explicit tracks.

---

## Fairness Analysis

**Groups:** Metal tracks (Group X) vs. Non-metal tracks (Group Y)  
**Metric:** Precision — of all tracks the model predicts as explicit, what fraction actually are?

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

The permutation p-value was **0.0** (less than 0.001). Since p < 0.01, we **reject the null hypothesis**. Our model's precision differs significantly between metal and non-metal tracks — it is less effective at correctly predicting explicitness for metal songs compared to non-metal songs. This fairness disparity likely stems from the fact that metal has a very low base rate of explicitness, making it harder for the model to achieve high precision on that group. Future work could explore genre-stratified training or threshold adjustment to address this imbalance.
