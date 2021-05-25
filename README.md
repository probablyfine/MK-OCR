# MK-OCR

<img width="30%" src='https://github.com/probablyfine/MK-OCR/raw/main/readme-img/meme.png'>

# Table of Contents

* [What/Why](https://github.com/probablyfine/MK-OCR#whatwhy)
* [How](https://github.com/probablyfine/MK-OCR#how)
* [Requirements](https://github.com/probablyfine/MK-OCR#requirements)
* [Usage](https://github.com/probablyfine/MK-OCR#usage)
    - [High-level functions](https://github.com/probablyfine/MK-OCR#high-level-functions)
    - [Using the individual classifiers](https://github.com/probablyfine/MK-OCR#using-the-individual-classifiers)
    - [Work with image data directly](https://github.com/probablyfine/MK-OCR#work-with-image-data-directly)
    - [Label your own training data](https://github.com/probablyfine/MK-OCR#label-your-own-training-data)


# What/Why

Well, it's code that uses machine learning to analyze Mario Kart race result screenshots and dump all the results into a spreadsheet. Basically it's a very silly, highly specific [OCR](https://en.wikipedia.org/wiki/Optical_character_recognition) routine.

<img src='https://github.com/probablyfine/MK-OCR/raw/main/readme-img/intro.jpg'>

I wrote it after playing a bunch of Mario Kart online and wondering how the game awards points after a race. It's obvious that you get more points for beating players with a better rating than you, and lose more points for losing to someone with a worse rating than you, but the exact rules aren't clear. I thought I might be able to work out the pattern if I could analyze the data. I haven't figured it out really (I'll probably post more about that another time), but maybe someone else wants to take a crack at it. If nothing else, this might be useful as an example of a simple machine learning task.

# How

The stats shown after a race are an easy target for simple machine learning since the digits are very distinct and consistent in appearance.

The first step is to extract the relevant pixel data from the race results screen. We need: 
- Each digit for each player's current rating (called "VR", for Versus Rating)
- Each digit for the number of points awarded to each player
- The awarded points' sign (+ or -) designating a gain or a loss.

This is a [supervised classification](https://en.wikipedia.org/wiki/Supervised_learning) problem, so the next step is to create a training data set. I did this already by manually labeling a bunch of the pixel data, for use in a classifier. This classification problem is not particularly hard and I suspect the choice of classifier doesn't make a whole lot of difference, but I chose a [multi-class support vector machine (SVM)](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html). SVMs are really binary classifiers (distinguish A from B only), so scikit-learn implements a one-vs-rest scheme for the multi-class case. That just means that, for the classifier to succeed, a given digit's pixel data need to be separable from the pixel data of all the other digits.

Rather than feed the raw pixels into the classifier, they're processed first by:
- Converting to grayscale
- Applying a sharpening filter to accentuate edges
- [Reducing the dimensionality](https://en.wikipedia.org/wiki/Dimensionality_reduction) using [factor analysis](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FactorAnalysis.html)

The model fitting process uses [cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)) to choose the number of components to retain from the factor analysis, and to tune the [regularization strength](https://stats.stackexchange.com/a/18765) of the SVM. It selects the best-performing model and exposes it to the user.

# Requirements
- Mario Kart 8 Deluxe screenshots. I captured mine using a Switch Lite. I suspect a full-sized Switch would be fine, too, but I didn't try it. You can also just use my screenshots.
- Python3 (I'm using 3.7.1)
- PIL
- NumPy
- scikit-learn
- pandas (and xlrd or openpyxl)
- joblib
- [pigeon](https://github.com/agermanidis/pigeon), only if you want to label your own training data for some reason

# Usage

## High-level functions

The high-level MKDataCompiler class is the easiest place to start, and it's very simple to use. Just give it training data and your screenshots, and get a pandas DataFrame.


```python
from MKDataCompiler import MKDataCompiler
import glob
```

#### load training data, tune/fit image classifiers (one for each of the three types of image data)


```python
paths_to_labeled = {'vr_digits':  'vr_digits_labeled.xlsx',
                    'pts_digits': 'pts_digits_labeled.xlsx',
                    'pts_signs':  'pts_signs_labeled.xlsx'}
mkdc = MKDataCompiler( paths_to_labeled, n_jobs=4)
df = mkdc.compile( glob.glob( 'images-redacted/*.jpg'))
df.head(20).fillna('')
```

    tuning vr_digits...done.
    classification accuracy w/ 10-fold xval: 100.0% using ncomp=20 and C=0.1
    tuning pts_digits...done.
    classification accuracy w/ 10-fold xval: 100.0% using ncomp=15 and C=0.01
    tuning pts_signs...done.
    classification accuracy w/ 10-fold xval: 100.0% using ncomp=4 and C=0.1
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>VR</th>
      <th>points</th>
      <th>is user</th>
    </tr>
    <tr>
      <th>race</th>
      <th>rank</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="8" valign="top">1</th>
      <th>1</th>
      <td>13453</td>
      <td>20</td>
      <td></td>
    </tr>
    <tr>
      <th>2</th>
      <td>10178</td>
      <td>16</td>
      <td></td>
    </tr>
    <tr>
      <th>3</th>
      <td>2417</td>
      <td>21</td>
      <td></td>
    </tr>
    <tr>
      <th>4</th>
      <td>10937</td>
      <td>4</td>
      <td></td>
    </tr>
    <tr>
      <th>5</th>
      <td>10342</td>
      <td>-2</td>
      <td></td>
    </tr>
    <tr>
      <th>6</th>
      <td>10243</td>
      <td>-7</td>
      <td>x</td>
    </tr>
    <tr>
      <th>7</th>
      <td>10455</td>
      <td>-16</td>
      <td></td>
    </tr>
    <tr>
      <th>8</th>
      <td>10010</td>
      <td>-19</td>
      <td></td>
    </tr>
    <tr>
      <th rowspan="12" valign="top">2</th>
      <th>1</th>
      <td>10311</td>
      <td>27</td>
      <td></td>
    </tr>
    <tr>
      <th>2</th>
      <td>10119</td>
      <td>21</td>
      <td></td>
    </tr>
    <tr>
      <th>3</th>
      <td>12154</td>
      <td>17</td>
      <td></td>
    </tr>
    <tr>
      <th>4</th>
      <td>17105</td>
      <td>3</td>
      <td></td>
    </tr>
    <tr>
      <th>5</th>
      <td>9956</td>
      <td>9</td>
      <td></td>
    </tr>
    <tr>
      <th>6</th>
      <td>11916</td>
      <td>2</td>
      <td></td>
    </tr>
    <tr>
      <th>7</th>
      <td>11778</td>
      <td>-2</td>
      <td></td>
    </tr>
    <tr>
      <th>8</th>
      <td>9104</td>
      <td>-1</td>
      <td></td>
    </tr>
    <tr>
      <th>9</th>
      <td>10265</td>
      <td>-9</td>
      <td>x</td>
    </tr>
    <tr>
      <th>10</th>
      <td>12190</td>
      <td>-18</td>
      <td></td>
    </tr>
    <tr>
      <th>11</th>
      <td>11686</td>
      <td>-24</td>
      <td></td>
    </tr>
    <tr>
      <th>12</th>
      <td>1007</td>
      <td>-2</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>



That's it! It works well; the cross-validation result suggests the classifier has perfect accuracy. It's not super surprising, but pretty cool.

The biggest trouble with this classification task has to do with detecting blank spaces that don't contain digits. Blanks can occur for a few reasons: if there are less than 12 players; if a "VR" rating is less than 5 digits; or if the points awarded are single-digit. In any of those scenarios, some random part of the game's visuals ends up in the extracted data.

This issue made it tough at first to get perfect accuracy. There were always a couple samples that got misclassified. My first solution was to just add more training data, which is why there is such a stupidly large number of labeled samples in my training set. It didn't really help though, so I started playing around with pre-processing and found that applying a sharpening filter to the image made a big difference. This filter made each digit stand out better against the background scenes.

## Using the individual classifiers

You don't need to use the high-level MKDataCompiler class. Using the classifiers directly might also be interesting:


```python
from MKImageClassifier import MKImageClassifier
import matplotlib.pyplot as plt
import numpy as np
```

#### Train the SVM for the points-awarded digits


```python
clf_pts_digits = MKImageClassifier( 'pts_digits_labeled.xlsx', 'pts_digits', n_splits=10)
_ = clf_pts_digits.tune( n_jobs=4)
```

    tuning pts_digits...done.
    classification accuracy w/ 10-fold xval: 100.0% using ncomp=15 and C=0.01
    

#### Plot the low-dimensional training data features (i.e. scores from factor analysis)


```python
fig,axs = plt.subplots( ncols=3, figsize=(13,4))
for d in np.unique( clf_pts_digits.mdl.y_train):
    mask = (clf_pts_digits.mdl.y_train == d)
    _ = axs[0].scatter( clf_pts_digits.mdl.scores[mask,1], clf_pts_digits.mdl.scores[mask,3], s=2)
    _ = axs[0].set_xlabel( 'Factor 2', size=16)
    _ = axs[0].set_ylabel( 'Factor 4', size=16)
    
    _ = axs[1].scatter( clf_pts_digits.mdl.scores[mask,3], clf_pts_digits.mdl.scores[mask,6], s=2)
    _ = axs[1].set_xlabel( 'Factor 4', size=16)
    _ = axs[1].set_ylabel( 'Factor 7', size=16)
    
    _ = axs[2].scatter( clf_pts_digits.mdl.scores[mask,0], clf_pts_digits.mdl.scores[mask,6], s=2)
    _ = axs[2].set_xlabel( 'Factor 1', size=16)
    _ = axs[2].set_ylabel( 'Factor 7', size=16)
plt.tight_layout()
```


<img src='https://github.com/probablyfine/MK-OCR/raw/main/readme-img/low-dim.png'>


Each dot is the low-dimensional representation of one sample of digit pixels. Different samples of the same digit are shown in the same color, and they tend to cluster together (which is why the classifier can tell them apart). Cluster separation isn't great along every factor/dimension so I selected a couple of the more interesting planes for plotting. The clusters appear elongated in the low-D space-- there's probably a way to adjust the normalization to fix this but it works well anyway, so it's Probably Fine as-is.

#### Fit the other two classifiers


```python
clf_pts_signs = MKImageClassifier( 'pts_signs_labeled.xlsx', 'pts_signs', n_splits=10)
_ = clf_pts_signs.tune( n_jobs=4)

clf_vr_digits = MKImageClassifier( 'vr_digits_labeled.xlsx', 'vr_digits', n_splits=10)
_ = clf_vr_digits.tune( n_jobs=4)
```

    tuning pts_signs...done.
    classification accuracy w/ 10-fold xval: 100.0% using ncomp=4 and C=0.1
    tuning vr_digits...done.
    classification accuracy w/ 10-fold xval: 100.0% using ncomp=20 and C=0.1
    

#### Load a bunch of images to classify (using MKImageLoader, more on that below)...


```python
from MKImageLoader import load_images
vr_digits, pts_digits, pts_signs, user_ranks = load_images( glob.glob( 'images-redacted\\*jpg'))
```

#### ...and run the loaded data through each classifier


```python
vr_digits_hat  = clf_vr_digits.predict( vr_digits)
pts_digits_hat = clf_pts_digits.predict( pts_digits)
pts_signs_hat  = clf_pts_signs.predict( pts_signs)
```

#### Take a look at the predictions


```python
img = 4
fig,axs = plt.subplots( nrows=12, ncols=8, figsize=(9,13))
for rank in range(12):
    for digit in range(5):
        axs[rank,digit+3].imshow( vr_digits[:,:,digit,rank,img])
        axs[rank,digit+3].set_title( vr_digits_hat[digit,rank,img], fontsize=26)
    
    _ = axs[rank,0].imshow( pts_signs[:,:,0,rank,img])
    _ = axs[rank,0].set_title( pts_signs_hat[0,rank,img], fontsize=26)
    
    _ = axs[rank,1].imshow( pts_digits[:,:,0,rank,img])
    _ = axs[rank,1].set_title( pts_digits_hat[0,rank,img], fontsize=26)
    
    _ = axs[rank,2].imshow( pts_digits[:,:,1,rank,img])
    _ = axs[rank,2].set_title( pts_digits_hat[1,rank,img], fontsize=26)
for ax in axs.flatten():
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
```


<img src='https://github.com/probablyfine/MK-OCR/raw/main/readme-img/px_pred.png'>


## Work with image data directly
You can use the image loader class, MKImageLoader, if you want more direct access to the pixel data.


```python
from MKImageLoader import MKImageLoader
```


```python
mkil = MKImageLoader( 'images-redacted\\2020082317571100-16851BE00BC6068871FE49D98876D6C5.jpg')
mkil.main_region
```




<img src='https://github.com/probablyfine/MK-OCR/raw/main/readme-img/race_stats.png'>



#### View the winner's pixels


```python
mkil.player_regions[0]
```




<img src='https://github.com/probablyfine/MK-OCR/raw/main/readme-img/winner_stats.png'>



#### View your own pixels
The image loader auto-detects which place *you* came in, so you can track yourself easily. This is also important because the image loader has to invert the colors for your stats, otherwise they won't be white-on-a-black-background like all the other players.


```python
mkil.player_regions[mkil.user_rank]
```




<img src='https://github.com/probablyfine/MK-OCR/raw/main/readme-img/my_stats.png'>



#### Extract every VR rating as a numpy array of image data (height-by-width-by-digit-by-rank)


```python
vr_digits = mkil.get_vr_digits()
type( vr_digits), vr_digits.shape
```




    (numpy.ndarray, (23, 15, 5, 12))



#### Inspect individual VR digits from numpy array


```python
fig,axs = plt.subplots( ncols=5, figsize=(4,1))
for i,ax in enumerate( axs):
    ax.imshow( vr_digits[:,:,i,mkil.user_rank])
```


<img src='https://github.com/probablyfine/MK-OCR/raw/main/readme-img/vr_rating.png'>


#### Do the same with the sign and digits of points awarded


```python
pts_signs = mkil.get_pts_signs()
pts_digits = mkil.get_pts_digits()
pts_signs.shape, pts_digits.shape
```




    ((13, 13, 1, 12), (18, 12, 2, 12))




```python
fig,axs = plt.subplots( ncols=3, figsize=(2.5,1))
_ = axs[0].imshow( pts_signs[:,:,0,mkil.user_rank])
_ = axs[1].imshow( pts_digits[:,:,0,mkil.user_rank])
_ = axs[2].imshow( pts_digits[:,:,1,mkil.user_rank])
```


<img src='https://github.com/probablyfine/MK-OCR/raw/main/readme-img/pts_awarded.png'>


#### Load a whole folder full of images
The resulting numpy arrays will be essentially the same dimensions as above, except there is one more dimension added to stack multiple images


```python
all_vr_digits, all_pts_digits, all_pts_signs, all_user_ranks = load_images( glob.glob( 'images-redacted\\*.jpg'))
```


```python
type(all_vr_digits), all_vr_digits.shape # height,width,ndigit,nplayer,nimages
```




    (numpy.ndarray, (23, 15, 5, 12, 747))



#### This also returns a list of the user's ranks for each race (rank goes from 0-11, not 1-12)


```python
type(all_user_ranks), len(all_user_ranks)
```




    (list, 747)



## Label your own training data
You don't need to do this, but it's here if you want to for some reason. You can label your own data any way you like, but I used [pigeon](https://github.com/agermanidis/pigeon) and thought it was pretty convenient. The MKImageLabeler class is provided to help with this:


```python
from MKImageLabeler import MKImageLabeler
mklbl = MKImageLabeler( glob.glob( 'images-redacted\\*.jpg'))
```

#### Label  the digits for VR ratings (capped at 5 samples just for the example)


```python
mklbl.label_vr_digits( 5)
```

<img src='https://github.com/probablyfine/MK-OCR/raw/main/readme-img/label-vr-digits.png'>

#### Convert your labeled data to a pandas DataFrame


```python
df_vr_digits = mklbl.vr_digits_as_df()
df_vr_digits
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rank</th>
      <th>digit</th>
      <th>path</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>images-redacted\2020082314144200-16851BE00BC60...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>images-redacted\2020082314144200-16851BE00BC60...</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2</td>
      <td>images-redacted\2020082314144200-16851BE00BC60...</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>3</td>
      <td>images-redacted\2020082314144200-16851BE00BC60...</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>4</td>
      <td>images-redacted\2020082314144200-16851BE00BC60...</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_vr_digits.to_excel( 'vr_digits_labeled_example.xlsx', index=False)
```

#### Repeat for the digits and signs of points awarded


```python
mklbl.label_pts_digits( 5)
```

<img src='https://github.com/probablyfine/MK-OCR/raw/main/readme-img/label-pts-digits.png'>


```python
df_pts_digits = mklbl.pts_digits_as_df()
df_pts_digits
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rank</th>
      <th>digit</th>
      <th>path</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>images-redacted\2020082314144200-16851BE00BC60...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>images-redacted\2020082314144200-16851BE00BC60...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>images-redacted\2020082314144200-16851BE00BC60...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>images-redacted\2020082314144200-16851BE00BC60...</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>0</td>
      <td>images-redacted\2020082314144200-16851BE00BC60...</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
mklbl.label_pts_signs( 5)
```

<img src='https://github.com/probablyfine/MK-OCR/raw/main/readme-img/label-pts-signs.png'>


```python
df_pts_signs = mklbl.pts_signs_as_df()
df_pts_signs
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rank</th>
      <th>digit</th>
      <th>path</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>images-redacted\2020082314144200-16851BE00BC60...</td>
      <td>+</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>images-redacted\2020082314144200-16851BE00BC60...</td>
      <td>+</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>images-redacted\2020082314144200-16851BE00BC60...</td>
      <td>+</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0</td>
      <td>images-redacted\2020082314144200-16851BE00BC60...</td>
      <td>+</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0</td>
      <td>images-redacted\2020082314144200-16851BE00BC60...</td>
      <td>-</td>
    </tr>
  </tbody>
</table>
</div>



## License
TBD
