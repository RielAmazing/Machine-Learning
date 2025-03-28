**Introduction**

As art enthusiasts, we have decided to work on a set of data regarding art auction
valuation. The value of an artwork is extremely subjective and appears to be
unmeasurable. Famous artists like Pablo Picasso and Andy Warhol have had artworks
valued and sold for over $100 million, where other lesser known artists have had
artworks valued and sold for $20. Leading to the question, how much does an artist's
reputation and notoriety influence the price/value of an artwork? Does the value of
artwork take into account the features and physical qualities of said piece or is it solely
based on the artist name alone? With the data in hand, we would want to apply our
machine learning skills to produce a model that can predict the value of an art piece
using the physical qualities of the piece like height, width, materials used, etc.. We find
this project fairly interesting since it can allow art buyers and sellers to estimate the
price of the art piece in hand. Furthermore allowing for the possibility of a machine
learning model to become an asset to auction houses and artists.

**Related Work

Some of the previous work done in art valuation with machine learning, has used a
neural network model to predict the price of artwork at an auction1. They built this
model on physical characteristics, visible and non-visible, and used auction house
estimates to further understand an auctioneer’s accuracy when valuing an artwork.
Another article by Jason Bailey, discusses the subjectivity of artwork valuation and the
necessity for a machine learning model in artwork valuation2. Although machine
learning models are not as accurate, Bailey claims that building a machine learning
model can increase the speed, cost, volume, and frequency of valuing an artwork.

**Data Description**

The dataset is acquired from GitHub3, the original dataset contains 41,253 objects with
23 columns(variables). The data was collected by user jasonshi10 6 years ago, where he
webscrapped the data from ArtInfo.com as a txt file. We have concised the dataset to 17
columns removing columns including X, yearofBirth, yearofDeath, link, source, and soldTime, which we find irrelevant in conducting our prediction. For now, our dataset
consists of both character variables ("artist","country", "name", "material", and
"dominantColor") and numerical variables("year", "price","height", “width","brightness",
"ratioUniqueColors", "thresholdBlackPerc", "highbrightnessPerc", "lowbrightnessPerc",
"CornerPer", "EdgePer", and "FaceCount" ) these variables have depicted the elements that
make an art piece valuable. Through data visualization, we saw that the variable year
has a 32% miss rate of data, while height, and width are 6% each, which is not a great
concern since it's still under the 30% benchmark. In total, 3.4% of our dataset contains
missing values (Figure 1).
Price, our response variable, has a maximum value of $119.92 million and minimum
value of $20, with the sum of price being $9.47 billion. When visualizing the
distribution of our response variable price, we noticed it was heavily skewed. Thus we
normalized our price variable by taking the natural log of price (Figure 2). We have also
done a series of visualizations in seeing the relationship between the log of price and
materials. Discovering that the materials used and price do have an impact between
each other (Figure 3).
With the addition of the natural log of price, we converted the material feature into
individual features by taking the top ten most common materials like ‘canvas’, ‘oil’,
‘watercolor’, etc. and making them binary (Figure 4).
Furthermore, since we would like to understand if the artist name is influential to the
art work, we have decided to create a binary column of the artist by splitting them into
two categories, which are well known artists and lesser known artists. We define well
known artists as the average value of the artist artwork is greater than or equal to
$241,528.50 and lesser known artists as the average price of the artist’s artwork is less
than $241,528.50. We decided the threshold of well known and lesser known artists by
determining the average price of artwork in the dataset and comparing the artist’s
average artwork price. In total, there are 8,608 unique artists in the data set with 56 of
them being well known artists. By doing so, we are able to compare and see the impact
of an artist’s fame affects the pricing of an art piece in an auction.

**Methods & Results**

Before building our models, we wanted to determine the best way to deal with the
missing values in the dataset. The two features that contained missing data are ‘height’
and ‘width’. Since we believed these were important values for building our model, we
decided to use MICE imputation to handle our missing data. After completing MICE
imputation, we split our dataset into a 8:2, training and test dataset to continue with
building our machine learning models.
The team has run linear regression, BIC, random forest, and XGBoost models, to see
which methods provide the lowest Root Mean Square Error (RMSE) result, with our
dependent/response variable the Natural Log of Price. The team first applied the whole
dataset to the models and tuned the hyperparameters of the XGBoost model with
cross-validation, to decrease the RMSE and build up a more model.
The team has built various models including linear regression, random forest, and
XGBoost. The first linear regression model we ran provided a RMSE of 2,436,931. Next
we ran a random forest model, this model produced an RMSE of 2,197,950. Next, we
decided to run an XGBoost model with no parameters set, which produced the lowest
RMSE of 2,187,908.
After running the initial XGBoost model, we wanted to see if we could improve our
model by tuning some of the parameters. These parameters are the max.depth, nrounds,
eta, gamma, min_child_weight, subsample, and the colsample_bytree. After tuning
these parameters, we ran our final model. The tuned XGBoost Model produced an
RMSE of 2,228,431. One of our main concerns was the increased RMSE score from our
tuned model. This increase suggests the model has overfit or the parameters need to be
re-evaluated.
Furthermore, through XGBoost, we have discovered that the top 3 key variables are
well_known, canvas, and oil and have done further investigation into it. After the first
test of our XGBoost, we extracted the variable importance. The team decided to remove
the insignificant variables (‘dominant colors’, ‘brightness’, ‘higherbrightnessPerc’,
‘CornerPer’, ‘facecount’, and most material features) and see if the removal of these
features will lower the RMSE value. We ran an XGBoost model with the ten key features,
this produced an RMSE of 2,216,996. The lower RMSE value suggests our original
training data variables have too much noise.
Another concern after trying various models was the high Root Mean Square Error
(RMSE) values for the models we built. The RMSE indicates the average difference of
the predicted price and the actual price. The high RMSE values of our models suggests
the models struggle with predicting lower and higher priced values, with a potential of
overfitting for the middle-range price values (Figure 5-6). This performance indicated
that the model handled middle-range artwork prices relatively well but struggled with
artworks at both the lower and higher ends of the price spectrum.

**Discussion & Future Work**

Through numerous tests and model building we have discovered that the artists’ fame is
a crucial variable to our dataset and greatly impacts the valuation of artwork (Figure 7).
With the importance variables extracted from XGBoost, indicated that our well_known
variable is the most impactful variable in our dataset.
However, it is likely the large range of price values in our dataset led to an increasingly
high RMSE for our predictions. With the large range of our price feature, it is difficult to
accurately train a model to predict the accurate price of an art piece. The large RMSE
values suggest that the model is heavily influenced by higher priced artwork and
performs poorly on smaller value artwork.
In the future, when one encounters such a huge dataset with huge gaps between prices,
we suggest collecting quantitative data for artist reputation, such as counts of search
engine results, views of their individual wikipedia pages, scraping websites, social media
presence, and conducting surveys4. By quantifying the fame of an artist it allows
training models to have a clearer and more accurate reference on the artist's notoriety,
which could provide more precise results on future valuation predictions on artwork.
Another way is to subset the original data into multiple datasets based on the artist's
reputation. An example of this with the current data is to use price to subset the artists’
fame into: Worldly-known, Known, Not-so-Known, and Unknown. By doing so it
provides the training models with a more concise dataset, which supports the training
models to have a zoom in on each artist notoriety category, allowing the training model
to have a more relatively related dataset to be trained on. The subsetting of the dataset
grants each dataset to have a more similar attribute in each variable, which would likely
decrease the RMSE and for the training model to have a more accurate result on their
predictions.

**Conclusion**

In conclusion, we have realized that artist fame is a key feature in predicting artwork
price. It would be valuable to gather more quantifiable information on the artist's
popularity. With more information on the artist’s popularity, the result of the prediction
will be more precise and provide a more indepth look into art piece price in auction
houses.

**Reference**

1 Aubry, Mathieu and Kraeussl, Roman and Manso, Gustavo and Spaenjers, Christophe, Biased
Auctioneers (January 6, 2022). Journal of Finance, Forthcoming, Available at SSRN:
https://ssrn.com/abstract=3347175 or http://dx.doi.org/10.2139/ssrn.3347175
2 Jason Bailey.(2020).Can Machine Learning Predict the Price of Art at Auction?
https://www.artnome.com/news/2020/5/5/can-machine-learning-predict-the-price-of-art-at-a
uction
3 GitHub: art_auction_valuation, https://github.com/jasonshi10/art_auction_valuation
4 Edward D Ramirez , Stephen J Hagen, The quantitative measure and statistical distribution of
fame.(July 6, 2018). https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0200196
