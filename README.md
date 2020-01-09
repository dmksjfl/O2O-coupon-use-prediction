# O2O-coupon-use-prediction
This is an implement of o2o coupon use prediction competition in Tianchi. You could find more infomation there. 

We get a score of 0.6377 with pure offline dataset.

# Scores in Tianchi
We hand in our result many times and the best score we get is 0.6377 and you can find it via the following picture.

![The scores we get in Tianchi](https://github.com/dmksjfl/O2O-coupon-use-prediction/blob/master/bestscore.png)

# How to use
One should first download datasets in Tianchi and then run the following code.
```python
python feature_extract.py
```
You could get training set and test set with the above code. You would get o2o_train.csv and o2o_test.csv. 
Note that it consumes time to handle the feature engineering. We run it on GPU server and it consumes about 3 hours.
After that, you should run the following code to train your model.
```python
python model.py
```
It trains tree models like regression tree, random forest, deep forest; and boosting models like XGBoost, LightGBM and so on.
It also consumes time to run the model (about four hours in GPU server).
If you do not want to train the model, you could see that there are trained model in saved_model file and you should run:
```python
python result.py
```
