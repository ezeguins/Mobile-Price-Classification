# Mobile-Price-Classification
Techgig Practice Problem

# Problem Statement:
Prabhat has started his own mobile company. He wants to give tough fight to big companies like Apple,Samsung etc.
He does not know how to estimate price of mobiles his company creates. In this competitive mobile phone market you cannot simply assume things. To solve this problem he collects sales data of mobile phones of various companies.
Prabhat wants to find out some relation between features of a mobile phone(eg:- RAM,Internal Memory etc) and its selling price. But he is not so good at Machine Learning. So he needs your help to solve this problem.
In this problem you do not have to predict actual price but a price range indicating how high the price is.

Datasets
# test.csv
The test set should be used to see how well your model performs on unseen data. For the test set we do not provide the outcome(target variable) for each mobile sales data point. For each data point in the test set use the model you trained to predict price range.
# train.csv
The training set should be used to build your machine learning models. For the training set we provide the outcome( also known as target or class label) for each mobile sales data point.

Data Dictionary
Here's a brief version of what you'll find in the data description file.

# Variable	Description
pc  - Primary Camera mega pixels

fc	- Front Camera mega pixels

sc_h	- Screen Height of mobile in cm

sc_w	- Screen Width of mobile in cm

m_dep	- Mobile Depth in cm

px_width	- Pixel Resolution Width

px_height	- Pixel Resolution Height

ram	- Random Access Memory in Mega Bytes

int_memory	- Internal Memory in Giga Bytes

four_g	- Has 4G or not

three_g	- Has 3G or not

dual_sim	- Has dual sim support or not

battery_power	- Total energy a battery can store in one time measured in mAh

touch_screen	- Has touch screen or not

clock_speed	- speed at which microprocessor executes instructions

n_cores	- Number of cores of processor

wifi	- Has wifi or not

blue	- Has bluetooth or not

mobile_wt	- Weight of mobile phone

talk_time	- longest time that a single battery charge will last when you are

price_range	- This is the target variable with value of 0(low cost), 1(medium cost), 2(high cost) and 3(very high cost).
