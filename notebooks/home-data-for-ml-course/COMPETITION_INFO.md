# Home Data For Ml Course

## Competition Information

**Competition Name:** home-data-for-ml-course  
**Kaggle URL:** https://www.kaggle.com/competitions/home-data-for-ml-course  
**Data Collected:** 2025-11-11 15:54:25

## Description

This competition focuses on machine learning and data science challenges. The notebooks in this folder contain various approaches and solutions submitted by Kaggle community members.



## Dataset Information

### Files Available

```
name                            size  creationDate                
------------------------  ----------  --------------------------  
data_description.txt           13370  2019-11-30 13:52:25.068000  
sample_submission.csv          31939  2019-11-30 13:52:24.982000  
sample_submission.csv.gz       15685  2019-11-30 13:52:24.975000  
test.csv                      451405  2019-11-30 13:52:25.228000  
test.csv.gz                    83948  2019-11-30 13:52:25.179000  
train.csv                     460676  2019-11-30 13:52:25.134000  
train.csv.gz                   91387  2019-11-30 13:52:25.063000  

```


### Dataset Columns

| Column Name | Data Type | Sample Values | Unique Values |
|-------------|-----------|---------------|---------------|
| `Id` | int64 | 1461, 1462 | 5 |
| `MSSubClass` | int64 | 20, 20 | 3 |
| `MSZoning` | object | RH, RL | 2 |
| `LotFrontage` | int64 | 80, 81 | 5 |
| `LotArea` | int64 | 11622, 14267 | 5 |
| `Street` | object | Pave, Pave | 1 |
| `Alley` | float64 |  | 0 |
| `LotShape` | object | Reg, IR1 | 2 |
| `LandContour` | object | Lvl, Lvl | 2 |
| `Utilities` | object | AllPub, AllPub | 1 |
| `LotConfig` | object | Inside, Corner | 2 |
| `LandSlope` | object | Gtl, Gtl | 1 |
| `Neighborhood` | object | NAmes, NAmes | 3 |
| `Condition1` | object | Feedr, Norm | 2 |
| `Condition2` | object | Norm, Norm | 1 |
| `BldgType` | object | 1Fam, 1Fam | 2 |
| `HouseStyle` | object | 1Story, 1Story | 2 |
| `OverallQual` | int64 | 5, 6 | 3 |
| `OverallCond` | int64 | 6, 6 | 2 |
| `YearBuilt` | int64 | 1961, 1958 | 5 |
| `YearRemodAdd` | int64 | 1961, 1958 | 4 |
| `RoofStyle` | object | Gable, Hip | 2 |
| `RoofMatl` | object | CompShg, CompShg | 1 |
| `Exterior1st` | object | VinylSd, Wd Sdng | 3 |
| `Exterior2nd` | object | VinylSd, Wd Sdng | 3 |
| `MasVnrType` | object | BrkFace, BrkFace | 1 |
| `MasVnrArea` | int64 | 0, 108 | 3 |
| `ExterQual` | object | TA, TA | 2 |
| `ExterCond` | object | TA, TA | 1 |
| `Foundation` | object | CBlock, CBlock | 2 |
| `BsmtQual` | object | TA, TA | 2 |
| `BsmtCond` | object | TA, TA | 1 |
| `BsmtExposure` | object | No, No | 1 |
| `BsmtFinType1` | object | Rec, ALQ | 3 |
| `BsmtFinSF1` | int64 | 468, 923 | 5 |
| `BsmtFinType2` | object | LwQ, Unf | 2 |
| `BsmtFinSF2` | int64 | 144, 0 | 2 |
| `BsmtUnfSF` | int64 | 270, 406 | 5 |
| `TotalBsmtSF` | int64 | 882, 1329 | 5 |
| `Heating` | object | GasA, GasA | 1 |
| `HeatingQC` | object | TA, TA | 3 |
| `CentralAir` | object | Y, Y | 1 |
| `Electrical` | object | SBrkr, SBrkr | 1 |
| `1stFlrSF` | int64 | 896, 1329 | 5 |
| `2ndFlrSF` | int64 | 0, 0 | 3 |
| `LowQualFinSF` | int64 | 0, 0 | 1 |
| `GrLivArea` | int64 | 896, 1329 | 5 |
| `BsmtFullBath` | int64 | 0, 0 | 1 |
| `BsmtHalfBath` | int64 | 0, 0 | 1 |
| `FullBath` | int64 | 1, 1 | 2 |
| `HalfBath` | int64 | 0, 1 | 2 |
| `BedroomAbvGr` | int64 | 2, 3 | 2 |
| `KitchenAbvGr` | int64 | 1, 1 | 1 |
| `KitchenQual` | object | TA, Gd | 2 |
| `TotRmsAbvGrd` | int64 | 5, 6 | 3 |
| `Functional` | object | Typ, Typ | 1 |
| `Fireplaces` | int64 | 0, 0 | 2 |
| `FireplaceQu` | object | TA, Gd | 2 |
| `GarageType` | object | Attchd, Attchd | 1 |
| `GarageYrBlt` | int64 | 1961, 1958 | 5 |
| `GarageFinish` | object | Unf, Unf | 3 |
| `GarageCars` | int64 | 1, 1 | 2 |
| `GarageArea` | int64 | 730, 312 | 5 |
| `GarageQual` | object | TA, TA | 1 |
| `GarageCond` | object | TA, TA | 1 |
| `PavedDrive` | object | Y, Y | 1 |
| `WoodDeckSF` | int64 | 140, 393 | 5 |
| `OpenPorchSF` | int64 | 0, 36 | 4 |
| `EnclosedPorch` | int64 | 0, 0 | 1 |
| `3SsnPorch` | int64 | 0, 0 | 1 |
| `ScreenPorch` | int64 | 120, 0 | 3 |
| `PoolArea` | int64 | 0, 0 | 1 |
| `PoolQC` | float64 |  | 0 |
| `Fence` | object | MnPrv, MnPrv | 1 |
| `MiscFeature` | object | Gar2 | 1 |
| `MiscVal` | int64 | 0, 12500 | 2 |
| `MoSold` | int64 | 6, 6 | 3 |
| `YrSold` | int64 | 2010, 2010 | 1 |
| `SaleType` | object | WD, WD | 1 |
| `SaleCondition` | object | Normal, Normal | 1 |


### Dataset Location

The competition dataset is available in the `dataset/` subdirectory of this folder.

## Notebooks

This folder contains 99 downloaded notebooks from this competition.

### How to Use

1. **View Notebooks**: Browse the subdirectories to find individual notebooks
2. **Access Dataset**: Check the `dataset/` folder for competition data files
3. **Run Notebooks**: Install required dependencies and run notebooks in your environment

## API Information

```
ref                                                          deadline             category            reward  teamCount  userHasEntered  
-----------------------------------------------------------  -------------------  ---------------  ---------  ---------  --------------  
https://www.kaggle.com/competitions/home-data-for-ml-course  2030-01-01 23:59:00  Getting Started  Knowledge       5373           False  

```

## Attribution

All notebooks and datasets are from Kaggle's public repository.  
Please attribute original authors when using their work.

---
*Generated by Kaggle Notebook Extractor*  
*Repository: https://github.com/Surfing-Ninja/Kaggle_NB_EXTRACTOR*
