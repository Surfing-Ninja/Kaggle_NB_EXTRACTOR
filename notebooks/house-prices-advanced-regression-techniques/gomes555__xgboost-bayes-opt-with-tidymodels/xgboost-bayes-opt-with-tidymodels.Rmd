---
title: "XGBoost bayes opt with tidymodels"
author: "Fellipe Gomes"
date: "`r format(Sys.time(), '%d %B, %Y')`"
output:
  html_document:
    toc: true
    toc_depth: 2
    number_sections: true
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE, cache = F, message = F, warning = F, comment = F)
options(scipen = 999999)
```

<p align="right"><span style="color:firebrick">Dont forget the upvote if you liked the post! <i class="fas fa-hand-peace"></i></span> </p>

# Problem definition

The data presented in this challenge are widely used in data science education and illustrate a problem of regression. The variable we want to estimate is numeric.

More information about data: <http://jse.amstat.org/v19n3/decock.pdf>

<center>![](https://gomesfellipe.github.io/img/2018/08/img1.png){width=80%}</br><small>Source: <https://gomesfellipe.github.io/> (my blog)</small></center>

The solution for this competition will be based on an old post on my blog about tree-based machine learning models.

See the post at: <https://gomesfellipe.github.io/post/2018-08-31-modelos-em-arvore/modelos-em-arvore/>

# Data Science framework

Lets follow a well-known framework in data science by R users:

<center>![](https://d33wubrfki0l68.cloudfront.net/571b056757d68e6df81a3e3853f54d3c76ad6efc/32d37/diagrams/data-science.png)</br><small>Source: <https://r4ds.had.co.nz/introduction.html></small></center>

Dependencies:

```{r, include = F}
devtools::install("../input/r-visdat-package/visdat/", dependencies = F)
library(visdat)
library(tidyverse)
library(tidymodels)
library(patchwork)

theme_set(theme_bw())

ncores <- 4
```

```{r, eval = F}
library(visdat)
library(tidyverse)
library(tidymodels)
library(patchwork)

theme_set(theme_bw())

ncores <- 4
```


# Import data

Loading train and test data:

```{r}
train_data <- read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test_data <- read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
```

# Tidy

Before modeling, lets check out some information about the data structure:

## Data Structure

Take a look at the data structure:

```{r}
vis_dat(train_data)
```

There are columns with many missing values

## Percentages

Percentage of data types

```{r}
DataExplorer::plot_intro(train_data)
```

Most of the data is discrete

## Missing data

Take a look at the columns that have missing data:

```{r}
p <- DataExplorer::plot_missing(train_data, missing_only = T)
cols_to_remove <- p$data %>% filter(Band %in% c("Remove", "Bad")) %>% pull(feature)
```

Columns to remove: `r cols_to_remove`

## Target

The target variable in this problem is numeric. Lets see what the behavior of your distribution is

```{r}
# hist
g1 <- ggplot(train_data, aes(x=SalePrice)) + 
 geom_histogram(aes(y=..density..), colour="black", fill="white")+
 geom_density(alpha=.2, fill="#FF6666")+ 
  labs(x = "", y = "")

# boxplot
g2 <- ggplot(train_data, aes(y=SalePrice)) + 
 geom_boxplot(aes(x=""), colour="black", fill="white")+
  coord_flip()+ 
  labs(x = "", y = "")

# qqplot
g3 <- ggplot(train_data, aes(sample = SalePrice))+ 
  stat_qq()+
  stat_qq_line()+ 
  labs(x = "", y = "")
  
g3 | g1 / g2 
```

What if we apply the log transformation?

```{r}
train_data <- train_data %>% mutate(SalePrice = log(SalePrice))

# hist
g1 <- ggplot(train_data, aes(x=SalePrice)) + 
 geom_histogram(aes(y=..density..), colour="black", fill="white")+
 geom_density(alpha=.2, fill="#FF6666")+ 
  labs(x = "", y = "")

# boxplot
g2 <- ggplot(train_data, aes(y=SalePrice)) + 
 geom_boxplot(aes(x=""), colour="black", fill="white")+
  coord_flip()+ 
  labs(x = "", y = "")

# qqplot
g3 <- ggplot(train_data, aes(sample = SalePrice))+ 
  stat_qq()+
  stat_qq_line()+ 
  labs(x = "", y = "")
  
g3 | g1 / g2 
```

# Transform

We arrived at the moment of analysis when we need to apply transformations to the data so that they reflect the study in a more realistic way

The same pre-processing applied to the training data needs to be applied to the test data. For this task we will use the `recipes` package of the `tidymodels` framework to prepare a transformation recipe.

```{r}
SalePrice_recipe <- recipe(train_data, SalePrice ~ .) %>%
  step_rm(Id, Street, Utilities) %>% 
  step_rm(one_of(cols_to_remove)) %>%
  step_log(all_numeric(),-all_outcomes(), offset = 1) %>%
  step_normalize(all_numeric(),-all_outcomes()) %>%
  step_other(all_nominal(), -all_outcomes(), threshold = 0.01) %>%
  step_novel(all_predictors(), -all_numeric()) %>%
  step_knnimpute(all_predictors()) %>%
  step_dummy(all_nominal(), -all_outcomes()) 

# prep(SalePrice_recipe)
# juice(prep(SalePrice_recipe))
# SalePrice_bake <- bake(prep(SalePrice_recipe), new_data = test_data)
```

# Model / Visualise

Define model:

```{r}
SalePrice_xgb_model <- 
  boost_tree(
  trees = tune(), learn_rate = tune(),
  tree_depth = tune(), min_n = tune(),
  loss_reduction = tune(), 
  sample_size = tune(), mtry = tune(), 
  ) %>% 
  set_mode("regression") %>% 
  set_engine("xgboost", nthread = ncores)
```

Define workflow:

```{r}
SalePrice_workflow <- workflow() %>% add_recipe(SalePrice_recipe)

SalePrice_xgb_workflow <-SalePrice_workflow %>% add_model(SalePrice_xgb_model)
```

Define params:

```{r}
xgboost_params <- parameters(
  trees(), learn_rate(),
  tree_depth(), min_n(), 
  loss_reduction(),
  sample_size = sample_prop(), finalize(mtry(), train_data)  
)

xgboost_params <- xgboost_params %>% update(trees = trees(c(100, 500))) 

```

Define folds to cross validation:

```{r}
set.seed(123)
SalePrice_vfold <- vfold_cv(train_data, v = 5, strata = SalePrice)
```

Increment workflow:

```{r}
workflow_SalePrice_xgb_model <- 
  workflow() %>% 
  add_model(SalePrice_xgb_model) %>% 
  add_recipe(SalePrice_recipe)
```

Iterative Bayesian optimization:

```{r}
# doParallel::registerDoParallel(ncores)

set.seed(321)
xgboost_tune <-
  workflow_SalePrice_xgb_model %>%
  tune_bayes(
    resamples = SalePrice_vfold,
    param_info = xgboost_params,
    # initial = ?,
    iter = 30, 
    metrics = metric_set(rmse, mape),
    control = control_bayes(no_improve = 10, 
                            save_pred = T, verbose = T)
  )

# doParallel::stopImplicitCluster()
autoplot(xgboost_tune)
```

Check results:

```{r}
SalePrice_best_model <- select_best(xgboost_tune, "rmse", maximize = F)
print(SalePrice_best_model)
```

Finalize model:

```{r}
SalePrice_final_model <- finalize_model(SalePrice_xgb_model, SalePrice_best_model)
SalePrice_workflow    <- workflow_SalePrice_xgb_model %>% update_model(SalePrice_final_model)
SalePrice_xgb_fit     <- fit(SalePrice_workflow, data = train_data)
```

Evalue model:

```{r}
pred <- 
  predict(SalePrice_xgb_fit, test_data) %>% 
  mutate(modelo = "XGBoost",
         .pred = exp(.pred)) %>% 
  bind_cols(read_csv("../input/saleprice-full-score/full-score.csv") %>% select(SalePrice))

g1 <- 
  pred %>% 
  ggplot(aes(x = .pred, y = SalePrice))+
  geom_point()+ 
  geom_abline(intercept = 0, col = "red")


g2 <- 
  pred %>% 
  select(.pred, SalePrice) %>% 
  gather(key, value) %>% 
  ggplot(aes(x=value, volor = key, fill = key)) + 
  geom_density(alpha=.2)+ 
  labs(x = "", y = "")

g1 / g2
```

# Communicate

In this case of a data science project, the communication of our results will be via submission:

```{r}
read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv") %>%
  select(-SalePrice) %>% 
  bind_cols(pred %>% transmute(SalePrice = .pred)) %>% 
  write_csv("submission.csv")
```

