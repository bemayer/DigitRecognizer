# ------------------------------------------------------------------------------
# Libraries
library(caret)
library(dplyr)
library(readr)
library(class)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Read and convert the data
data_test <- read_csv("../../Data/data_test_completed.csv")
data_test[["class"]] <- as.factor(make.names(data_test[["class"]]))
y_test <- data_test[["class"]]
load("../../Data/data_train.rda")
data_train[["class"]] <- as.factor(make.names(data_train[["class"]]))
y_train <- data_train[["class"]]
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Confusion matrix for k=5, best for caret
model <- readRDS(paste0("../../Model/knn_2012180031.rds"))
X_train <- data_train %>% dplyr::select(dplyr::matches("pix*")) %>% scale()
X_test <- data_test %>% dplyr::select(dplyr::matches("pix*")) %>% scale()
pred_test <- knn(X_train, X_test, y_train, k=5)
matrix_test <- confusionMatrix(y_test, pred_test)
matrix_test[["overall"]][1]
pred_test %>% as_tibble() %>% mutate(id = as.character(row_number() - 1)) %>%
  filter(pred_test != y_test)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Find 5 nearest neighbours of img #9
dist_test <- as.data.frame.matrix(proxy:::dist(X_test, X_train))
dist_test[10, ][order(dist_test[10, ])[1:5]]
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Retrain the model on all groups
acc_retrain <- tibble(
  col = character(),
  k = numeric(), 
  acc_test = numeric())
col_groups <- c("fac*", "fou*", "kar*", "pix*", "zer*", "_")
names(col_groups) <- c("Correlation", "Fourier", "Karhunen-Loève", "Pixels", 
                       "Moments de Zernike", "Tous")
for (grp in col_groups)
{
  X_train <- data_train %>% dplyr::select(dplyr::matches(grp)) %>% scale()
  X_test <- data_test %>% dplyr::select(dplyr::matches(grp)) %>% scale()
  for (k in 1:500)
  {
    brk <- FALSE
    tryCatch(pred_test <- knn(X_train, X_test, y_train, k=k), 
             error = function(e) {brk <- TRUE})
    if (brk) {break}
    matrix_test <- confusionMatrix(y_test, pred_test)
    acc_retrain <- acc_retrain %>% add_row(
      col = names(col_groups)[col_groups == grp],
      k = k,
      acc_test = matrix_test[["overall"]][1])
  }
}
write_csv(acc_retrain, "acc_retrain.csv")
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Plot accuracy
ggplot(acc_retrain, aes(x = k, y = acc_test, color = col)) + 
  geom_line() +
  scale_x_log10(name = "k") +
  scale_y_continuous(name = "Précision sur l'échantillon de test", 
                     labels = scales::percent) +
  scale_color_discrete(name = "Légende") +
  theme_light() +
  theme(legend.justification=c(0,0), 
        legend.position=c(0.01,0.01), 
        legend.box.background = element_rect(colour = "black"))
# ------------------------------------------------------------------------------
