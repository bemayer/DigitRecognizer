
# ------------------------------------------------------------------------------
# Stratified sampling
library(splitstackshape)
set.seed(42)
strat <- stratified(data_train, c('class'), 0.7, bothSets = TRUE)
X_train <- strat[[1]] %>% dplyr::select(dplyr::matches("pix*"))
X_validation <- strat[[2]] %>% dplyr::select(dplyr::matches("pix*"))
X_test <- data_test %>% dplyr::select(dplyr::matches("pix*"))
y_train <- strat[[1]][["class"]]
y_validation <- strat[[2]][["class"]]
y_test <- data_test[["class"]]
prop.table(table(y_train))
prop.table(table(y_validation))
prop.table(table(y_test))


# ------------------------------------------------------------------------------
# kNN
pred_train <- knn(X_train, X_train, y_train, k=4)
pred_validation <- knn(X_train, X_validation, y_train, k=4)
pred_test <- knn(X_train, X_test, y_train, k=4)
matrix_train <- confusionMatrix(y_train, pred_train)
matrix_validation <- confusionMatrix(y_validation, pred_validation)
matrix_test <- confusionMatrix(y_test, pred_test)
matrix_train[["overall"]][["Accuracy"]]
matrix_validation[["overall"]][["Accuracy"]]
matrix_test[["overall"]][["Accuracy"]]
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# RDA
rdaParam <- data.frame(gamma = 0.1, lambda = 0.05)
model <- caret::train(y=y_train,
					  x=X_train,
					  method="rda",
					  trControl=fitControl,
					  tuneGrid = rdaGrid)
pred_train <- predict(model, X_train)
pred_validation <- predict(model, X_validation)
pred_test <- predict(model, X_test)
matrix_train <- confusionMatrix(y_train, pred_train)
matrix_validation <- confusionMatrix(y_validation, pred_validation)
matrix_test <- confusionMatrix(y_test, pred_test)
matrix_train[["overall"]][["Accuracy"]]
matrix_validation[["overall"]][["Accuracy"]]
matrix_test[["overall"]][["Accuracy"]]
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# GPS
dist_eucl_train <- proxy:::dist(X_train/6, X_train/6)
dist_eucl_validation <- proxy:::dist(X_validation/6, X_train/6)
dist_eucl_test <- proxy:::dist(X_test/6, X_train/6)
eps = 4.6
model <- protoclass(y = y_train, dxz = dist_eucl_train,
					eps=eps, lambda = 1/dim(X_train)[1])
pred_train <- predictwithd.protoclass(model, dist_eucl_train)
pred_validation <- predictwithd.protoclass(model, dist_eucl_validation)
pred_test <- predictwithd.protoclass(model, dist_eucl_test)
matrix_train <- confusionMatrix(y_train, pred_train)
matrix_validation <- confusionMatrix(y_validation, pred_validation)
matrix_test <- confusionMatrix(y_test, pred_test)
matrix_train[["overall"]][["Accuracy"]]
matrix_validation[["overall"]][["Accuracy"]]
matrix_test[["overall"]][["Accuracy"]]
cat("acc: ", matrix_test[["overall"]][["Accuracy"]], "\n")
# ------------------------------------------------------------------------------
