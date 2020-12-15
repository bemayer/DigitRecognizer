library(dplyr)
library(readr)
library(stringr)
library(caret)


# Read, convert and subset the data
load("./Data/data_train.rda")
load("./Data/data_test.rda")
X_train <- data_train %>% select(matches("pix*"))
X_test <- data_test %>% select(matches("pix*"))
# Numerical factors not accepted by Keras 
# Error: "Please use factor levels that can be used as valid R variable names"
data_train[["class"]] <- as.factor(make.names(data_train[["class"]]))
y_train <- data_train[["class"]]

# Control using cross-validation
fitControl <- trainControl(method="cv", 
                           number=3,
                           verboseIter=TRUE)

# Hyperparameters tunning
# What value to affect to tuneLength ??
modelFit <- caret::train(y=y_train, 
                         x=X_train,
                         method="svmRadial",
                         trControl=fitControl,
                         tuneLength = 10)

# Accuracy check on train sample
pred.test <- predict(modelFit,newdata=subset(data_train, select = -c(class)))
matrix <- confusionMatrix(data_train[["class"]], pred.test)
cat(c(format(Sys.time(), "%d/%m/%Y %H:%M"),
      modelFit[["method"]],
      round(matrix[["overall"]][1], 5),"\n"),
    file="./Log/log.txt",
    sep = "\t",
    append = TRUE)

# Predict test sample
pred <- predict(modelFit,newdata=X_test) %>% str_remove("[X]")
write.table(pred, 
            paste0("./Pred/pred_",modelFit[["method"]],"_",format(Sys.time(), "%y%m%d%H%M"),".csv"),
            col.names=FALSE,
            row.names=FALSE,
            quote = FALSE)

# Save model
saveRDS(modelFit,
        paste0("./Model/",modelFit[["method"]],"_",format(Sys.time(), "%y%m%d%H%M"),".rds"))
