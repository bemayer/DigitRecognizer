# ------------------------------------------------------------------------------
# Libraries
library(caret)
library(dplyr)
library(readr)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Read data
data_test <- read.csv("../Data/data_test_completed.csv")
data_test[["class"]] <- as.factor(make.names(data_test[["class"]]))
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Load models and predict on test set
file_names <- dir("Model", pattern =".rds")
for(i in 53:length(file_names)){
	print(file_names[i])
	model <- readRDS(paste0("../Model/", file_names[i]))
	pred_test <- predict(model,newdata=subset(data_test, select = -c(class)))
	matrix <- confusionMatrix(data_test[["class"]], pred_test)
	cat(c(model[["method"]],
				round(matrix[["overall"]][1], 5),"\n"),
			file="../Log/log_test.txt",
			sep = "\t",
			append = TRUE)
}
# ------------------------------------------------------------------------------
