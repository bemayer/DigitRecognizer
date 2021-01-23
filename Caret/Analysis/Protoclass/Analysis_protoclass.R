# ------------------------------------------------------------------------------
# Libraries
library(dplyr)
library(readr)
library(caret)
library(protoclass)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Read, convert and subset the data
data_test <- read_csv("../../Data/data_test_completed.csv")
X_test <- data_test %>% dplyr::select(dplyr::matches("pix*"))
data_test[["class"]] <- as.factor(make.names(data_test[["class"]]))
y_test <- data_test[["class"]]
load("../../Data/data_train.rda")
X_train <- data_train %>% dplyr::select(dplyr::matches("pix*"))
data_train[["class"]] <- as.factor(make.names(data_train[["class"]]))
y_train <- data_train[["class"]]
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Confusion matrix for best caret model
model <- readRDS(paste0("../../Model/protoclass_2101081027.rds"))
pred_test <- predict(model,X_test)
matrix_test <- confusionMatrix(y_test, pred_test)
pred_train <- predict(model,X_train)
matrix_train <- confusionMatrix(y_train, pred_train)
matrix_train[["overall"]]
pred_test %>% as_tibble() %>% mutate(id = as.character(row_number() - 1)) %>%
	filter(pred_test != y_test)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Retrain the model
acc_retrain <- tibble(
	eps = numeric(),
	n_proto = numeric(),
	acc_train = numeric(),
	acc_test = numeric())
dist_eucl_train <- proxy:::dist(X_train/6, X_train/6)
dist_eucl_test <- proxy:::dist(X_test/6, X_train/6)
for (eps in seq(0.1, 8.9, 0.1))
{
	model <- protoclass(y = y_train, dxz = dist_eucl_train,
	eps=eps, lambda = 1/dim(X_train)[1])
	pred_train <- predictwithd.protoclass(
			model,
			dist_eucl_train)
	pred_test <- predictwithd.protoclass(
			model,
			dist_eucl_test)
	matrix_train <- confusionMatrix(
			y_train,
			pred_train)
	matrix_test <- confusionMatrix(
			y_test,
			pred_test)
	acc_retrain <- acc_retrain %>%
	add_row(eps = eps, n_proto = sum(model$nproto),
			acc_train = matrix_train[["overall"]][1],
			acc_test = matrix_test[["overall"]][1])
	matrix_train[["table"]]
}
write_csv(acc_retrain, "acc_retrain.csv")
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Graph of precisions and epsilon
scale = 1500
colors <- c("Nombre de prototypes" = "black",
			"Précision entrainement" = "blue",
			"Précision test" = "red")
ggplot(acc_retrain, aes(x = eps)) +
	geom_line(aes(y = n_proto, color = "Nombre de prototypes")) +
	geom_line(aes(y = acc_test*scale, color = "Précision test")) +
	geom_line(aes(y = acc_train*scale, color = "Précision entrainement")) +
	scale_y_continuous(name = "Nombre de prototypes",
					 sec.axis = sec_axis(~./scale, name = "Précision",
										 labels = scales::percent)) +
	scale_color_manual(name="Légende", values = colors) +
	theme_light() +
	theme(legend.justification=c(0,0), legend.position=c(0.01,0.01),
		legend.box.background = element_rect(colour = "black"))
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Drawing 12 most representative digits
eps = 7.9
model <- protoclass(y = y_train, dxz = dist_eucl_train,
					eps=eps, lambda = 1/dim(X_train)[1])
write_csv(X_train[model$proto.order,], "representative_digits.csv")
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Retrain the model with all the column
data_test <- read_csv("../../Data/data_test_completed.csv")
X_test <- data_test %>% dplyr::select(dplyr::matches("_"))
data_test[["class"]] <- as.factor(make.names(data_test[["class"]]))
y_test <- data_test[["class"]]
load("../../Data/data_train.rda")
X_train <- data_train %>% dplyr::select(dplyr::matches("_"))
data_train[["class"]] <- as.factor(make.names(data_train[["class"]]))
y_train <- data_train[["class"]]
X_train <- scale(X_train)
X_test <- scale(X_test)
acc_allcols <- tibble(
	eps = numeric(),
	n_proto = numeric(),
	acc_train = numeric(),
	acc_test = numeric())
dist_eucl_train <- proxy:::dist(X_train, X_train, method = "Euclidean")
dist_eucl_test <- proxy:::dist(X_test, X_train, method = "Euclidean")
for (eps in seq(15, 38, 0.1))
{
	model <- protoclass(y = y_train, dxz = dist_eucl_train,
						eps=eps, lambda = 1/dim(X_train)[1])
	pred_train <- predictwithd.protoclass(
	model,
	dist_eucl_train)
	pred_test <- predictwithd.protoclass(
	model,
	dist_eucl_test)
	matrix_train <- confusionMatrix(
	y_train,
	pred_train)
	matrix_test <- confusionMatrix(
	y_test,
	pred_test)
	acc_allcols <- acc_allcols %>%
	add_row(eps = eps, n_proto = sum(model$nproto),
			acc_train = matrix_train[["overall"]][1],
			acc_test = matrix_test[["overall"]][1])
}
write_csv(acc_allcols, "acc_allcols.csv")
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Graph of precision and epsilon with all columns
scale = 1500
colors <- c("Nombre de prototypes" = "black",
			"Précision entrainement" = "blue",
			"Précision test" = "red")
ggplot(acc_allcols, aes(x = eps)) +
	geom_line(aes(y = n_proto, color = "Nombre de prototypes")) +
	geom_line(aes(y = acc_test*scale, color = "Précision test")) +
	geom_line(aes(y = acc_train*scale, color = "Précision entrainement")) +
	scale_y_continuous(name = "Nombre de prototypes",
					 sec.axis = sec_axis(~./scale, name = "Précision",
										 labels = scales::percent)) +
	scale_color_manual(name="Légende", values = colors) +
	theme_light() +
	theme(legend.justification=c(0,0), legend.position=c(0.01,0.01),
		legend.box.background = element_rect(colour = "black"))
# ------------------------------------------------------------------------------
