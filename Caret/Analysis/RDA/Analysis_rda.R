# ------------------------------------------------------------------------------
# Libraries
library(caret)
library(dplyr)
library(readr)
library(klaR)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Read, convert and subset the test data
data_test <- read_csv("../../Data/data_test_completed.csv")
data_test[["class"]] <- as.factor(make.names(data_test[["class"]]))
y_test <- data_test[["class"]]
load("../../Data/data_train.rda")
data_train[["class"]] <- as.factor(make.names(data_train[["class"]]))
y_train <- data_train[["class"]]
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Caret model
X_test <- data_test %>% dplyr::select(dplyr::matches("pix*"))
model_3 <- readRDS(paste0("../../Model/rda_2012201631.rds"))
pred_test <- predict(model_3,newdata=subset(data_test, select = -c(class)))
matrix <- confusionMatrix(y_test, pred_test)
pred_test %>% as_tibble() %>% mutate(id = as.character(row_number() - 1)) %>%
	filter(pred_test != y_test)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Jarque-Bera Normality test of variables for LDA
jbtest <- function(x) {
	n <- length(x)
	m1 <- sum(x)/n
	m2 <- sum((x-m1)^2)/n
	m3 <- sum((x-m1)^3)/n
	m4 <- sum((x-m1)^4)/n
	b1 <- (m3/m2^(3/2))^2
	b2 <- (m4/m2^2)
	n*b1/6+n*(b2-3)^2/24
}
jbtest_p <- function(x) {
	n <- length(x)
	m1 <- sum(x)/n
	m2 <- sum((x-m1)^2)/n
	m3 <- sum((x-m1)^3)/n
	m4 <- sum((x-m1)^4)/n
	b1 <- (m3/m2^(3/2))^2
	b2 <- (m4/m2^2)
	val <- n*b1/6+n*(b2-3)^2/24
	pchisq(val,df = 2)
}
X_train <- data_train %>% dplyr::select(dplyr::matches("_")) %>% scale()
jbt_val <- apply(X_train, 2, jbtest)
jbt_p <- apply(X_train, 2, jbtest_p)
length(jbt_p[jbt_p >= 0.999])/length(jbt_p)
ggplot(as.data.frame(jbt_p), aes(x = jbt_p)) +
	geom_histogram() +
	labs(x = "p-value du test de Jarque-Bera", y = "Nombre de caractéristiques") +
	theme_light()
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Fit the model for various gamma and lambda
X_test <- data_test %>% dplyr::select(dplyr::matches("pix*")) %>% scale()
X_train <- data_train %>% dplyr::select(dplyr::matches("pix*")) %>% scale()
fitControl <- trainControl(method="none", verboseIter=TRUE)
acc_retrain <- tibble(
	gamma = numeric(),
	lambda = numeric(),
	acc_train = numeric(),
	acc_test = numeric())
for (gamma in 1:20/20)
{
	for (lambda in 1:20/20)
	{
	rdaGrid = data.frame(gamma = gamma, lambda = lambda)
	model <- caret::train(y=y_train,
												x=X_train,
												method="rda",
												trControl=fitControl,
												tuneGrid = rdaGrid)

	pred_train <- predict(model, X_train)
	pred_test <- predict(model, X_test)
	matrix_train <- confusionMatrix(y_train, pred_train)
	matrix_test <- confusionMatrix(y_test, pred_test)
	acc_retrain <- acc_retrain %>%
		add_row(gamma = gamma,
						lambda = lambda,
						acc_train = matrix_train[["overall"]][1],
						acc_test = matrix_test[["overall"]][1])
	assign(paste0("model_", gamma*100, "_", lambda*100), model)
	remove(model)
	}
}
write_csv(acc_retrain, "accuracy.csv")
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Plot the precision vs parameters
acc_train <- cbind("Précision sur l'échantillon d'entraînement",
									 acc_retrain[,1:3])
acc_test <- cbind("Précision sur l'échantillon de test",
									acc_retrain[,c(1,2,4)])
names(acc_train) <- c("Nom", "gamma", "lambda", "Précision")
names(acc_test) <- c("Nom", "gamma", "lambda", "Précision")
acc <- rbind(acc_test, acc_train)
ggplot(acc, aes(x = gamma, y = lambda, color = Précision)) +
	scale_color_gradient(
		low = "dark red",
		high = "dark green",
		labels = scales::percent,
		name = "Précision") +
	geom_point(size = 4) +
	facet_grid(. ~ Nom) +
	theme_light()
# ------------------------------------------------------------------------------

