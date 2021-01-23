# ------------------------------------------------------------------------------
# Libraries
library(dplyr)
library(stringr)
library(ggplot2)
library(readr)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Read the model files
folder = "./Log/"
files <- dir(folder, pattern = "^M")
data_test <- read.csv(".Data/data_test_completed.csv")
data_test[["class"]] <- as.factor(make.names(data_test[["class"]]))
acc <- tibble(
	Modèle = character(),
	Id = character(),
	Nom = character(),
	Epoch = numeric(),
	Précision = numeric())
for (file in files) {
	tab <- read_csv(paste0(folder, file), col_names = FALSE)
	acc_train <- bind_cols(Modèle = str_split(file,"_|\\.")[[1]][1],
							Id = str_split(file,"_|\\.")[[1]][2],
							Nom = "Précision sur l'échantillon d'entraînement",
							setNames(tab[1], "Epoch"),
							setNames(tab[2], "Précision"))
	acc_test <- bind_cols(Modèle = str_split(file,"_|\\.")[[1]][1],
							Id = str_split(file,"_|\\.")[[1]][2],
							Nom = "Précision sur l'échantillon de test",
							setNames(tab[1], "Epoch"),
							setNames(tab[3], "Précision"))
	acc <- bind_rows(acc, acc_train, acc_test)
	remove(acc_train)
	remove(acc_test)
	remove(tab)
	remove(file)
}
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Plot the accuracy vs epoch
acc %>% filter(Modèle == c("M0","M1","M2")) %>%
	ggplot(aes(x = Epoch, y = Précision, color = Nom)) +
	scale_color_discrete(name = "Légende") +
	geom_line(color = "grey", alpha = 0.5, position=position_jitter(w=0.02, h=0),
						aes(group=interaction(Id, Nom))) +
	geom_smooth(method="gam", se = FALSE) +
	scale_y_continuous(limits = c(0.94,1)) +
	facet_grid(. ~ Modèle) +
	theme_light() +
	theme(legend.position=c(0.8, 0.1),
				# legend.justification=c(0.01,0.01),
				legend.box.background = element_rect(colour = "black"))
acc %>% filter(Modèle == c("M3")) %>%
	ggplot(aes(x = Epoch, y = Précision, color = Nom)) +
	scale_color_discrete(name = "Légende") +
	geom_line(color = "grey", alpha = 0.5, position=position_jitter(w=0.02, h=0),
						aes(group=interaction(Id, Nom))) +
	geom_smooth(method="gam", se = FALSE) +
	scale_y_continuous(limits = c(0.8,1)) +
	theme_light() +
	theme(legend.position=c(0.8, 0.1),
				legend.box.background = element_rect(colour = "black"))
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Read the pred files and find classification errors
folder = "./Log/"
files <- dir(folder, pattern = "^p")
file <- files[1]
data_test <- read_csv("./Data/data_test_completed.csv")
data_test[["class"]] <- as.factor(make.names(data_test[["class"]]))
data_test <- data_test %>% mutate(id = as.character(row_number() - 1))
error <- tibble(
	Modèle = character(),
	Id = character(),
	Error = character())
for (file in files) {
	tab <- read_csv(paste0(folder, file), col_names = FALSE)
	names(tab) <- paste0("X", 0:9)
	pred <- as.factor(make.names(colnames(tab)[apply(tab,1,which.max)]))
	err <- bind_cols(Modèle = str_split(file,"_|\\.")[[1]][2],
									Id = str_split(file,"_|\\.")[[1]][3],
									setNames(data_test[data_test$class != pred,
									"id"], "Error"))
	error <- bind_rows(error, err)
	remove(tab)
	remove(pred)
}
error <- error %>% group_by(Modèle, Error) %>% summarize(cnt = n())
error_global <- error %>% group_by(Error) %>% summarize(cnt = sum(cnt))
# ------------------------------------------------------------------------------
