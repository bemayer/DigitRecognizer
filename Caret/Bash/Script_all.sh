for model in adaboost AdaBoost.M1 amdai vglmAdjCat AdaBag \
treebag bagFDAGCV bagFDA logicBag bagEarth bagEarthGCV bag \
bartMachine bayesglm binda ada gamboost glmboost BstLm \
LogitBoost bstSm blackboost bstTree J48 C5.0 rpart rpart1SE \
rpart2 rpartScore chaid cforest ctree ctree2 vglmContRatio \
C5.0Cost rpartCost vglmCumulative deepboost dda dwdPoly \
dwdRadial randomGLM xgbDART xgbLinear xgbTree elm RFlda \
fda FRBCS.CHI FH.GBML SLAVE FRBCS.W gaussprLinear \
gaussprPoly gaussprRadial gamLoess bam gam gamSpline \
glm glmStepAIC gpls glmnet glmnet_h2o gbm_h2o protoclass \
hda hdda hdrda kknn knn svmLinearWeights2 svmLinear3 \
lvq lssvmLinear lssvmPoly lssvmRadial lda lda2 stepLDA \
dwdLinear svmLinearWeights loclda logreg LMT Mlda mda \
manb avNNet monmlp mlp mlpWeightDecay mlpWeightDecayML \
mlpML msaenet mlpSGD mlpKerasDropout mlpKerasDropoutCost \
mlpKerasDecay mlpKerasDecayCost earth gcvEarth naive_bayes \
nb nbDiscrete awnb pam mxnet mxnetAdam nnet pcaNNet null \
ORFlog ORFpls ORFridge ORFsvm ownn polr parRF partDSA \
kernelpls pls simpls widekernelpls plsRglm PRIM pda pda2 \
PenalizedLDA plr multinom ordinalNet qda stepQDA rbf rbfDDA \
rFerns ranger Rborist rf ordinalRF extraTrees rfRules rda \
rlda regLogistic RRF RRFglobal Linda rmda QdaCov rrlda \
RSimca rocc rotationForest rotationForestCp JRip PART xyf \
nbSearch sda CSimca C5.0Rules C5.0Tree OneR sdwd sparseLDA \
smda spls slda snn dnn gbm svmBoundrangeString svmRadialWeights \
svmExpoString svmLinear svmLinear2 svmPoly svmRadial \
svmRadialCost svmRadialSigma svmSpectrumString tan tanSearch \
awtan evtree nodeHarvest vbmpRadial wsrf
do
cat > ../Script_$model.R << EOF
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
                         method="${model}",
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
EOF
done