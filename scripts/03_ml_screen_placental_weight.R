# Machine-learning screening for placental weight
# Run this script from the repository root.
# This minimally cleaned version preserves the original computational logic.

# Reset the workspace.
rm(list = ls())
# Preserve character columns when creating data frames.
options(stringsAsFactors = F)

# Load required packages.
library(dplyr)
library(tibble)
library(Matrix)
library(tidymodels)
library(mlr3)
library(mlr3measures)
library(glmnet)
library(randomForestSRC)
library(gbm)
library(xgboost)
library(e1071)
library(caret)
library(nnet)
library(nlme)
library(rsample)

data_file <- file.path("data", "GSE75010_expression_clinical.tsv.gz")
candidate_gene_file <- file.path("data", "angiogenesis_candidate_genes.tsv")
figure_dir <- file.path("results", "figures")
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

##################################
#### Data preparation ####
##################################

# Load the expression/clinical matrix and candidate-gene list.
input<-read.table(gzfile(data_file),header = T,sep = "\t",check.names = F,row.names = NULL)
# Extract the five clinical outcome rows.
inputclinical<-input[1:5,]
# Retain candidate angiogenesis genes.
inputgene<-read.table(candidate_gene_file,header = T,sep = "\t",check.names = F,row.names = NULL)
datagene<-inputgene[,1]
input2<-input[input$geneID%in%datagene,]

# Select placental weight as the regression outcome.
clinicalrow<-as.data.frame(inputclinical[3,])
# Use the shared internal outcome name required by downstream formulas.
rownames(clinicalrow)<-c("sybp")
clinicalrow[1,1]<-c("sybp")
input2<-rbind(clinicalrow,input2)

# Transpose samples to rows.
input2<-data.frame(t(input2))
# Retain complete cases for this outcome.
input2<-na.omit(input2)
# Promote the first row to column names.
colnames(input2)<-input2[1,]
input2<-input2[-1,]
# Convert all analysis variables to numeric.
input2<-as.data.frame(lapply(input2,as.numeric))
##   input2$sybp<-as.numeric(input2$sybp)



# Define model seeds and the random-forest node size.
seed<-1234
rf_nodesize<-5

# Create the original 70:30 training-test split.
splitdata<-initial_split(input2,prop = 0.7)
traindata<-splitdata %>% training()
testdata<-splitdata %>% testing()

# Store training and test sets for uniform evaluation.
listdata<-list(traindata=traindata,testdata=testdata)
# Initialize the model-performance table.
result<-data.frame()



##################################
#### 1-1.RRF ####
##################################

set.seed(seed)
# Fit the model.
fit<-rfsrc(sybp~.,traindata,
           ntree = 10000,nodesize = rf_nodesize,
           splitrule = 'mse',
           importance = T,
           proximity = T,
           forest = T,
           seed = seed)
# Generate predictions.
RS<-predict(fit,newdata = testdata)
RSdata<-as.numeric(RS$predicted)
rs<-lapply(listdata,function(x){cbind(x[,1],RS=predict(fit,newdata = x)$predicted)})
# Calculate RMSE and append results.
rmsedata<-rmse(testdata[,1], RSdata)
resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
  rownames_to_column('ID')
resultdata$Model<-'RRF'
result<-rbind(result,resultdata)


##################################
#### 1-2.RRF + Enet ####
##################################

set.seed(seed)
# Fit the model.
fit<-rfsrc(sybp~.,traindata,
           ntree = 10000,nodesize = rf_nodesize,
           splitrule = 'mse',
           importance = T,
           proximity = T,
           forest = T,
           seed = seed)
# Select features.
# Select variables by minimal depth in the random forest.
rid<-var.select(object = fit, conservative = "high")
# Retain the selected top features.
rid<-rid$topvars
# Subset the training and test matrices to the selected features.
traindata2<-traindata[,c('sybp',rid)]
testdata2<-testdata[,c('sybp',rid)]
x1<-as.matrix(traindata2[,rid])
x2<-as.matrix(traindata2[,'sybp'])
listdata2<-lapply(listdata, function(x){x[, c('sybp', rid)]})

# Fit the downstream model using the features selected above.
for (alpha in seq(0.1, 0.9, 0.1)){
  set.seed(seed)
  fit = cv.glmnet(x1, x2, family = "gaussian", alpha = alpha, nfolds = 10)
  RS<-predict(fit,type = 'link', newx = as.matrix(testdata2[,-1]), s = fit$lambda.min)
  RSdata<-as.numeric(RS)
  rs<-lapply(listdata2,function(x){cbind(x[,1],RS=predict(fit,type = 'link', newx = as.matrix(x[,-1]), s = fit$lambda.min))})
  rmsedata<-rmse(testdata2[,1], RSdata)
  resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
    rownames_to_column('ID')
  resultdata$Model<-paste0('RRF + ', 'Enet', '[α=', alpha, ']')
  result<-rbind(result,resultdata)
}

##################################
#### 1-3.RRF + Lasso ####
##################################

set.seed(seed)
# Fit the model.
fit<-rfsrc(sybp~.,traindata,
           ntree = 10000,nodesize = rf_nodesize,
           splitrule = 'mse',
           importance = T,
           proximity = T,
           forest = T,
           seed = seed)
# Select features.
rid <- var.select(object = fit, conservative = "high")
rid <- rid$topvars
traindata2<-traindata[,c('sybp',rid)]
testdata2<-testdata[,c('sybp',rid)]
x1<-as.matrix(traindata2[,rid])
x2<-as.matrix(traindata2[,'sybp'])
listdata2<-lapply(listdata, function(x){x[, c('sybp', rid)]})

# Fit the downstream model using the features selected above.
set.seed(seed)
fit = cv.glmnet(x1, x2, family = "gaussian", alpha = 1, nfolds = 10)
RS<-predict(fit,type = 'link', newx = as.matrix(testdata2[,-1]), s = fit$lambda.min)
RSdata<-as.numeric(RS)
rs<-lapply(listdata2,function(x){cbind(x[,1],RS=predict(fit,type = 'link', newx = as.matrix(x[,-1]), s = fit$lambda.min))})
rmsedata<-rmse(testdata2[,1], RSdata)
resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
  rownames_to_column('ID')
resultdata$Model<-paste0('RRF + ', 'Lasso')
result<-rbind(result,resultdata)

##################################
#### 1-4.RRF + Ridge ####
##################################

set.seed(seed)
# Fit the model.
fit<-rfsrc(sybp~.,traindata,
           ntree = 10000,nodesize = rf_nodesize,
           splitrule = 'mse',
           importance = T,
           proximity = T,
           forest = T,
           seed = seed)
# Select features.
rid <- var.select(object = fit, conservative = "high")
rid <- rid$topvars
traindata2<-traindata[,c('sybp',rid)]
testdata2<-testdata[,c('sybp',rid)]
x1<-as.matrix(traindata2[,rid])
x2<-as.matrix(traindata2[,'sybp'])
listdata2<-lapply(listdata, function(x){x[, c('sybp', rid)]})

# Fit the downstream model using the features selected above.
set.seed(seed)
fit = cv.glmnet(x1, x2, family = "gaussian", alpha = 0, nfolds = 10)
RS<-predict(fit,type = 'link', newx = as.matrix(testdata2[,-1]), s = fit$lambda.min)
RSdata<-as.numeric(RS)
rs<-lapply(listdata2,function(x){cbind(x[,1],RS=predict(fit,type = 'link', newx = as.matrix(x[,-1]), s = fit$lambda.min))})
rmsedata<-rmse(testdata2[,1], RSdata)
resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
  rownames_to_column('ID')
resultdata$Model<-paste0('RRF + ', 'Ridge')
result<-rbind(result,resultdata)

##################################
#### 1-5.RRF + GBM ####
##################################

set.seed(seed)
# Fit the model.
fit<-rfsrc(sybp~.,traindata,
           ntree = 10000,nodesize = rf_nodesize,
           splitrule = 'mse',
           importance = T,
           proximity = T,
           forest = T,
           seed = seed)
# Select features.
rid <- var.select(object = fit, conservative = "high")
rid <- rid$topvars
traindata2<-traindata[,c('sybp',rid)]
testdata2<-testdata[,c('sybp',rid)]
x1<-as.matrix(traindata2[,rid])
x2<-as.matrix(traindata2[,'sybp'])
listdata2<-lapply(listdata, function(x){x[, c('sybp', rid)]})

# Fit the downstream model using the features selected above.
set.seed(seed)
fit <- gbm(sybp~.,traindata2,distribution = 'gaussian',
           n.trees = 10000,
           interaction.depth = 3,
           n.minobsinnode = 10,
           shrinkage = 0.001,
           cv.folds = 10, n.cores = 6)
best <- which.min(fit$cv.error)
set.seed(seed)
fit <- gbm(sybp~.,traindata2,distribution = 'gaussian',
           n.trees = best,
           interaction.depth = 3,
           n.minobsinnode = 10,
           shrinkage = 0.001,
           cv.folds = 10, n.cores = 8)
RS<-predict(fit,newdata = testdata2,n.trees = best,type = 'link')
RSdata<-as.numeric(RS)
rs<-lapply(listdata2,function(x){cbind(x[,1],RS=as.numeric(predict(fit,newdata = x,n.trees = best,type = 'link')))})
rmsedata<-rmse(testdata2[,1], RSdata)
resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
  rownames_to_column('ID')
resultdata$Model<-paste0('RRF + ', 'GBM')
result<-rbind(result,resultdata)

##################################
#### 1-6.RRF + XGBoost ####
##################################

set.seed(seed)
# Fit the model.
fit<-rfsrc(sybp~.,traindata,
           ntree = 10000,nodesize = rf_nodesize,
           splitrule = 'mse',
           importance = T,
           proximity = T,
           forest = T,
           seed = seed)
# Select features.
rid <- var.select(object = fit, conservative = "medium")
rid <- rid$topvars
traindata2<-traindata[,c('sybp',rid)]
testdata2<-testdata[,c('sybp',rid)]
x1<-as.matrix(traindata2[,rid])
x2<-as.matrix(traindata2[,'sybp'])
listdata2<-lapply(listdata, function(x){x[, c('sybp', rid)]})

# Fit the downstream model using the features selected above.
set.seed(seed)
cvfit<-xgb.cv(data = as.matrix(traindata2[,-1]),label = as.matrix(traindata2[,1]),objective="reg:squarederror",
              nrounds = 1000,
              nfold = 10)
best <- which.min(cvfit$evaluation_log$train_rmse_mean)
fit<-xgboost(data = as.matrix(traindata2[,-1]),label = as.matrix(traindata2[,1]),objective="reg:squarederror",
             nrounds = best)
RS<-predict(fit,as.matrix(testdata2[,-1]))
RSdata<-as.numeric(RS)
rs<-lapply(listdata2,function(x){cbind(x[,1],RS=as.numeric(predict(fit,as.matrix(x[,-1,drop = FALSE]))))})
rmsedata<-rmse(testdata2[,1], RSdata)
resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
  rownames_to_column('ID')
resultdata$Model<-paste0('RRF + ', 'XGBoost')
result<-rbind(result,resultdata)

##################################
#### 1-7.RRF + SVR ####
##################################

set.seed(seed)
# Fit the model.
fit<-rfsrc(sybp~.,traindata,
           ntree = 10000,nodesize = rf_nodesize,
           splitrule = 'mse',
           importance = T,
           proximity = T,
           forest = T,
           seed = seed)
# Select features.
rid <- var.select(object = fit, conservative = "high")
rid <- rid$topvars
traindata2<-traindata[,c('sybp',rid)]
testdata2<-testdata[,c('sybp',rid)]
x1<-as.matrix(traindata2[,rid])
x2<-as.matrix(traindata2[,'sybp'])
listdata2<-lapply(listdata, function(x){x[, c('sybp', rid)]})

# Fit the downstream model using the features selected above.
set.seed(seed)
bestfit<-tune.svm(x = traindata2[,-1],y = traindata2[,1],type = "eps-regression",
                  kernel = "radial",
                  gamma = 5*10^(-2:2),
                  cost = c(0.001,0.01,0.1,1,1.5,10,50,100))
bestgamma<-bestfit$best.parameters$gamma
bestcost<-bestfit$best.parameters$cost
fit<-svm(sybp~.,traindata2,type = "eps-regression",
         kernel = "radial",
         scale = F,
         gamma = bestgamma,
         cost = bestcost)
RS<-predict(fit,testdata2)
RSdata<-as.numeric(RS)
rs<-lapply(listdata2,function(x){cbind(x[,1],RS=predict(fit,newdata = x))})
rmsedata<-rmse(testdata2[,1], RSdata)
resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
  rownames_to_column('ID')
resultdata$Model<-paste0('RRF + ', 'SVR')
result<-rbind(result,resultdata)

##################################
#### 1-8.RRF + KNN ####
##################################

set.seed(seed)
# Fit the model.
fit<-rfsrc(sybp~.,traindata,
           ntree = 10000,nodesize = rf_nodesize,
           splitrule = 'mse',
           importance = T,
           proximity = T,
           forest = T,
           seed = seed)
# Select features.
rid <- var.select(object = fit, conservative = "high")
rid <- rid$topvars
traindata2<-traindata[,c('sybp',rid)]
testdata2<-testdata[,c('sybp',rid)]
x1<-as.matrix(traindata2[,rid])
x2<-as.matrix(traindata2[,'sybp'])
listdata2<-lapply(listdata, function(x){x[, c('sybp', rid)]})

# Fit the downstream model using the features selected above.
set.seed(seed)
traincontrol<-trainControl(method = "cv",number = 10)
traink<-expand.grid(.k = seq(1,20,by = 1))
bestfit<-train(sybp~.,traindata2,
               method = "knn",
               trControl = traincontrol,
               tuneGrid = traink)
bestk<-bestfit$bestTune$k
fit<-knnreg(sybp~.,traindata2,k = bestk)
RS<-predict(fit,testdata2)
RSdata<-as.numeric(RS)
rs<-lapply(listdata2,function(x){cbind(x[,1],RS=predict(fit,newdata = x))})
rmsedata<-rmse(testdata2[,1], RSdata)
resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
  rownames_to_column('ID')
resultdata$Model<-paste0('RRF + ', 'KNN')
result<-rbind(result,resultdata)



##################################
#### 2-1.Enet ####
##################################

set.seed(seed)
# Fit, predict, and evaluate.
x1<-as.matrix(traindata[,-1])
x2<-as.matrix(traindata[,'sybp'])

for (alpha in seq(0.1,0.9,0.1)){
  set.seed(seed)
  fit = cv.glmnet(x1, x2, family = "gaussian", alpha = alpha, nfolds = 10)
  RS<-predict(fit,type = 'link', newx = as.matrix(testdata[,-1]), s = fit$lambda.min)
  RSdata<-as.numeric(RS)
  rs<-lapply(listdata,function(x){cbind(x[,1],RS=predict(fit,type = 'link', newx = as.matrix(x[,-1]), s = fit$lambda.min))})
  rmsedata<-rmse(testdata[,1], RSdata)
  resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
    rownames_to_column('ID')
  resultdata$Model<-paste0('Enet', '[α=', alpha, ']')
  result<-rbind(result,resultdata)
}

##################################
#### 2-2.Enet + RRF ####
##################################

set.seed(seed)
x1<-as.matrix(traindata[,-1])
x2<-as.matrix(traindata[,'sybp'])

for (alpha in seq(0.1,0.9,0.1)){
  set.seed(seed)
  fit = cv.glmnet(x1, x2, family = "gaussian", alpha = alpha, nfolds = 10)
  myCoefs <- coef(fit, s = "lambda.min")  # Extract non-zero coefficients.
  rid <- myCoefs@Dimnames[[1]][which(myCoefs != 0 )]
  rid<-rid[-1]
  traindata2<-traindata[,c('sybp',rid)]
  testdata2<-testdata[,c('sybp',rid)]
  listdata2<-lapply(listdata, function(x){x[, c('sybp', rid)]})
  fit<-rfsrc(sybp~.,traindata2,
             ntree = 10000,nodesize = rf_nodesize,
           splitrule = 'mse',
             importance = T,
             proximity = T,
             forest = T,
             seed = seed)
  RS<-predict(fit,newdata = testdata2)
  RSdata<-as.numeric(RS$predicted)
  rs<-lapply(listdata2,function(x){cbind(x[,1],RS=predict(fit,newdata = x)$predicted)})
  rmsedata<-rmse(testdata2[,1], RSdata)
  resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
    rownames_to_column('ID')
  resultdata$Model<-paste0('Enet', '[α=', alpha, ']',' + ', 'RRF')
  result<-rbind(result,resultdata)
}

##################################
#### 2-3.Enet + GBM ####
##################################

set.seed(seed)
x1<-as.matrix(traindata[,-1])
x2<-as.matrix(traindata[,'sybp'])

for (alpha in seq(0.1,0.9,0.1)){
  set.seed(seed)
  fit = cv.glmnet(x1, x2, family = "gaussian", alpha = alpha, nfolds = 10)
  myCoefs <- coef(fit, s = "lambda.min")  # Extract non-zero coefficients.
  rid <- myCoefs@Dimnames[[1]][which(myCoefs != 0 )]
  rid<-rid[-1]
  traindata2<-traindata[,c('sybp',rid)]
  testdata2<-testdata[,c('sybp',rid)]
  listdata2<-lapply(listdata, function(x){x[, c('sybp', rid)]})
  fit <- gbm(sybp~.,traindata2,distribution = 'gaussian',
             n.trees = 10000,
             interaction.depth = 3,
             n.minobsinnode = 10,
             shrinkage = 0.001,
             cv.folds = 10, n.cores = 6)
  best <- which.min(fit$cv.error)
  set.seed(seed)
  fit <- gbm(sybp~.,traindata2,distribution = 'gaussian',
             n.trees = best,
             interaction.depth = 3,
             n.minobsinnode = 10,
             shrinkage = 0.001,
             cv.folds = 10, n.cores = 8)
  RS<-predict(fit,newdata = testdata2,n.trees = best,type = 'link')
  RSdata<-as.numeric(RS)
  rs<-lapply(listdata2,function(x){cbind(x[,1],RS=as.numeric(predict(fit,newdata = x,n.trees = best,type = 'link')))})
  rmsedata<-rmse(testdata2[,1], RSdata)
  resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
    rownames_to_column('ID')
  resultdata$Model<-paste0('Enet', '[α=', alpha, ']',' + ', 'GBM')
  result<-rbind(result,resultdata)
}

##################################
#### 2-4.Enet + XGBoost ####
##################################

set.seed(seed)
x1<-as.matrix(traindata[,-1])
x2<-as.matrix(traindata[,'sybp'])

for (alpha in seq(0.1,0.9,0.1)){
  set.seed(seed)
  fit = cv.glmnet(x1, x2, family = "gaussian", alpha = alpha, nfolds = 10)
  myCoefs <- coef(fit, s = "lambda.min")  # Extract non-zero coefficients.
  rid <- myCoefs@Dimnames[[1]][which(myCoefs != 0 )]
  rid<-rid[-1]
  traindata2<-traindata[,c('sybp',rid)]
  testdata2<-testdata[,c('sybp',rid)]
  listdata2<-lapply(listdata, function(x){x[, c('sybp', rid)]})
  cvfit<-xgb.cv(data = as.matrix(traindata2[,-1]),label = as.matrix(traindata2[,1]),objective="reg:squarederror",
                nrounds = 1000,
                nfold = 10)
  best <- which.min(cvfit$evaluation_log$train_rmse_mean)
  fit<-xgboost(data = as.matrix(traindata2[,-1]),label = as.matrix(traindata2[,1]),objective="reg:squarederror",
               nrounds = best)
  RS<-predict(fit,as.matrix(testdata2[,-1]))
  RSdata<-as.numeric(RS)
  rs<-lapply(listdata2,function(x){cbind(x[,1],RS=as.numeric(predict(fit,as.matrix(x[,-1,drop = FALSE]))))})
  rmsedata<-rmse(testdata2[,1], RSdata)
  resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
    rownames_to_column('ID')
  resultdata$Model<-paste0('Enet', '[α=', alpha, ']',' + ', 'XGBoost')
  result<-rbind(result,resultdata)
}

##################################
#### 2-5.Enet + SVR ####
##################################

set.seed(seed)
x1<-as.matrix(traindata[,-1])
x2<-as.matrix(traindata[,'sybp'])

for (alpha in seq(0.1,0.9,0.1)){
  set.seed(seed)
  fit = cv.glmnet(x1, x2, family = "gaussian", alpha = alpha, nfolds = 10)
  myCoefs <- coef(fit, s = "lambda.min")  # Extract non-zero coefficients.
  rid <- myCoefs@Dimnames[[1]][which(myCoefs != 0 )]
  rid<-rid[-1]
  traindata2<-traindata[,c('sybp',rid)]
  testdata2<-testdata[,c('sybp',rid)]
  listdata2<-lapply(listdata, function(x){x[, c('sybp', rid)]})
  bestfit<-tune.svm(x = traindata2[,-1],y = traindata2[,1],type = "eps-regression",
                    kernel = "radial",
                    gamma = 5*10^(-2:2),
                    cost = c(0.001,0.01,0.1,1,1.5,10,50,100))
  bestgamma<-bestfit$best.parameters$gamma
  bestcost<-bestfit$best.parameters$cost
  fit<-svm(sybp~.,traindata2,type = "eps-regression",
           kernel = "radial",
           scale = F,
           gamma = bestgamma,
           cost = bestcost)
  RS<-predict(fit,testdata2)
  RSdata<-as.numeric(RS)
  rs<-lapply(listdata2,function(x){cbind(x[,1],RS=predict(fit,newdata = x))})
  rmsedata<-rmse(testdata2[,1], RSdata)
  resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
    rownames_to_column('ID')
  resultdata$Model<-paste0('Enet', '[α=', alpha, ']',' + ', 'SVR')
  result<-rbind(result,resultdata)
}

##################################
#### 2-6.Enet + KNN ####
##################################

set.seed(seed)
x1<-as.matrix(traindata[,-1])
x2<-as.matrix(traindata[,'sybp'])

for (alpha in seq(0.1,0.9,0.1)){
  set.seed(seed)
  fit = cv.glmnet(x1, x2, family = "gaussian", alpha = alpha, nfolds = 10)
  myCoefs <- coef(fit, s = "lambda.min")  # Extract non-zero coefficients.
  rid <- myCoefs@Dimnames[[1]][which(myCoefs != 0 )]
  rid<-rid[-1]
  traindata2<-traindata[,c('sybp',rid)]
  testdata2<-testdata[,c('sybp',rid)]
  listdata2<-lapply(listdata, function(x){x[, c('sybp', rid)]})
  traincontrol<-trainControl(method = "cv",number = 10)
  traink<-expand.grid(.k = seq(1,20,by = 1))
  bestfit<-train(sybp~.,traindata2,
                 method = "knn",
                 trControl = traincontrol,
                 tuneGrid = traink)
  bestk<-bestfit$bestTune$k
  fit<-knnreg(sybp~.,traindata2,k = bestk)
  RS<-predict(fit,testdata2)
  RSdata<-as.numeric(RS)
  rs<-lapply(listdata2,function(x){cbind(x[,1],RS=predict(fit,newdata = x))})
  rmsedata<-rmse(testdata2[,1], RSdata)
  resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
    rownames_to_column('ID')
  resultdata$Model<-paste0('Enet', '[α=', alpha, ']',' + ', 'KNN')
  result<-rbind(result,resultdata)
}



##################################
#### 3-1.Lasso ####
##################################

set.seed(seed)
# Fit the model.
x1<-as.matrix(traindata[,-1])
x2<-as.matrix(traindata[,'sybp'])
fit = cv.glmnet(x1, x2, family = "gaussian", alpha = 1, nfolds = 10)
# Generate predictions.
RS<-predict(fit,type = 'link', newx = as.matrix(testdata[,-1]), s = fit$lambda.min)
RSdata<-as.numeric(RS)
rs<-lapply(listdata,function(x){cbind(x[,1],RS=predict(fit,type = 'link', newx = as.matrix(x[,-1]), s = fit$lambda.min))})
# Calculate RMSE and append results.
rmsedata<-rmse(testdata[,1], RSdata)
resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
  rownames_to_column('ID')
resultdata$Model<-'Lasso'
result<-rbind(result,resultdata)

##################################
#### 3-2.Lasso + RRF ####
##################################

set.seed(seed)
# Fit the model.
x1<-as.matrix(traindata[,-1])
x2<-as.matrix(traindata[,'sybp'])
fit = cv.glmnet(x1, x2, family = "gaussian", alpha = 1, nfolds = 10)
# Select features.
# Retain variables with non-zero coefficients.
myCoefs <- coef(fit, s = "lambda.min")
rid <- myCoefs@Dimnames[[1]][which(myCoefs != 0 )]
rid<-rid[-1]
traindata2<-traindata[,c('sybp',rid)]
testdata2<-testdata[,c('sybp',rid)]
listdata2<-lapply(listdata, function(x){x[, c('sybp', rid)]})

# Fit the downstream model using the features selected above.
set.seed(seed)
fit<-rfsrc(sybp~.,traindata2,
           ntree = 10000,nodesize = rf_nodesize,
           splitrule = 'mse',
           importance = T,
           proximity = T,
           forest = T,
           seed = seed)
RS<-predict(fit,newdata = testdata2)
RSdata<-as.numeric(RS$predicted)
rs<-lapply(listdata2,function(x){cbind(x[,1],RS=predict(fit,newdata = x)$predicted)})
rmsedata<-rmse(testdata2[,1], RSdata)
resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
  rownames_to_column('ID')
resultdata$Model<-paste0('Lasso + ', 'RRF')
result<-rbind(result,resultdata)

##################################
#### 3-3.Lasso + GBM ####
##################################

set.seed(seed)
# Fit the model.
x1<-as.matrix(traindata[,-1])
x2<-as.matrix(traindata[,'sybp'])
fit = cv.glmnet(x1, x2, family = "gaussian", alpha = 1, nfolds = 10)
# Select features.
# Retain variables with non-zero coefficients.
myCoefs <- coef(fit, s = "lambda.min")
rid <- myCoefs@Dimnames[[1]][which(myCoefs != 0 )]
rid<-rid[-1]
traindata2<-traindata[,c('sybp',rid)]
testdata2<-testdata[,c('sybp',rid)]
listdata2<-lapply(listdata, function(x){x[, c('sybp', rid)]})

# Fit the downstream model using the features selected above.
set.seed(seed)
fit <- gbm(sybp~.,traindata2,distribution = 'gaussian',
           n.trees = 10000,
           interaction.depth = 3,
           n.minobsinnode = 10,
           shrinkage = 0.001,
           cv.folds = 10, n.cores = 6)
best <- which.min(fit$cv.error)
set.seed(seed)
fit <- gbm(sybp~.,traindata2,distribution = 'gaussian',
           n.trees = best,
           interaction.depth = 3,
           n.minobsinnode = 10,
           shrinkage = 0.001,
           cv.folds = 10, n.cores = 8)
RS<-predict(fit,newdata = testdata2,n.trees = best,type = 'link')
RSdata<-as.numeric(RS)
rs<-lapply(listdata2,function(x){cbind(x[,1],RS=as.numeric(predict(fit,newdata = x,n.trees = best,type = 'link')))})
rmsedata<-rmse(testdata2[,1], RSdata)
resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
  rownames_to_column('ID')
resultdata$Model<-paste0('Lasso + ', 'GBM')
result<-rbind(result,resultdata)

##################################
#### 3-4.Lasso + XGBoost ####
##################################

set.seed(seed)
# Fit the model.
x1<-as.matrix(traindata[,-1])
x2<-as.matrix(traindata[,'sybp'])
fit = cv.glmnet(x1, x2, family = "gaussian", alpha = 1, nfolds = 10)
# Select features.
# Retain variables with non-zero coefficients.
myCoefs <- coef(fit, s = "lambda.min")
rid <- myCoefs@Dimnames[[1]][which(myCoefs != 0 )]
rid<-rid[-1]
traindata2<-traindata[,c('sybp',rid)]
testdata2<-testdata[,c('sybp',rid)]
listdata2<-lapply(listdata, function(x){x[, c('sybp', rid)]})

# Fit the downstream model using the features selected above.
set.seed(seed)
cvfit<-xgb.cv(data = as.matrix(traindata2[,-1]),label = as.matrix(traindata2[,1]),objective="reg:squarederror",
              nrounds = 1000,
              nfold = 10)
best <- which.min(cvfit$evaluation_log$train_rmse_mean)
fit<-xgboost(data = as.matrix(traindata2[,-1]),label = as.matrix(traindata2[,1]),objective="reg:squarederror",
             nrounds = best)
RS<-predict(fit,as.matrix(testdata2[,-1]))
RSdata<-as.numeric(RS)
rs<-lapply(listdata2,function(x){cbind(x[,1],RS=as.numeric(predict(fit,as.matrix(x[,-1,drop = FALSE]))))})
rmsedata<-rmse(testdata2[,1], RSdata)
resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
  rownames_to_column('ID')
resultdata$Model<-paste0('Lasso + ', 'XGBoost')
result<-rbind(result,resultdata)

##################################
#### 3-5.Lasso + SVR ####
##################################

set.seed(seed)
# Fit the model.
x1<-as.matrix(traindata[,-1])
x2<-as.matrix(traindata[,'sybp'])
fit = cv.glmnet(x1, x2, family = "gaussian", alpha = 1, nfolds = 10)
# Select features.
# Retain variables with non-zero coefficients.
myCoefs <- coef(fit, s = "lambda.min")
rid <- myCoefs@Dimnames[[1]][which(myCoefs != 0 )]
rid<-rid[-1]
traindata2<-traindata[,c('sybp',rid)]
testdata2<-testdata[,c('sybp',rid)]
listdata2<-lapply(listdata, function(x){x[, c('sybp', rid)]})

# Fit the downstream model using the features selected above.
set.seed(seed)
bestfit<-tune.svm(x = traindata2[,-1],y = traindata2[,1],type = "eps-regression",
                  kernel = "radial",
                  gamma = 5*10^(-2:2),
                  cost = c(0.001,0.01,0.1,1,1.5,10,50,100))
bestgamma<-bestfit$best.parameters$gamma
bestcost<-bestfit$best.parameters$cost
fit<-svm(sybp~.,traindata2,type = "eps-regression",
         kernel = "radial",
         scale = F,
         gamma = bestgamma,
         cost = bestcost)
RS<-predict(fit,testdata2)
RSdata<-as.numeric(RS)
rs<-lapply(listdata2,function(x){cbind(x[,1],RS=predict(fit,newdata = x))})
rmsedata<-rmse(testdata2[,1], RSdata)
resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
  rownames_to_column('ID')
resultdata$Model<-paste0('Lasso + ', 'SVR')
result<-rbind(result,resultdata)

##################################
#### 3-6.Lasso + KNN ####
##################################

set.seed(seed)
# Fit the model.
x1<-as.matrix(traindata[,-1])
x2<-as.matrix(traindata[,'sybp'])
fit = cv.glmnet(x1, x2, family = "gaussian", alpha = 1, nfolds = 10)
# Select features.
# Retain variables with non-zero coefficients.
myCoefs <- coef(fit, s = "lambda.min")
rid <- myCoefs@Dimnames[[1]][which(myCoefs != 0 )]
rid<-rid[-1]
traindata2<-traindata[,c('sybp',rid)]
testdata2<-testdata[,c('sybp',rid)]
listdata2<-lapply(listdata, function(x){x[, c('sybp', rid)]})

# Fit the downstream model using the features selected above.
set.seed(seed)
traincontrol<-trainControl(method = "cv",number = 10)
traink<-expand.grid(.k = seq(1,20,by = 1))
bestfit<-train(sybp~.,traindata2,
               method = "knn",
               trControl = traincontrol,
               tuneGrid = traink)
bestk<-bestfit$bestTune$k
fit<-knnreg(sybp~.,traindata2,k = bestk)
RS<-predict(fit,testdata2)
RSdata<-as.numeric(RS)
rs<-lapply(listdata2,function(x){cbind(x[,1],RS=predict(fit,newdata = x))})
rmsedata<-rmse(testdata2[,1], RSdata)
resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
  rownames_to_column('ID')
resultdata$Model<-paste0('Lasso + ', 'KNN')
result<-rbind(result,resultdata)



##################################
#### 4-1.Ridge ####
##################################

set.seed(seed)
# Fit the model.
x1<-as.matrix(traindata[,-1])
x2<-as.matrix(traindata[,'sybp'])
fit = cv.glmnet(x1, x2, family = "gaussian", alpha = 0, nfolds = 10)
# Generate predictions.
RS<-predict(fit,type = 'link', newx = as.matrix(testdata[,-1]), s = fit$lambda.min)
RSdata<-as.numeric(RS)
rs<-lapply(listdata,function(x){cbind(x[,1],RS=predict(fit,type = 'link', newx = as.matrix(x[,-1]), s = fit$lambda.min))})
# Calculate RMSE and append results.
rmsedata<-rmse(testdata[,1], RSdata)
resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
  rownames_to_column('ID')
resultdata$Model<-paste0('Ridge')
result<-rbind(result,resultdata)

##################################
#### 4-2.Ridge + RRF ####
##################################

set.seed(seed)
# Fit the model.
x1<-as.matrix(traindata[,-1])
x2<-as.matrix(traindata[,'sybp'])
fit = cv.glmnet(x1, x2, family = "gaussian", alpha = 0, nfolds = 10)
# Select features.
# Retain variables with non-zero coefficients.
myCoefs <- coef(fit, s = "lambda.min")
rid <- myCoefs@Dimnames[[1]][which(myCoefs != 0 )]
rid<-rid[-1]
traindata2<-traindata[,c('sybp',rid)]
testdata2<-testdata[,c('sybp',rid)]
listdata2<-lapply(listdata, function(x){x[, c('sybp', rid)]})

# Fit the downstream model using the features selected above.
set.seed(seed)
fit<-rfsrc(sybp~.,traindata2,
           ntree = 10000,nodesize = rf_nodesize,
           splitrule = 'mse',
           importance = T,
           proximity = T,
           forest = T,
           seed = seed)
RS<-predict(fit,newdata = testdata2)
RSdata<-as.numeric(RS$predicted)
rs<-lapply(listdata2,function(x){cbind(x[,1],RS=predict(fit,newdata = x)$predicted)})
rmsedata<-rmse(testdata2[,1], RSdata)
resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
  rownames_to_column('ID')
resultdata$Model<-paste0('Ridge + ', 'RRF')
result<-rbind(result,resultdata)

##################################
#### 4-3.Ridge + GBM ####
##################################

set.seed(seed)
# Fit the model.
x1<-as.matrix(traindata[,-1])
x2<-as.matrix(traindata[,'sybp'])
fit = cv.glmnet(x1, x2, family = "gaussian", alpha = 0, nfolds = 10)
# Select features.
# Retain variables with non-zero coefficients.
myCoefs <- coef(fit, s = "lambda.min")
rid <- myCoefs@Dimnames[[1]][which(myCoefs != 0 )]
rid<-rid[-1]
traindata2<-traindata[,c('sybp',rid)]
testdata2<-testdata[,c('sybp',rid)]
listdata2<-lapply(listdata, function(x){x[, c('sybp', rid)]})

# Fit the downstream model using the features selected above.
set.seed(seed)
fit <- gbm(sybp~.,traindata2,distribution = 'gaussian',
           n.trees = 10000,
           interaction.depth = 3,
           n.minobsinnode = 10,
           shrinkage = 0.001,
           cv.folds = 10, n.cores = 6)
best <- which.min(fit$cv.error)
set.seed(seed)
fit <- gbm(sybp~.,traindata2,distribution = 'gaussian',
           n.trees = best,
           interaction.depth = 3,
           n.minobsinnode = 10,
           shrinkage = 0.001,
           cv.folds = 10, n.cores = 8)
RS<-predict(fit,newdata = testdata2,n.trees = best,type = 'link')
RSdata<-as.numeric(RS)
rs<-lapply(listdata2,function(x){cbind(x[,1],RS=as.numeric(predict(fit,newdata = x,n.trees = best,type = 'link')))})
rmsedata<-rmse(testdata2[,1], RSdata)
resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
  rownames_to_column('ID')
resultdata$Model<-paste0('Ridge + ', 'GBM')
result<-rbind(result,resultdata)

##################################
#### 4-4.Ridge + XGBoost ####
##################################

set.seed(seed)
# Fit the model.
x1<-as.matrix(traindata[,-1])
x2<-as.matrix(traindata[,'sybp'])
fit = cv.glmnet(x1, x2, family = "gaussian", alpha = 0, nfolds = 10)
# Select features.
# Retain variables with non-zero coefficients.
myCoefs <- coef(fit, s = "lambda.min")
rid <- myCoefs@Dimnames[[1]][which(myCoefs != 0 )]
rid<-rid[-1]
traindata2<-traindata[,c('sybp',rid)]
testdata2<-testdata[,c('sybp',rid)]
listdata2<-lapply(listdata, function(x){x[, c('sybp', rid)]})

# Fit the downstream model using the features selected above.
set.seed(seed)
cvfit<-xgb.cv(data = as.matrix(traindata2[,-1]),label = as.matrix(traindata2[,1]),objective="reg:squarederror",
              nrounds = 1000,
              nfold = 10)
best <- which.min(cvfit$evaluation_log$train_rmse_mean)
fit<-xgboost(data = as.matrix(traindata2[,-1]),label = as.matrix(traindata2[,1]),objective="reg:squarederror",
             nrounds = best)
RS<-predict(fit,as.matrix(testdata2[,-1]))
RSdata<-as.numeric(RS)
rs<-lapply(listdata2,function(x){cbind(x[,1],RS=as.numeric(predict(fit,as.matrix(x[,-1,drop = FALSE]))))})
rmsedata<-rmse(testdata2[,1], RSdata)
resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
  rownames_to_column('ID')
resultdata$Model<-paste0('Ridge + ', 'XGBoost')
result<-rbind(result,resultdata)

##################################
#### 4-5.Ridge + SVR ####
##################################

set.seed(seed)
# Fit the model.
x1<-as.matrix(traindata[,-1])
x2<-as.matrix(traindata[,'sybp'])
fit = cv.glmnet(x1, x2, family = "gaussian", alpha = 0, nfolds = 10)
# Select features.
# Retain variables with non-zero coefficients.
myCoefs <- coef(fit, s = "lambda.min")
rid <- myCoefs@Dimnames[[1]][which(myCoefs != 0 )]
rid<-rid[-1]
traindata2<-traindata[,c('sybp',rid)]
testdata2<-testdata[,c('sybp',rid)]
listdata2<-lapply(listdata, function(x){x[, c('sybp', rid)]})

# Fit the downstream model using the features selected above.
set.seed(seed)
bestfit<-tune.svm(x = traindata2[,-1],y = traindata2[,1],type = "eps-regression",
                  kernel = "radial",
                  gamma = 5*10^(-2:2),
                  cost = c(0.001,0.01,0.1,1,1.5,10,50,100))
bestgamma<-bestfit$best.parameters$gamma
bestcost<-bestfit$best.parameters$cost
fit<-svm(sybp~.,traindata2,type = "eps-regression",
         kernel = "radial",
         scale = F,
         gamma = bestgamma,
         cost = bestcost)
RS<-predict(fit,testdata2)
RSdata<-as.numeric(RS)
rs<-lapply(listdata2,function(x){cbind(x[,1],RS=predict(fit,newdata = x))})
rmsedata<-rmse(testdata2[,1], RSdata)
resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
  rownames_to_column('ID')
resultdata$Model<-paste0('Ridge + ', 'SVR')
result<-rbind(result,resultdata)

##################################
#### 4-6.Ridge + KNN ####
##################################

set.seed(seed)
# Fit the model.
x1<-as.matrix(traindata[,-1])
x2<-as.matrix(traindata[,'sybp'])
fit = cv.glmnet(x1, x2, family = "gaussian", alpha = 0, nfolds = 10)
# Select features.
# Retain variables with non-zero coefficients.
myCoefs <- coef(fit, s = "lambda.min")
rid <- myCoefs@Dimnames[[1]][which(myCoefs != 0 )]
rid<-rid[-1]
traindata2<-traindata[,c('sybp',rid)]
testdata2<-testdata[,c('sybp',rid)]
listdata2<-lapply(listdata, function(x){x[, c('sybp', rid)]})

# Fit the downstream model using the features selected above.
set.seed(seed)
traincontrol<-trainControl(method = "cv",number = 10)
traink<-expand.grid(.k = seq(1,20,by = 1))
bestfit<-train(sybp~.,traindata2,
               method = "knn",
               trControl = traincontrol,
               tuneGrid = traink)
bestk<-bestfit$bestTune$k
fit<-knnreg(sybp~.,traindata2,k = bestk)
RS<-predict(fit,testdata2)
RSdata<-as.numeric(RS)
rs<-lapply(listdata2,function(x){cbind(x[,1],RS=predict(fit,newdata = x))})
rmsedata<-rmse(testdata2[,1], RSdata)
resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
  rownames_to_column('ID')
resultdata$Model<-paste0('Ridge + ', 'KNN')
result<-rbind(result,resultdata)



##################################
#### 5-1.GBM ####
##################################

set.seed(seed)
# Fit the model.
fit <- gbm(sybp~.,traindata,distribution = 'gaussian',
           n.trees = 10000,
           interaction.depth = 3,
           n.minobsinnode = 10,
           shrinkage = 0.001,
           cv.folds = 10, n.cores = 6)
# Select the tree count minimizing cross-validation error.
best <- which.min(fit$cv.error)
set.seed(seed)
fit <- gbm(sybp~.,traindata,distribution = 'gaussian',
           n.trees = best,
           interaction.depth = 3,
           n.minobsinnode = 10,
           shrinkage = 0.001,
           cv.folds = 10, n.cores = 8)
# Generate predictions.
RS<-predict(fit,newdata = testdata,n.trees = best,type = 'link')
RSdata<-as.numeric(RS)
rs<-lapply(listdata,function(x){cbind(x[,1],RS=as.numeric(predict(fit,newdata = x,n.trees = best,type = 'link')))})
# Calculate RMSE and append results.
rmsedata<-rmse(testdata[,1], RSdata)
resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
  rownames_to_column('ID')
resultdata$Model<-'GBM'
result<-rbind(result,resultdata)

##################################
#### 5-2.GBM + RRF ####
##################################

set.seed(seed)
# Fit the model.
fit <- gbm(sybp~.,traindata,distribution = 'gaussian',
           n.trees = 10000,
           interaction.depth = 3,
           n.minobsinnode = 10,
           shrinkage = 0.001,
           cv.folds = 10, n.cores = 6)
# Select the tree count minimizing cross-validation error.
best <- which.min(fit$cv.error)
set.seed(seed)
fit <- gbm(sybp~.,traindata,distribution = 'gaussian',
           n.trees = best,
           interaction.depth = 3,
           n.minobsinnode = 10,
           shrinkage = 0.001,
           cv.folds = 10, n.cores = 8)
# Select features.
# Rank variables by relative influence at the selected tree count.
rid<-summary(fit,n.trees = best)
# Retain the top 10 features.
rid<-rid[1:10,1]
traindata2<-traindata[,c('sybp',rid)]
testdata2<-testdata[,c('sybp',rid)]
listdata2<-lapply(listdata, function(x){x[, c('sybp', rid)]})

# Fit the downstream model using the features selected above.
set.seed(seed)
fit<-rfsrc(sybp~.,traindata2,
           ntree = 10000,nodesize = rf_nodesize,
           splitrule = 'mse',
           importance = T,
           proximity = T,
           forest = T,
           seed = seed)
RS<-predict(fit,newdata = testdata2)
RSdata<-as.numeric(RS$predicted)
rs<-lapply(listdata2,function(x){cbind(x[,1],RS=predict(fit,newdata = x)$predicted)})
rmsedata<-rmse(testdata2[,1], RSdata)
resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
  rownames_to_column('ID')
resultdata$Model<-paste0('GBM + ', 'RRF')
result<-rbind(result,resultdata)

##################################
#### 5-3.GBM + Lasso ####
##################################

set.seed(seed)
# Fit the model.
fit <- gbm(sybp~.,traindata,distribution = 'gaussian',
           n.trees = 10000,
           interaction.depth = 3,
           n.minobsinnode = 10,
           shrinkage = 0.001,
           cv.folds = 10, n.cores = 6)
# Select the tree count minimizing cross-validation error.
best <- which.min(fit$cv.error)
set.seed(seed)
fit <- gbm(sybp~.,traindata,distribution = 'gaussian',
           n.trees = best,
           interaction.depth = 3,
           n.minobsinnode = 10,
           shrinkage = 0.001,
           cv.folds = 10, n.cores = 8)
# Select features.
# Rank variables by relative influence at the selected tree count.
rid<-summary(fit,n.trees = best)
# Retain the top 10 features.
rid<-rid[1:10,1]
traindata2<-traindata[,c('sybp',rid)]
testdata2<-testdata[,c('sybp',rid)]
x1<-as.matrix(traindata2[,rid])
x2<-as.matrix(traindata2[,'sybp'])
listdata2<-lapply(listdata, function(x){x[, c('sybp', rid)]})

# Fit the downstream model using the features selected above.
set.seed(seed)
fit = cv.glmnet(x1, x2, family = "gaussian", alpha = 1, nfolds = 10)
RS<-predict(fit,type = 'link', newx = as.matrix(testdata2[,-1]), s = fit$lambda.min)
RSdata<-as.numeric(RS)
rs<-lapply(listdata2,function(x){cbind(x[,1],RS=predict(fit,type = 'link', newx = as.matrix(x[,-1]), s = fit$lambda.min))})
rmsedata<-rmse(testdata2[,1], RSdata)
resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
  rownames_to_column('ID')
resultdata$Model<-paste0('GBM + ', 'Lasso')
result<-rbind(result,resultdata)

##################################
#### 5-4.GBM + Ridge ####
##################################

set.seed(seed)
# Fit the model.
fit <- gbm(sybp~.,traindata,distribution = 'gaussian',
           n.trees = 10000,
           interaction.depth = 3,
           n.minobsinnode = 10,
           shrinkage = 0.001,
           cv.folds = 10, n.cores = 6)
# Select the tree count minimizing cross-validation error.
best <- which.min(fit$cv.error)
set.seed(seed)
fit <- gbm(sybp~.,traindata,distribution = 'gaussian',
           n.trees = best,
           interaction.depth = 3,
           n.minobsinnode = 10,
           shrinkage = 0.001,
           cv.folds = 10, n.cores = 8)
# Select features.
# Rank variables by relative influence at the selected tree count.
rid<-summary(fit,n.trees = best)
# Retain the top 10 features.
rid<-rid[1:10,1]
traindata2<-traindata[,c('sybp',rid)]
testdata2<-testdata[,c('sybp',rid)]
x1<-as.matrix(traindata2[,rid])
x2<-as.matrix(traindata2[,'sybp'])
listdata2<-lapply(listdata, function(x){x[, c('sybp', rid)]})

# Fit the downstream model using the features selected above.
set.seed(seed)
fit = cv.glmnet(x1, x2, family = "gaussian", alpha = 0, nfolds = 10)
RS<-predict(fit,type = 'link', newx = as.matrix(testdata2[,-1]), s = fit$lambda.min)
RSdata<-as.numeric(RS)
rs<-lapply(listdata2,function(x){cbind(x[,1],RS=predict(fit,type = 'link', newx = as.matrix(x[,-1]), s = fit$lambda.min))})
rmsedata<-rmse(testdata2[,1], RSdata)
resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
  rownames_to_column('ID')
resultdata$Model<-paste0('GBM + ', 'Ridge')
result<-rbind(result,resultdata)

##################################
#### 5-5.GBM + Enet ####
##################################

set.seed(seed)
# Fit the model.
fit <- gbm(sybp~.,traindata,distribution = 'gaussian',
           n.trees = 10000,
           interaction.depth = 3,
           n.minobsinnode = 10,
           shrinkage = 0.001,
           cv.folds = 10, n.cores = 6)
# Select the tree count minimizing cross-validation error.
best <- which.min(fit$cv.error)
set.seed(seed)
fit <- gbm(sybp~.,traindata,distribution = 'gaussian',
           n.trees = best,
           interaction.depth = 3,
           n.minobsinnode = 10,
           shrinkage = 0.001,
           cv.folds = 10, n.cores = 8)
# Select features.
# Rank variables by relative influence at the selected tree count.
rid<-summary(fit,n.trees = best)
# Retain the top 10 features.
rid<-rid[1:10,1]
traindata2<-traindata[,c('sybp',rid)]
testdata2<-testdata[,c('sybp',rid)]
x1<-as.matrix(traindata2[,rid])
x2<-as.matrix(traindata2[,'sybp'])
listdata2<-lapply(listdata, function(x){x[, c('sybp', rid)]})

# Fit the downstream model using the features selected above.
for (alpha in seq(0.1, 0.9, 0.1)){
  set.seed(seed)
  fit = cv.glmnet(x1, x2, family = "gaussian", alpha = alpha, nfolds = 10)
  RS<-predict(fit,type = 'link', newx = as.matrix(testdata2[,-1]), s = fit$lambda.min)
  RSdata<-as.numeric(RS)
  rs<-lapply(listdata2,function(x){cbind(x[,1],RS=predict(fit,type = 'link', newx = as.matrix(x[,-1]), s = fit$lambda.min))})
  rmsedata<-rmse(testdata2[,1], RSdata)
  resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
    rownames_to_column('ID')
  resultdata$Model<-paste0('GBM + ', 'Enet', '[α=', alpha, ']')
  result<-rbind(result,resultdata)
}

##################################
#### 5-6.GBM + XGBoost ####
##################################

set.seed(seed)
# Fit the model.
fit <- gbm(sybp~.,traindata,distribution = 'gaussian',
           n.trees = 10000,
           interaction.depth = 3,
           n.minobsinnode = 10,
           shrinkage = 0.001,
           cv.folds = 10, n.cores = 6)
# Select the tree count minimizing cross-validation error.
best <- which.min(fit$cv.error)
set.seed(seed)
fit <- gbm(sybp~.,traindata,distribution = 'gaussian',
           n.trees = best,
           interaction.depth = 3,
           n.minobsinnode = 10,
           shrinkage = 0.001,
           cv.folds = 10, n.cores = 8)
# Select features.
# Rank variables by relative influence at the selected tree count.
rid<-summary(fit,n.trees = best)
# Retain the top 10 features.
rid<-rid[1:10,1]
traindata2<-traindata[,c('sybp',rid)]
testdata2<-testdata[,c('sybp',rid)]
listdata2<-lapply(listdata, function(x){x[, c('sybp', rid)]})

# Fit the downstream model using the features selected above.
set.seed(seed)
cvfit<-xgb.cv(data = as.matrix(traindata2[,-1]),label = as.matrix(traindata2[,1]),objective="reg:squarederror",
              nrounds = 1000,
              nfold = 10)
best <- which.min(cvfit$evaluation_log$train_rmse_mean)
fit<-xgboost(data = as.matrix(traindata2[,-1]),label = as.matrix(traindata2[,1]),objective="reg:squarederror",
             nrounds = best)
RS<-predict(fit,as.matrix(testdata2[,-1]))
RSdata<-as.numeric(RS)
rs<-lapply(listdata2,function(x){cbind(x[,1],RS=as.numeric(predict(fit,as.matrix(x[,-1,drop = FALSE]))))})
rmsedata<-rmse(testdata2[,1], RSdata)
resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
  rownames_to_column('ID')
resultdata$Model<-paste0('GBM + ', 'XGBoost')
result<-rbind(result,resultdata)

##################################
#### 5-7.GBM + SVR ####
##################################

set.seed(seed)
# Fit the model.
fit <- gbm(sybp~.,traindata,distribution = 'gaussian',
           n.trees = 10000,
           interaction.depth = 3,
           n.minobsinnode = 10,
           shrinkage = 0.001,
           cv.folds = 10, n.cores = 6)
# Select the tree count minimizing cross-validation error.
best <- which.min(fit$cv.error)
set.seed(seed)
fit <- gbm(sybp~.,traindata,distribution = 'gaussian',
           n.trees = best,
           interaction.depth = 3,
           n.minobsinnode = 10,
           shrinkage = 0.001,
           cv.folds = 10, n.cores = 8)
# Select features.
# Rank variables by relative influence at the selected tree count.
rid<-summary(fit,n.trees = best)
# Retain the top 10 features.
rid<-rid[1:10,1]
traindata2<-traindata[,c('sybp',rid)]
testdata2<-testdata[,c('sybp',rid)]
x1<-as.matrix(traindata2[,rid])
x2<-as.matrix(traindata2[,'sybp'])
listdata2<-lapply(listdata, function(x){x[, c('sybp', rid)]})

# Fit the downstream model using the features selected above.
set.seed(seed)
bestfit<-tune.svm(x = traindata2[,-1],y = traindata2[,1],type = "eps-regression",
                  kernel = "radial",
                  gamma = 5*10^(-2:2),
                  cost = c(0.001,0.01,0.1,1,1.5,10,50,100))
bestgamma<-bestfit$best.parameters$gamma
bestcost<-bestfit$best.parameters$cost
fit<-svm(sybp~.,traindata2,type = "eps-regression",
         kernel = "radial",
         scale = F,
         gamma = bestgamma,
         cost = bestcost)
RS<-predict(fit,testdata2)
RSdata<-as.numeric(RS)
rs<-lapply(listdata2,function(x){cbind(x[,1],RS=predict(fit,newdata = x))})
rmsedata<-rmse(testdata2[,1], RSdata)
resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
  rownames_to_column('ID')
resultdata$Model<-paste0('GBM + ', 'SVR')
result<-rbind(result,resultdata)

##################################
#### 5-8.GBM + KNN ####
##################################

set.seed(seed)
# Fit the model.
fit <- gbm(sybp~.,traindata,distribution = 'gaussian',
           n.trees = 10000,
           interaction.depth = 3,
           n.minobsinnode = 10,
           shrinkage = 0.001,
           cv.folds = 10, n.cores = 6)
# Select the tree count minimizing cross-validation error.
best <- which.min(fit$cv.error)
set.seed(seed)
fit <- gbm(sybp~.,traindata,distribution = 'gaussian',
           n.trees = best,
           interaction.depth = 3,
           n.minobsinnode = 10,
           shrinkage = 0.001,
           cv.folds = 10, n.cores = 8)
# Select features.
# Rank variables by relative influence at the selected tree count.
rid<-summary(fit,n.trees = best)
# Retain the top 10 features.
rid<-rid[1:10,1]
traindata2<-traindata[,c('sybp',rid)]
testdata2<-testdata[,c('sybp',rid)]
x1<-as.matrix(traindata2[,rid])
x2<-as.matrix(traindata2[,'sybp'])
listdata2<-lapply(listdata, function(x){x[, c('sybp', rid)]})

# Fit the downstream model using the features selected above.
set.seed(seed)
traincontrol<-trainControl(method = "cv",number = 10)
traink<-expand.grid(.k = seq(1,20,by = 1))
bestfit<-train(sybp~.,traindata2,
               method = "knn",
               trControl = traincontrol,
               tuneGrid = traink)
bestk<-bestfit$bestTune$k
fit<-knnreg(sybp~.,traindata2,k = bestk)
RS<-predict(fit,testdata2)
RSdata<-as.numeric(RS)
rs<-lapply(listdata2,function(x){cbind(x[,1],RS=predict(fit,newdata = x))})
rmsedata<-rmse(testdata2[,1], RSdata)
resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
  rownames_to_column('ID')
resultdata$Model<-paste0('GBM + ', 'KNN')
result<-rbind(result,resultdata)



##################################
#### 6-1.XGBoost ####
##################################

set.seed(seed)
# Fit the model.
# Select the boosting rounds with the lowest cross-validation RMSE.
# XGBoost requires matrix inputs.
cvfit<-xgb.cv(data = as.matrix(traindata[,-1]),label = as.matrix(traindata[,1]),objective="reg:squarederror",
              nrounds = 1000,
              nfold = 10)
best <- which.min(cvfit$evaluation_log$train_rmse_mean)
# Fit XGBoost using the selected boosting rounds.
fit<-xgboost(data = as.matrix(traindata[,-1]),label = as.matrix(traindata[,1]),objective="reg:squarederror",
             nrounds = best)
# Generate predictions.
RS<-predict(fit,as.matrix(testdata[,-1]))
RSdata<-as.numeric(RS)
rs<-lapply(listdata,function(x){cbind(x[,1],RS=as.numeric(predict(fit,as.matrix(x[,-1,drop = FALSE]))))})
# Calculate RMSE and append results.
rmsedata<-rmse(testdata[,1], RSdata)
resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
  rownames_to_column('ID')
resultdata$Model<-'XGBoost'
result<-rbind(result,resultdata)

##################################
#### 6-2.XGBoost + RRF ####
##################################

set.seed(seed)
# Fit the model.
# Select the boosting rounds with the lowest cross-validation RMSE.
cvfit<-xgb.cv(data = as.matrix(traindata[,-1]),label = as.matrix(traindata[,1]),objective="reg:squarederror",
              nrounds = 1000,
              nfold = 10)
best <- which.min(cvfit$evaluation_log$train_rmse_mean)
# Fit XGBoost using the selected boosting rounds.
fit<-xgboost(data = as.matrix(traindata[,-1]),label = as.matrix(traindata[,1]),objective="reg:squarederror",
             nrounds = best)
# Select features.
# Rank variables using XGBoost importance.
namedata<-dimnames(data.matrix(traindata[,-1]))[[2]]
rid<-xgb.importance(namedata,fit)
# Retain the top 10 features.
rid<-rid[1:10,1]$Feature
traindata2<-traindata[,c('sybp',rid)]
testdata2<-testdata[,c('sybp',rid)]
listdata2<-lapply(listdata, function(x){x[, c('sybp', rid)]})

# Fit the downstream model using the features selected above.
set.seed(seed)
fit<-rfsrc(sybp~.,traindata2,
           ntree = 10000,nodesize = rf_nodesize,
           splitrule = 'mse',
           importance = T,
           proximity = T,
           forest = T,
           seed = seed)
RS<-predict(fit,newdata = testdata2)
RSdata<-as.numeric(RS$predicted)
rs<-lapply(listdata2,function(x){cbind(x[,1],RS=predict(fit,newdata = x)$predicted)})
rmsedata<-rmse(testdata2[,1], RSdata)
resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
  rownames_to_column('ID')
resultdata$Model<-paste0('XGBoost + ', 'RRF')
result<-rbind(result,resultdata)


##################################
#### 6-3.XGBoost + Lasso ####
##################################

set.seed(seed)
# Fit the model.
# Select the boosting rounds with the lowest cross-validation RMSE.
cvfit<-xgb.cv(data = as.matrix(traindata[,-1]),label = as.matrix(traindata[,1]),objective="reg:squarederror",
              nrounds = 1000,
              nfold = 10)
best <- which.min(cvfit$evaluation_log$train_rmse_mean)
# Fit XGBoost using the selected boosting rounds.
fit<-xgboost(data = as.matrix(traindata[,-1]),label = as.matrix(traindata[,1]),objective="reg:squarederror",
             nrounds = best)
# Select features.
# Rank variables using XGBoost importance.
namedata<-dimnames(data.matrix(traindata[,-1]))[[2]]
rid<-xgb.importance(namedata,fit)
# Retain the top 10 features.
rid<-rid[1:10,1]$Feature
traindata2<-traindata[,c('sybp',rid)]
testdata2<-testdata[,c('sybp',rid)]
x1<-as.matrix(traindata2[,rid])
x2<-as.matrix(traindata2[,'sybp'])
listdata2<-lapply(listdata, function(x){x[, c('sybp', rid)]})

# Fit the downstream model using the features selected above.
set.seed(seed)
fit = cv.glmnet(x1, x2, family = "gaussian", alpha = 1, nfolds = 10)
RS<-predict(fit,type = 'link', newx = as.matrix(testdata2[,-1]), s = fit$lambda.min)
RSdata<-as.numeric(RS)
rs<-lapply(listdata2,function(x){cbind(x[,1],RS=predict(fit,type = 'link', newx = as.matrix(x[,-1]), s = fit$lambda.min))})
rmsedata<-rmse(testdata2[,1], RSdata)
resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
  rownames_to_column('ID')
resultdata$Model<-paste0('XGBoost + ', 'Lasso')
result<-rbind(result,resultdata)

##################################
#### 6-4.XGBoost + Ridge ####
##################################

set.seed(seed)
# Fit the model.
# Select the boosting rounds with the lowest cross-validation RMSE.
cvfit<-xgb.cv(data = as.matrix(traindata[,-1]),label = as.matrix(traindata[,1]),objective="reg:squarederror",
              nrounds = 1000,
              nfold = 10)
best <- which.min(cvfit$evaluation_log$train_rmse_mean)
# Fit XGBoost using the selected boosting rounds.
fit<-xgboost(data = as.matrix(traindata[,-1]),label = as.matrix(traindata[,1]),objective="reg:squarederror",
             nrounds = best)
# Select features.
# Rank variables using XGBoost importance.
namedata<-dimnames(data.matrix(traindata[,-1]))[[2]]
rid<-xgb.importance(namedata,fit)
# Retain the top 10 features.
rid<-rid[1:10,1]$Feature
traindata2<-traindata[,c('sybp',rid)]
testdata2<-testdata[,c('sybp',rid)]
x1<-as.matrix(traindata2[,rid])
x2<-as.matrix(traindata2[,'sybp'])
listdata2<-lapply(listdata, function(x){x[, c('sybp', rid)]})

# Fit the downstream model using the features selected above.
set.seed(seed)
fit = cv.glmnet(x1, x2, family = "gaussian", alpha = 0, nfolds = 10)
RS<-predict(fit,type = 'link', newx = as.matrix(testdata2[,-1]), s = fit$lambda.min)
RSdata<-as.numeric(RS)
rs<-lapply(listdata2,function(x){cbind(x[,1],RS=predict(fit,type = 'link', newx = as.matrix(x[,-1]), s = fit$lambda.min))})
rmsedata<-rmse(testdata2[,1], RSdata)
resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
  rownames_to_column('ID')
resultdata$Model<-paste0('XGBoost + ', 'Ridge')
result<-rbind(result,resultdata)

##################################
#### 6-5.XGBoost + Enet ####
##################################

set.seed(seed)
# Fit the model.
# Select the boosting rounds with the lowest cross-validation RMSE.
cvfit<-xgb.cv(data = as.matrix(traindata[,-1]),label = as.matrix(traindata[,1]),objective="reg:squarederror",
              nrounds = 1000,
              nfold = 10)
best <- which.min(cvfit$evaluation_log$train_rmse_mean)
# Fit XGBoost using the selected boosting rounds.
fit<-xgboost(data = as.matrix(traindata[,-1]),label = as.matrix(traindata[,1]),objective="reg:squarederror",
             nrounds = best)
# Select features.
# Rank variables using XGBoost importance.
namedata<-dimnames(data.matrix(traindata[,-1]))[[2]]
rid<-xgb.importance(namedata,fit)
# Retain the top 10 features.
rid<-rid[1:10,1]$Feature
traindata2<-traindata[,c('sybp',rid)]
testdata2<-testdata[,c('sybp',rid)]
x1<-as.matrix(traindata2[,rid])
x2<-as.matrix(traindata2[,'sybp'])
listdata2<-lapply(listdata, function(x){x[, c('sybp', rid)]})

# Fit the downstream model using the features selected above.
for (alpha in seq(0.1, 0.9, 0.1)){
  set.seed(seed)
  fit = cv.glmnet(x1, x2, family = "gaussian", alpha = alpha, nfolds = 10)
  RS<-predict(fit,type = 'link', newx = as.matrix(testdata2[,-1]), s = fit$lambda.min)
  RSdata<-as.numeric(RS)
  rs<-lapply(listdata2,function(x){cbind(x[,1],RS=predict(fit,type = 'link', newx = as.matrix(x[,-1]), s = fit$lambda.min))})
  rmsedata<-rmse(testdata2[,1], RSdata)
  resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
    rownames_to_column('ID')
  resultdata$Model<-paste0('XGBoost + ', 'Enet', '[α=', alpha, ']')
  result<-rbind(result,resultdata)
}

##################################
#### 6-6.XGBoost + GBM ####
##################################

set.seed(seed)
# Fit the model.
# Select the boosting rounds with the lowest cross-validation RMSE.
cvfit<-xgb.cv(data = as.matrix(traindata[,-1]),label = as.matrix(traindata[,1]),objective="reg:squarederror",
              nrounds = 1000,
              nfold = 10)
best <- which.min(cvfit$evaluation_log$train_rmse_mean)
# Fit XGBoost using the selected boosting rounds.
fit<-xgboost(data = as.matrix(traindata[,-1]),label = as.matrix(traindata[,1]),objective="reg:squarederror",
             nrounds = best)
# Select features.
# Rank variables using XGBoost importance.
namedata<-dimnames(data.matrix(traindata[,-1]))[[2]]
rid<-xgb.importance(namedata,fit)
# Retain the top 10 features.
rid<-rid[1:10,1]$Feature
traindata2<-traindata[,c('sybp',rid)]
testdata2<-testdata[,c('sybp',rid)]
listdata2<-lapply(listdata, function(x){x[, c('sybp', rid)]})

# Fit the downstream model using the features selected above.
set.seed(seed)
fit <- gbm(sybp~.,traindata2,distribution = 'gaussian',
           n.trees = 10000,
           interaction.depth = 3,
           n.minobsinnode = 10,
           shrinkage = 0.001,
           cv.folds = 10, n.cores = 6)
best <- which.min(fit$cv.error)
set.seed(seed)
fit <- gbm(sybp~.,traindata2,distribution = 'gaussian',
           n.trees = best,
           interaction.depth = 3,
           n.minobsinnode = 10,
           shrinkage = 0.001,
           cv.folds = 10, n.cores = 8)
RS<-predict(fit,newdata = testdata2,n.trees = best,type = 'link')
RSdata<-as.numeric(RS)
rs<-lapply(listdata2,function(x){cbind(x[,1],RS=as.numeric(predict(fit,newdata = x,n.trees = best,type = 'link')))})
rmsedata<-rmse(testdata2[,1], RSdata)
resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
  rownames_to_column('ID')
resultdata$Model<-paste0('XGBoost + ', 'GBM')
result<-rbind(result,resultdata)

##################################
#### 6-7.XGBoost + SVR ####
##################################

set.seed(seed)
# Fit the model.
# Select the boosting rounds with the lowest cross-validation RMSE.
cvfit<-xgb.cv(data = as.matrix(traindata[,-1]),label = as.matrix(traindata[,1]),objective="reg:squarederror",
              nrounds = 1000,
              nfold = 10)
best <- which.min(cvfit$evaluation_log$train_rmse_mean)
# Fit XGBoost using the selected boosting rounds.
fit<-xgboost(data = as.matrix(traindata[,-1]),label = as.matrix(traindata[,1]),objective="reg:squarederror",
             nrounds = best)
# Select features.
# Rank variables using XGBoost importance.
namedata<-dimnames(data.matrix(traindata[,-1]))[[2]]
rid<-xgb.importance(namedata,fit)
# Retain the top 10 features.
rid<-rid[1:10,1]$Feature
traindata2<-traindata[,c('sybp',rid)]
testdata2<-testdata[,c('sybp',rid)]
listdata2<-lapply(listdata, function(x){x[, c('sybp', rid)]})

# Fit the downstream model using the features selected above.
set.seed(seed)
bestfit<-tune.svm(x = traindata2[,-1],y = traindata2[,1],type = "eps-regression",
                  kernel = "radial",
                  gamma = 5*10^(-2:2),
                  cost = c(0.001,0.01,0.1,1,1.5,10,50,100))
bestgamma<-bestfit$best.parameters$gamma
bestcost<-bestfit$best.parameters$cost
fit<-svm(sybp~.,traindata2,type = "eps-regression",
         kernel = "radial",
         scale = F,
         gamma = bestgamma,
         cost = bestcost)
RS<-predict(fit,testdata2)
RSdata<-as.numeric(RS)
rs<-lapply(listdata2,function(x){cbind(x[,1],RS=predict(fit,newdata = x))})
rmsedata<-rmse(testdata2[,1], RSdata)
resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
  rownames_to_column('ID')
resultdata$Model<-paste0('XGBoost + ', 'SVR')
result<-rbind(result,resultdata)

##################################
#### 6-8.XGBoost + KNN ####
##################################

set.seed(seed)
# Fit the model.
# Select the boosting rounds with the lowest cross-validation RMSE.
cvfit<-xgb.cv(data = as.matrix(traindata[,-1]),label = as.matrix(traindata[,1]),objective="reg:squarederror",
              nrounds = 1000,
              nfold = 10)
best <- which.min(cvfit$evaluation_log$train_rmse_mean)
# Fit XGBoost using the selected boosting rounds.
fit<-xgboost(data = as.matrix(traindata[,-1]),label = as.matrix(traindata[,1]),objective="reg:squarederror",
             nrounds = best)
# Select features.
# Rank variables using XGBoost importance.
namedata<-dimnames(data.matrix(traindata[,-1]))[[2]]
rid<-xgb.importance(namedata,fit)
# Retain the top 10 features.
rid<-rid[1:10,1]$Feature
traindata2<-traindata[,c('sybp',rid)]
testdata2<-testdata[,c('sybp',rid)]
listdata2<-lapply(listdata, function(x){x[, c('sybp', rid)]})

# Fit the downstream model using the features selected above.
set.seed(seed)
traincontrol<-trainControl(method = "cv",number = 10)
traink<-expand.grid(.k = seq(1,20,by = 1))
bestfit<-train(sybp~.,traindata2,
               method = "knn",
               trControl = traincontrol,
               tuneGrid = traink)
bestk<-bestfit$bestTune$k
fit<-knnreg(sybp~.,traindata2,k = bestk)
RS<-predict(fit,testdata2)
RSdata<-as.numeric(RS)
rs<-lapply(listdata2,function(x){cbind(x[,1],RS=predict(fit,newdata = x))})
rmsedata<-rmse(testdata2[,1], RSdata)
resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
  rownames_to_column('ID')
resultdata$Model<-paste0('XGBoost + ', 'KNN')
result<-rbind(result,resultdata)



##################################
#### 7-1.SVR ####
##################################

set.seed(seed)
# Fit the model.
# Tune radial SVR hyperparameters.
bestfit<-tune.svm(x = traindata[,-1],y = traindata[,1],type = "eps-regression",
                  kernel = "radial",
                  gamma = 5*10^(-2:2),
                  cost = c(0.001,0.01,0.1,1,1.5,10,50,100))
bestgamma<-bestfit$best.parameters$gamma
bestcost<-bestfit$best.parameters$cost
# Fit radial SVR using the selected hyperparameters.
fit<-svm(sybp~.,traindata,type = "eps-regression",
         kernel = "radial",
         scale = F,
         gamma = bestgamma,
         cost = bestcost)

# Generate predictions.
RS<-predict(fit,testdata)
RSdata<-as.numeric(RS)
rs<-lapply(listdata,function(x){cbind(x[,1],RS=predict(fit,newdata = x))})
# Calculate RMSE and append results.
rmsedata<-rmse(testdata[,1], RSdata)
resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
  rownames_to_column('ID')
resultdata$Model<-'SVR'
result<-rbind(result,resultdata)

##################################
#### 7-2.SVR + RRF ####
##################################

set.seed(seed)
# Fit the model.
# Tune radial SVR hyperparameters.
bestfit<-tune.svm(x = traindata[,-1],y = traindata[,1],type = "eps-regression",
                  kernel = "radial",
                  gamma = 5*10^(-2:2),
                  cost = c(0.001,0.01,0.1,1,1.5,10,50,100))
bestgamma<-bestfit$best.parameters$gamma
bestcost<-bestfit$best.parameters$cost
# Fit radial SVR using the selected hyperparameters.
fit<-svm(sybp~.,traindata,type = "eps-regression",
         kernel = "radial",
         scale = F,
         gamma = bestgamma,
         cost = bestcost)
# Select features.
# Define recursive feature elimination with cross-validation.
rfeCNTL<-rfeControl(functions = lmFuncs,method = "cv",number = 10)
# Run recursive feature elimination.
svmfeature<-rfe(sybp~.,traindata,
                sizes = c(10:24),  # Evaluate subsets containing 10-24 features.
                rfeControl = rfeCNTL,
                methods = "svmRadial")
rid<-svmfeature$optVariables
traindata2<-traindata[,c('sybp',rid)]
testdata2<-testdata[,c('sybp',rid)]
listdata2<-lapply(listdata, function(x){x[, c('sybp', rid)]})

# Fit the downstream model using the features selected above.
set.seed(seed)
fit<-rfsrc(sybp~.,traindata2,
           ntree = 10000,nodesize = rf_nodesize,
           splitrule = 'mse',
           importance = T,
           proximity = T,
           forest = T,
           seed = seed)
RS<-predict(fit,newdata = testdata2)
RSdata<-as.numeric(RS$predicted)
rs<-lapply(listdata2,function(x){cbind(x[,1],RS=predict(fit,newdata = x)$predicted)})
rmsedata<-rmse(testdata2[,1], RSdata)
resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
  rownames_to_column('ID')
resultdata$Model<-paste0('SVR + ', 'RRF')
result<-rbind(result,resultdata)


##################################
#### 7-3.SVR + Lasso ####
##################################

set.seed(seed)
# Fit the model.
# Tune radial SVR hyperparameters.
bestfit<-tune.svm(x = traindata[,-1],y = traindata[,1],type = "eps-regression",
                  kernel = "radial",
                  gamma = 5*10^(-2:2),
                  cost = c(0.001,0.01,0.1,1,1.5,10,50,100))
bestgamma<-bestfit$best.parameters$gamma
bestcost<-bestfit$best.parameters$cost
# Fit radial SVR using the selected hyperparameters.
fit<-svm(sybp~.,traindata,type = "eps-regression",
         kernel = "radial",
         scale = F,
         gamma = bestgamma,
         cost = bestcost)
# Select features.
# Define recursive feature elimination with cross-validation.
rfeCNTL<-rfeControl(functions = lmFuncs,method = "cv",number = 10)
# Run recursive feature elimination.
svmfeature<-rfe(sybp~.,traindata,
                sizes = c(10:24),  # Evaluate subsets containing 10-24 features.
                rfeControl = rfeCNTL,
                methods = "svmRadial")
rid<-svmfeature$optVariables
traindata2<-traindata[,c('sybp',rid)]
testdata2<-testdata[,c('sybp',rid)]
x1<-as.matrix(traindata2[,rid])
x2<-as.matrix(traindata2[,'sybp'])
listdata2<-lapply(listdata, function(x){x[, c('sybp', rid)]})

# Fit the downstream model using the features selected above.
set.seed(seed)
fit = cv.glmnet(x1, x2, family = "gaussian", alpha = 1, nfolds = 10)
RS<-predict(fit,type = 'link', newx = as.matrix(testdata2[,-1]), s = fit$lambda.min)
RSdata<-as.numeric(RS)
rs<-lapply(listdata2,function(x){cbind(x[,1],RS=predict(fit,type = 'link', newx = as.matrix(x[,-1]), s = fit$lambda.min))})
rmsedata<-rmse(testdata2[,1], RSdata)
resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
  rownames_to_column('ID')
resultdata$Model<-paste0('SVR + ', 'Lasso')
result<-rbind(result,resultdata)

##################################
#### 7-4.SVR + Ridge ####
##################################

set.seed(seed)
# Fit the model.
# Tune radial SVR hyperparameters.
bestfit<-tune.svm(x = traindata[,-1],y = traindata[,1],type = "eps-regression",
                  kernel = "radial",
                  gamma = 5*10^(-2:2),
                  cost = c(0.001,0.01,0.1,1,1.5,10,50,100))
bestgamma<-bestfit$best.parameters$gamma
bestcost<-bestfit$best.parameters$cost
# Fit radial SVR using the selected hyperparameters.
fit<-svm(sybp~.,traindata,type = "eps-regression",
         kernel = "radial",
         scale = F,
         gamma = bestgamma,
         cost = bestcost)
# Select features.
# Define recursive feature elimination with cross-validation.
rfeCNTL<-rfeControl(functions = lmFuncs,method = "cv",number = 10)
# Run recursive feature elimination.
svmfeature<-rfe(sybp~.,traindata,
                sizes = c(10:24),  # Evaluate subsets containing 10-24 features.
                rfeControl = rfeCNTL,
                methods = "svmRadial")
rid<-svmfeature$optVariables
traindata2<-traindata[,c('sybp',rid)]
testdata2<-testdata[,c('sybp',rid)]
x1<-as.matrix(traindata2[,rid])
x2<-as.matrix(traindata2[,'sybp'])
listdata2<-lapply(listdata, function(x){x[, c('sybp', rid)]})

# Fit the downstream model using the features selected above.
set.seed(seed)
fit = cv.glmnet(x1, x2, family = "gaussian", alpha = 0, nfolds = 10)
RS<-predict(fit,type = 'link', newx = as.matrix(testdata2[,-1]), s = fit$lambda.min)
RSdata<-as.numeric(RS)
rs<-lapply(listdata2,function(x){cbind(x[,1],RS=predict(fit,type = 'link', newx = as.matrix(x[,-1]), s = fit$lambda.min))})
rmsedata<-rmse(testdata2[,1], RSdata)
resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
  rownames_to_column('ID')
resultdata$Model<-paste0('SVR + ', 'Ridge')
result<-rbind(result,resultdata)

##################################
#### 7-5.SVR + Enet ####
##################################

set.seed(seed)
# Fit the model.
# Tune radial SVR hyperparameters.
bestfit<-tune.svm(x = traindata[,-1],y = traindata[,1],type = "eps-regression",
                  kernel = "radial",
                  gamma = 5*10^(-2:2),
                  cost = c(0.001,0.01,0.1,1,1.5,10,50,100))
bestgamma<-bestfit$best.parameters$gamma
bestcost<-bestfit$best.parameters$cost
# Fit radial SVR using the selected hyperparameters.
fit<-svm(sybp~.,traindata,type = "eps-regression",
         kernel = "radial",
         scale = F,
         gamma = bestgamma,
         cost = bestcost)
# Select features.
# Define recursive feature elimination with cross-validation.
rfeCNTL<-rfeControl(functions = lmFuncs,method = "cv",number = 10)
# Run recursive feature elimination.
svmfeature<-rfe(sybp~.,traindata,
                sizes = c(10:24),  # Evaluate subsets containing 10-24 features.
                rfeControl = rfeCNTL,
                methods = "svmRadial")
rid<-svmfeature$optVariables
traindata2<-traindata[,c('sybp',rid)]
testdata2<-testdata[,c('sybp',rid)]
x1<-as.matrix(traindata2[,rid])
x2<-as.matrix(traindata2[,'sybp'])
listdata2<-lapply(listdata, function(x){x[, c('sybp', rid)]})

# Fit the downstream model using the features selected above.
for (alpha in seq(0.1, 0.9, 0.1)){
  set.seed(seed)
  fit = cv.glmnet(x1, x2, family = "gaussian", alpha = alpha, nfolds = 10)
  RS<-predict(fit,type = 'link', newx = as.matrix(testdata2[,-1]), s = fit$lambda.min)
  RSdata<-as.numeric(RS)
  rs<-lapply(listdata2,function(x){cbind(x[,1],RS=predict(fit,type = 'link', newx = as.matrix(x[,-1]), s = fit$lambda.min))})
  rmsedata<-rmse(testdata2[,1], RSdata)
  resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
    rownames_to_column('ID')
  resultdata$Model<-paste0('SVR + ', 'Enet', '[α=', alpha, ']')
  result<-rbind(result,resultdata)
}

##################################
#### 7-6.SVR + GBM ####
##################################

set.seed(seed)
# Fit the model.
# Tune radial SVR hyperparameters.
bestfit<-tune.svm(x = traindata[,-1],y = traindata[,1],type = "eps-regression",
                  kernel = "radial",
                  gamma = 5*10^(-2:2),
                  cost = c(0.001,0.01,0.1,1,1.5,10,50,100))
bestgamma<-bestfit$best.parameters$gamma
bestcost<-bestfit$best.parameters$cost
# Fit radial SVR using the selected hyperparameters.
fit<-svm(sybp~.,traindata,type = "eps-regression",
         kernel = "radial",
         scale = F,
         gamma = bestgamma,
         cost = bestcost)
# Select features.
# Define recursive feature elimination with cross-validation.
rfeCNTL<-rfeControl(functions = lmFuncs,method = "cv",number = 10)
# Run recursive feature elimination.
svmfeature<-rfe(sybp~.,traindata,
                sizes = c(10:24),  # Evaluate subsets containing 10-24 features.
                rfeControl = rfeCNTL,
                methods = "svmRadial")
rid<-svmfeature$optVariables
traindata2<-traindata[,c('sybp',rid)]
testdata2<-testdata[,c('sybp',rid)]
listdata2<-lapply(listdata, function(x){x[, c('sybp', rid)]})

# Fit the downstream model using the features selected above.
set.seed(seed)
fit <- gbm(sybp~.,traindata2,distribution = 'gaussian',
           n.trees = 10000,
           interaction.depth = 3,
           n.minobsinnode = 10,
           shrinkage = 0.001,
           cv.folds = 10, n.cores = 6)
best <- which.min(fit$cv.error)
set.seed(seed)
fit <- gbm(sybp~.,traindata2,distribution = 'gaussian',
           n.trees = best,
           interaction.depth = 3,
           n.minobsinnode = 10,
           shrinkage = 0.001,
           cv.folds = 10, n.cores = 8)
RS<-predict(fit,newdata = testdata2,n.trees = best,type = 'link')
RSdata<-as.numeric(RS)
rs<-lapply(listdata2,function(x){cbind(x[,1],RS=as.numeric(predict(fit,newdata = x,n.trees = best,type = 'link')))})
rmsedata<-rmse(testdata2[,1], RSdata)
resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
  rownames_to_column('ID')
resultdata$Model<-paste0('SVR + ', 'GBM')
result<-rbind(result,resultdata)

##################################
#### 7-7.SVR + XGBoost ####
##################################

set.seed(seed)
# Fit the model.
# Tune radial SVR hyperparameters.
bestfit<-tune.svm(x = traindata[,-1],y = traindata[,1],type = "eps-regression",
                  kernel = "radial",
                  gamma = 5*10^(-2:2),
                  cost = c(0.001,0.01,0.1,1,1.5,10,50,100))
bestgamma<-bestfit$best.parameters$gamma
bestcost<-bestfit$best.parameters$cost
# Fit radial SVR using the selected hyperparameters.
fit<-svm(sybp~.,traindata,type = "eps-regression",
         kernel = "radial",
         scale = F,
         gamma = bestgamma,
         cost = bestcost)
# Select features.
# Define recursive feature elimination with cross-validation.
rfeCNTL<-rfeControl(functions = lmFuncs,method = "cv",number = 10)
# Run recursive feature elimination.
svmfeature<-rfe(sybp~.,traindata,
                sizes = c(10:24),  # Evaluate subsets containing 10-24 features.
                rfeControl = rfeCNTL,
                methods = "svmRadial")
rid<-svmfeature$optVariables
traindata2<-traindata[,c('sybp',rid)]
testdata2<-testdata[,c('sybp',rid)]
listdata2<-lapply(listdata, function(x){x[, c('sybp', rid)]})

# Fit the downstream model using the features selected above.
set.seed(seed)
cvfit<-xgb.cv(data = as.matrix(traindata2[,-1]),label = as.matrix(traindata2[,1]),objective="reg:squarederror",
              nrounds = 1000,
              nfold = 10)
best <- which.min(cvfit$evaluation_log$train_rmse_mean)
fit<-xgboost(data = as.matrix(traindata2[,-1]),label = as.matrix(traindata2[,1]),objective="reg:squarederror",
             nrounds = best)
RS<-predict(fit,as.matrix(testdata2[,-1]))
RSdata<-as.numeric(RS)
rs<-lapply(listdata2,function(x){cbind(x[,1],RS=as.numeric(predict(fit,as.matrix(x[,-1,drop = FALSE]))))})
rmsedata<-rmse(testdata2[,1], RSdata)
resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
  rownames_to_column('ID')
resultdata$Model<-paste0('SVR + ', 'XGBoost')
result<-rbind(result,resultdata)

##################################
#### 7-8.SVR + KNN ####
##################################

set.seed(seed)
# Fit the model.
# Tune radial SVR hyperparameters.
bestfit<-tune.svm(x = traindata[,-1],y = traindata[,1],type = "eps-regression",
                  kernel = "radial",
                  gamma = 5*10^(-2:2),
                  cost = c(0.001,0.01,0.1,1,1.5,10,50,100))
bestgamma<-bestfit$best.parameters$gamma
bestcost<-bestfit$best.parameters$cost
# Fit radial SVR using the selected hyperparameters.
fit<-svm(sybp~.,traindata,type = "eps-regression",
         kernel = "radial",
         scale = F,
         gamma = bestgamma,
         cost = bestcost)
# Select features.
# Define recursive feature elimination with cross-validation.
rfeCNTL<-rfeControl(functions = lmFuncs,method = "cv",number = 10)
# Run recursive feature elimination.
svmfeature<-rfe(sybp~.,traindata,
                sizes = c(10:24),  # Evaluate subsets containing 10-24 features.
                rfeControl = rfeCNTL,
                methods = "svmRadial")
rid<-svmfeature$optVariables
traindata2<-traindata[,c('sybp',rid)]
testdata2<-testdata[,c('sybp',rid)]
listdata2<-lapply(listdata, function(x){x[, c('sybp', rid)]})

# Fit the downstream model using the features selected above.
set.seed(seed)
traincontrol<-trainControl(method = "cv",number = 10)
traink<-expand.grid(.k = seq(1,20,by = 1))
bestfit<-train(sybp~.,traindata2,
               method = "knn",
               trControl = traincontrol,
               tuneGrid = traink)
bestk<-bestfit$bestTune$k
fit<-knnreg(sybp~.,traindata2,k = bestk)
RS<-predict(fit,testdata2)
RSdata<-as.numeric(RS)
rs<-lapply(listdata2,function(x){cbind(x[,1],RS=predict(fit,newdata = x))})
rmsedata<-rmse(testdata2[,1], RSdata)
resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
  rownames_to_column('ID')
resultdata$Model<-paste0('SVR + ', 'KNN')
result<-rbind(result,resultdata)



##################################
#### 8-1.KNN ####
##################################

set.seed(seed)
# Fit the model.
# Tune the number of neighbors by cross-validation.
traincontrol<-trainControl(method = "cv",number = 10)
traink<-expand.grid(.k = seq(1,20,by = 1))
bestfit<-train(sybp~.,traindata,
               method = "knn",
               trControl = traincontrol,
               tuneGrid = traink)
bestk<-bestfit$bestTune$k
# Fit KNN using the selected number of neighbors.
fit<-knnreg(sybp~.,traindata,k = bestk)
# Generate predictions.
RS<-predict(fit,testdata)
RSdata<-as.numeric(RS)
rs<-lapply(listdata,function(x){cbind(x[,1],RS=predict(fit,newdata = x))})
# Calculate RMSE and append results.
rmsedata<-rmse(testdata[,1], RSdata)
resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
  rownames_to_column('ID')
resultdata$Model<-'KNN'
result<-rbind(result,resultdata)



##################################
#### 9-1.RFE + RRF ####
##################################

set.seed(seed)
# Select features.
# Define recursive feature elimination with cross-validation.
rfeCNTL<-rfeControl(functions = lmFuncs,method = "cv",number = 10)
# Run recursive feature elimination.
knnfeature<-rfe(sybp~.,traindata,
                sizes = c(10:24),  # Evaluate subsets containing 10-24 features.
                rfeControl = rfeCNTL)
rid<-knnfeature$optVariables
traindata2<-traindata[,c('sybp',rid)]
testdata2<-testdata[,c('sybp',rid)]
x1<-as.matrix(traindata2[,rid])
x2<-as.matrix(traindata2[,'sybp'])
listdata2<-lapply(listdata, function(x){x[, c('sybp', rid)]})

# Fit the downstream model using the features selected above.
set.seed(seed)
fit<-rfsrc(sybp~.,traindata2,
           ntree = 10000,nodesize = rf_nodesize,
           splitrule = 'mse',
           importance = T,
           proximity = T,
           forest = T,
           seed = seed)
RS<-predict(fit,newdata = testdata2)
RSdata<-as.numeric(RS$predicted)
rs<-lapply(listdata2,function(x){cbind(x[,1],RS=predict(fit,newdata = x)$predicted)})
rmsedata<-rmse(testdata2[,1], RSdata)
resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
  rownames_to_column('ID')
resultdata$Model<-paste0('RFE + ', 'RRF')
result<-rbind(result,resultdata)


##################################
#### 9-2.RFE + Lasso ####
##################################

set.seed(seed)
# Select features.
# Define recursive feature elimination with cross-validation.
rfeCNTL<-rfeControl(functions = lmFuncs,method = "cv",number = 10)
# Run recursive feature elimination.
knnfeature<-rfe(sybp~.,traindata,
                sizes = c(10:24),  # Evaluate subsets containing 10-24 features.
                rfeControl = rfeCNTL)
rid<-knnfeature$optVariables
traindata2<-traindata[,c('sybp',rid)]
testdata2<-testdata[,c('sybp',rid)]
x1<-as.matrix(traindata2[,rid])
x2<-as.matrix(traindata2[,'sybp'])
listdata2<-lapply(listdata, function(x){x[, c('sybp', rid)]})

# Fit the downstream model using the features selected above.
set.seed(seed)
fit = cv.glmnet(x1, x2, family = "gaussian", alpha = 1, nfolds = 10)
RS<-predict(fit,type = 'link', newx = as.matrix(testdata2[,-1]), s = fit$lambda.min)
RSdata<-as.numeric(RS)
rs<-lapply(listdata2,function(x){cbind(x[,1],RS=predict(fit,type = 'link', newx = as.matrix(x[,-1]), s = fit$lambda.min))})
rmsedata<-rmse(testdata2[,1], RSdata)
resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
  rownames_to_column('ID')
resultdata$Model<-paste0('RFE + ', 'Lasso')
result<-rbind(result,resultdata)

##################################
#### 9-3.RFE + Ridge ####
##################################

set.seed(seed)
# Select features.
# Define recursive feature elimination with cross-validation.
rfeCNTL<-rfeControl(functions = lmFuncs,method = "cv",number = 10)
# Run recursive feature elimination.
knnfeature<-rfe(sybp~.,traindata,
                sizes = c(10:24),  # Evaluate subsets containing 10-24 features.
                rfeControl = rfeCNTL)
rid<-knnfeature$optVariables
traindata2<-traindata[,c('sybp',rid)]
testdata2<-testdata[,c('sybp',rid)]
x1<-as.matrix(traindata2[,rid])
x2<-as.matrix(traindata2[,'sybp'])
listdata2<-lapply(listdata, function(x){x[, c('sybp', rid)]})

# Fit the downstream model using the features selected above.
set.seed(seed)
fit = cv.glmnet(x1, x2, family = "gaussian", alpha = 0, nfolds = 10)
RS<-predict(fit,type = 'link', newx = as.matrix(testdata2[,-1]), s = fit$lambda.min)
RSdata<-as.numeric(RS)
rs<-lapply(listdata2,function(x){cbind(x[,1],RS=predict(fit,type = 'link', newx = as.matrix(x[,-1]), s = fit$lambda.min))})
rmsedata<-rmse(testdata2[,1], RSdata)
resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
  rownames_to_column('ID')
resultdata$Model<-paste0('RFE + ', 'Ridge')
result<-rbind(result,resultdata)

##################################
#### 9-4.RFE + Enet ####
##################################

set.seed(seed)
# Select features.
# Define recursive feature elimination with cross-validation.
rfeCNTL<-rfeControl(functions = lmFuncs,method = "cv",number = 10)
# Run recursive feature elimination.
knnfeature<-rfe(sybp~.,traindata,
                sizes = c(10:24),  # Evaluate subsets containing 10-24 features.
                rfeControl = rfeCNTL)
rid<-knnfeature$optVariables
traindata2<-traindata[,c('sybp',rid)]
testdata2<-testdata[,c('sybp',rid)]
x1<-as.matrix(traindata2[,rid])
x2<-as.matrix(traindata2[,'sybp'])
listdata2<-lapply(listdata, function(x){x[, c('sybp', rid)]})

# Fit the downstream model using the features selected above.
for (alpha in seq(0.1, 0.9, 0.1)){
  set.seed(seed)
  fit = cv.glmnet(x1, x2, family = "gaussian", alpha = alpha, nfolds = 10)
  RS<-predict(fit,type = 'link', newx = as.matrix(testdata2[,-1]), s = fit$lambda.min)
  RSdata<-as.numeric(RS)
  rs<-lapply(listdata2,function(x){cbind(x[,1],RS=predict(fit,type = 'link', newx = as.matrix(x[,-1]), s = fit$lambda.min))})
  rmsedata<-rmse(testdata2[,1], RSdata)
  resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
    rownames_to_column('ID')
  resultdata$Model<-paste0('RFE + ', 'Enet', '[α=', alpha, ']')
  result<-rbind(result,resultdata)
}

##################################
#### 9-5.RFE + GBM ####
##################################

set.seed(seed)
# Select features.
# Define recursive feature elimination with cross-validation.
rfeCNTL<-rfeControl(functions = lmFuncs,method = "cv",number = 10)
# Run recursive feature elimination.
knnfeature<-rfe(sybp~.,traindata,
                sizes = c(10:24),  # Evaluate subsets containing 10-24 features.
                rfeControl = rfeCNTL)
rid<-knnfeature$optVariables
traindata2<-traindata[,c('sybp',rid)]
testdata2<-testdata[,c('sybp',rid)]
x1<-as.matrix(traindata2[,rid])
x2<-as.matrix(traindata2[,'sybp'])
listdata2<-lapply(listdata, function(x){x[, c('sybp', rid)]})

# Fit the downstream model using the features selected above.
set.seed(seed)
fit <- gbm(sybp~.,traindata2,distribution = 'gaussian',
           n.trees = 10000,
           interaction.depth = 3,
           n.minobsinnode = 10,
           shrinkage = 0.001,
           cv.folds = 10, n.cores = 6)
best <- which.min(fit$cv.error)
set.seed(seed)
fit <- gbm(sybp~.,traindata2,distribution = 'gaussian',
           n.trees = best,
           interaction.depth = 3,
           n.minobsinnode = 10,
           shrinkage = 0.001,
           cv.folds = 10, n.cores = 8)
RS<-predict(fit,newdata = testdata2,n.trees = best,type = 'link')
RSdata<-as.numeric(RS)
rs<-lapply(listdata2,function(x){cbind(x[,1],RS=as.numeric(predict(fit,newdata = x,n.trees = best,type = 'link')))})
rmsedata<-rmse(testdata2[,1], RSdata)
resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
  rownames_to_column('ID')
resultdata$Model<-paste0('RFE + ', 'GBM')
result<-rbind(result,resultdata)

##################################
#### 9-6.RFE + XGBoost ####
##################################

set.seed(seed)
# Select features.
# Define recursive feature elimination with cross-validation.
rfeCNTL<-rfeControl(functions = lmFuncs,method = "cv",number = 10)
# Run recursive feature elimination.
knnfeature<-rfe(sybp~.,traindata,
                sizes = c(10:24),  # Evaluate subsets containing 10-24 features.
                rfeControl = rfeCNTL)
rid<-knnfeature$optVariables
traindata2<-traindata[,c('sybp',rid)]
testdata2<-testdata[,c('sybp',rid)]
x1<-as.matrix(traindata2[,rid])
x2<-as.matrix(traindata2[,'sybp'])
listdata2<-lapply(listdata, function(x){x[, c('sybp', rid)]})

# Fit the downstream model using the features selected above.
set.seed(seed)
cvfit<-xgb.cv(data = as.matrix(traindata2[,-1]),label = as.matrix(traindata2[,1]),objective="reg:squarederror",
              nrounds = 1000,
              nfold = 10)
best <- which.min(cvfit$evaluation_log$train_rmse_mean)
fit<-xgboost(data = as.matrix(traindata2[,-1]),label = as.matrix(traindata2[,1]),objective="reg:squarederror",
             nrounds = best)
RS<-predict(fit,as.matrix(testdata2[,-1]))
RSdata<-as.numeric(RS)
rs<-lapply(listdata2,function(x){cbind(x[,1],RS=as.numeric(predict(fit,as.matrix(x[,-1,drop = FALSE]))))})
rmsedata<-rmse(testdata2[,1], RSdata)
resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
  rownames_to_column('ID')
resultdata$Model<-paste0('RFE + ', 'XGBoost')
result<-rbind(result,resultdata)

##################################
#### 9-7.RFE + SVR ####
##################################

set.seed(seed)
# Select features.
# Define recursive feature elimination with cross-validation.
rfeCNTL<-rfeControl(functions = lmFuncs,method = "cv",number = 10)
# Run recursive feature elimination.
knnfeature<-rfe(sybp~.,traindata,
                sizes = c(10:24),  # Evaluate subsets containing 10-24 features.
                rfeControl = rfeCNTL)
rid<-knnfeature$optVariables
traindata2<-traindata[,c('sybp',rid)]
testdata2<-testdata[,c('sybp',rid)]
x1<-as.matrix(traindata2[,rid])
x2<-as.matrix(traindata2[,'sybp'])
listdata2<-lapply(listdata, function(x){x[, c('sybp', rid)]})

# Fit the downstream model using the features selected above.
set.seed(seed)
bestfit<-tune.svm(x = traindata2[,-1],y = traindata2[,1],type = "eps-regression",
                  kernel = "radial",
                  gamma = 5*10^(-2:2),
                  cost = c(0.001,0.01,0.1,1,1.5,10,50,100))
bestgamma<-bestfit$best.parameters$gamma
bestcost<-bestfit$best.parameters$cost
fit<-svm(sybp~.,traindata2,type = "eps-regression",
         kernel = "radial",
         scale = F,
         gamma = bestgamma,
         cost = bestcost)
RS<-predict(fit,testdata2)
RSdata<-as.numeric(RS)
rs<-lapply(listdata2,function(x){cbind(x[,1],RS=predict(fit,newdata = x))})
rmsedata<-rmse(testdata2[,1], RSdata)
resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
  rownames_to_column('ID')
resultdata$Model<-paste0('RFE + ', 'SVR')
result<-rbind(result,resultdata)

##################################
#### 9-8.RFE + KNN ####
##################################

set.seed(seed)
# Select features.
# Define recursive feature elimination with cross-validation.
rfeCNTL<-rfeControl(functions = lmFuncs,method = "cv",number = 10)
# Run recursive feature elimination.
knnfeature<-rfe(sybp~.,traindata,
                sizes = c(10:24),  # Evaluate subsets containing 10-24 features.
                rfeControl = rfeCNTL)
rid<-knnfeature$optVariables
traindata2<-traindata[,c('sybp',rid)]
testdata2<-testdata[,c('sybp',rid)]
x1<-as.matrix(traindata2[,rid])
x2<-as.matrix(traindata2[,'sybp'])
listdata2<-lapply(listdata, function(x){x[, c('sybp', rid)]})

# Fit the downstream model using the features selected above.
set.seed(seed)
traincontrol<-trainControl(method = "cv",number = 10)
traink<-expand.grid(.k = seq(1,20,by = 1))
bestfit<-train(sybp~.,traindata2,
               method = "knn",
               trControl = traincontrol,
               tuneGrid = traink)
bestk<-bestfit$bestTune$k
fit<-knnreg(sybp~.,traindata2,k = bestk)
RS<-predict(fit,testdata2)
RSdata<-as.numeric(RS)
rs<-lapply(listdata2,function(x){cbind(x[,1],RS=predict(fit,newdata = x))})
rmsedata<-rmse(testdata2[,1], RSdata)
resultdata<-data.frame(RMSE=sapply(rs,function(x){as.numeric(rmse(x[,1],x[,2]))}))%>%
  rownames_to_column('ID')
resultdata$Model<-paste0('RFE + ', 'KNN')
result<-rbind(result,resultdata)



##################################
#### RMSE heatmap ####
##################################

# Summarize model performance.
result2<-result
# Reshape the performance results for the heatmap.
### pivot_wider(data, names_from, values_from)
result3<-pivot_wider(result2, names_from = 'ID', values_from = 'RMSE') %>% as.data.frame()
# Convert RMSE values to numeric.
result3[,-1]<-apply(result3[,-1], 2, as.numeric)

# Calculate mean RMSE across the training and test sets.
result3$All<-apply(result3[,2:3], 1, mean)
# Rank models by mean RMSE.
result3<-result3[order(result3$All, decreasing = F),]
rownames(result3)<-result3$Model
result4<-result3[,2:3]



# Build the RMSE heatmap.
library(ComplexHeatmap)

# Prepare the heatmap matrix.
averageRMSE<-apply(result4, 1, mean)  # Mean RMSE across the training and test sets.
averageRMSE<-sort(averageRMSE,decreasing = F)  # Sort from lowest to highest mean RMSE.
result5<-result4[names(averageRMSE),]  # Reorder the RMSE matrix.
averageRMSE<- as.numeric(format(averageRMSE, digits = 3, nsmall = 3))  # Format to three decimal places.

# Prepare heatmap annotations and dimensions.
# Define cohort colors and labels.
CohortCol<-c("#CF394B","#122B3B")
names(CohortCol) <- colnames(result5)
# Add the mean-RMSE row annotation.
rowannotation<-rowAnnotation(bar = anno_barplot(averageRMSE, bar_width = 0.8, border = FALSE,
                                                gp = gpar(fill = "#C3613A", col = NA),
                                                add_numbers = T, numbers_offset = unit(-10, "mm"),
                                                axis_param = list("labels_rot" = 0),
                                                numbers_gp = gpar(fontsize = 9, col = "white"),
                                                width = unit(3, "cm")),
                             show_annotation_name = F)
# Add cohort column annotations.
colannotation<-columnAnnotation("Cohort" = colnames(result5),
                                col = list("Cohort" = CohortCol),
                                show_annotation_name = F)
# Set heatmap cell dimensions.
cellwidth<-1
cellheight<-0.5

# Draw and save the heatmap.
hm<-Heatmap(as.matrix(result5), name = "RMSE",
            right_annotation = rowannotation, 
            top_annotation = colannotation,
            col = c("#F3A944","#3E2351","#123151"),  # Color scale.
            rect_gp = gpar(col = "#FFFFFF", lwd = 1),  # White cell borders.
            cluster_columns = FALSE, cluster_rows = FALSE,  # Preserve the model and cohort order.
            show_column_names = FALSE, 
            show_row_names = TRUE,
            row_names_side = "left",
            width = unit(cellwidth * ncol(result5) + 2, "cm"),
            height = unit(cellheight * nrow(result5), "cm"),
            column_split = factor(colnames(result5), levels = colnames(result5)), 
            column_title = NULL,
            cell_fun = function(j, i, x, y, w, h, col) {
              grid.text(label = format(result5[i, j], digits = 3, nsmall = 3),
                        x, y, gp = gpar(fontsize = 10))
            })
pdf(file.path(figure_dir, "rmse_placental_weight.pdf"), width = cellwidth * ncol(result5) + 5, height = cellheight * nrow(result5) * 0.45)
draw(hm)
invisible(dev.off())



