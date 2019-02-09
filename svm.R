#kdavis315
#Assignment1

#Required packages
require(caret)
require(kernlab)

#citations
#http://topepo.github.io/caret/index.html
citation(caret)
citation(kernlab)



#Read data files
wine <- read.csv("winequality-white.csv")
bank <- read.csv("bank.csv")

###
#Wine SVM
###
#Wine classification data pre-processing
#Classification: high quality - "Yes" or "No"
HighQuality <- ifelse(wine$quality <= 6, "No", "Yes")
#Add it to the data
winedf <- data.frame(wine, HighQuality)
#remove numeric quality from winedf (still there in wine)
winedf$quality <- NULL

#Create train and test sets
set.seed(8)
wine_test_split <- sample(1:nrow(winedf), 1200)
wine_test_set <- winedf[wine_test_split,]
wine_train_set <- winedf[-wine_test_split,]

#tuning and learning curve
kcv_control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
wine_svmLinear1 <- caret::train(HighQuality~., data = wine_train_set, method = "svmLinear", trControl=kcv_control)
wine_svmLinear1
varImp(wine_svmLinear1)

wine_svmPoly1 <- caret::train(HighQuality~., data = wine_train_set, method = "svmPoly", trControl=kcv_control)
wine_svmPoly1
varImp(wine_svmPoly1)

wine_svmRB1 <- caret::train(HighQuality~., data = wine_train_set, method = "svmRadial", trControl=kcv_control)
wine_svmRB1
varImp(wine_svmRB1)

wine_paramgrid_linear <- expand.grid(C=1)
linear_lc_data <- learing_curve_dat(dat=wine_train_set, outcome = "HighQuality", test_prop = .25, method="svmLinear", metric="Accuracy", trControl=kcv_control, tuneGrid=wine_paramgrid_linear)
linear_lc <- linear_lc_data[linear_lc_data$Data!="Resampling",]
ggplot(linear_lc, aes(x=Training_Size, y=Accuracy, color=Data))+
  geom_smooth(method=loess, span=.8)+
  theme_bw()

wine_paramgrid_poly <- expand.grid(degree=3, scale=.1, C=.5)
poly_lc_data <- learing_curve_dat(dat=wine_train_set, outcome = "HighQuality", test_prop = .25, method="svmPoly", metric="Accuracy", trControl=kcv_control, tuneGrid=wine_paramgrid_poly)
poly_lc <- poly_lc_data[poly_lc_data$Data!="Resampling",]
ggplot(poly_lc, aes(x=Training_Size, y=Accuracy, color=Data))+
  geom_smooth(method=loess, span=.8)+
  theme_bw()

wine_paramgrid_RB <- expand.grid(sigma= .07990059, C=1)
RB_lc_data <- learing_curve_dat(dat=wine_train_set, outcome = "HighQuality", test_prop = .25, method="svmRadial", metric="Accuracy", trControl=kcv_control, tuneGrid=wine_paramgrid_RB)
RB_lc <- RB_lc_data[RB_lc_data$Data!="Resampling",]
ggplot(RB_lc, aes(x=Training_Size, y=Accuracy, color=Data))+
  geom_smooth(method=loess, span=.8)+
  theme_bw()


#the svm
linear_train_split <- sample(1:nrow(wine_train_set), 1600)
linear_training_set <- wine_train_set[linear_train_split,]
linear_val_set <- wine_train_set[-linear_train_split,]

poly_train_split <- sample(1:nrow(wine_train_set), 1900)
poly_training_set <- wine_train_set[poly_train_split,]
poly_val_set <- wine_train_set[-poly_train_split,]

RB_train_split <- sample(1:nrow(wine_train_set), 2000)
RB_training_set <- wine_train_set[RB_train_split,]
RB_val_set <- wine_train_set[-RB_train_split,]

wine_svmLinear <- ksvm(HighQuality~., linear_training_set, C=1, kernel="vanilladot")
wine_svmPoly <- ksvm(HighQuality~., poly_training_set, degree=3, scale=.1, C=.5, kernel="polydot")
wine_svmRB <- ksvm(HighQuality~., RB_training_set, sigma= .07990059, C=1, kernel="rbfdot")


#Confusion matrix
#training set
svmLinear.fitted <- predict(wine_svmLinear, linear_training_set)
confusionMatrix(linear_training_set$HighQuality, svmLinear.fitted)

svmPoly.fitted <- predict(wine_svmPoly, poly_training_set)
confusionMatrix(poly_training_set$HighQuality, svmPoly.fitted)

svmRB.fitted <- predict(wine_svmRB, RB_training_set)
confusionMatrix(RB_training_set$HighQuality, svmRB.fitted)

#validation set
svmLinear.pred <- predict(wine_svmLinear, linear_val_set)
confusionMatrix(linear_val_set$HighQuality, svmLinear.pred)

svmPoly.pred <- predict(wine_svmPoly, poly_val_set)
confusionMatrix(poly_val_set$HighQuality, svmPoly.pred)

svmRB.pred <- predict(wine_svmRB, RB_val_set)
confusionMatrix(RB_val_set$HighQuality, svmRB.pred)

#test set
svmLinear.pred <- predict(wine_svmLinear, wine_test_set)
confusionMatrix(wine_test_set$HighQuality, svmLinear.pred)

svmPoly.pred <- predict(wine_svmPoly, wine_test_set)
confusionMatrix(wine_test_set$HighQuality, svmPoly.pred)

svmRB.pred <- predict(wine_svmRB, wine_test_set)
confusionMatrix(wine_test_set$HighQuality, svmRB.pred)




###
#Bank SVM
###
#Create train and test sets
bank$job <- gsub("-", "", bank$job)
dummies <- dummyVars(y ~ ., data = bank)
newbank <- predict(dummies, newdata = bank)
bankdata <- data.frame(newbank, bank$y)
set.seed(8)
bank_test_split <- createDataPartition(bankdata$bank.y, p=.75, list=FALSE, times=1)
bank_test_set <- bankdata[-bank_test_split,]
bank_train_set <- bankdata[bank_test_split,]

#tuning and learning curve
kcv_control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
bank_svmLinear1 <- caret::train(bank.y~., data = bank_train_set, method = "svmLinear", trControl=kcv_control)
bank_svmLinear1

bank_svmPoly1 <- caret::train(bank.y~., data = bank_train_set, method = "svmPoly", trControl=kcv_control)
bank_svmPoly1

bank_svmRB1 <- caret::train(bank.y~., data = bank_train_set, method = "svmRadial", trControl=kcv_control)
bank_svmRB1 <- caret::train(bank.y~., data = bank_train_set, method = "svmRadial", trControl=kcv_control)
bank_svmRB1

bank_paramgrid_linear <- expand.grid(C=1)
linear_lc_data <- learing_curve_dat(dat=bank_train_set, outcome = "bank.y", test_prop = .25, method="svmLinear", metric="Accuracy", trControl=kcv_control, tuneGrid=bank_paramgrid_linear)
linear_lc <- linear_lc_data[linear_lc_data$Data!="Resampling",]
ggplot(linear_lc, aes(x=Training_Size, y=Accuracy, color=Data))+
  geom_smooth(method=loess, span=.8)+
  theme_bw()

bank_paramgrid_poly <- expand.grid(degree=1, scale=.1, C=.1)
poly_lc_data <- learing_curve_dat(dat=bank_train_set, outcome = "bank.y", test_prop = .25, method="svmPoly", metric="Accuracy", trControl=kcv_control, tuneGrid=bank_paramgrid_poly)
poly_lc <- poly_lc_data[poly_lc_data$Data!="Resampling",]
ggplot(poly_lc, aes(x=Training_Size, y=Accuracy, color=Data))+
  geom_smooth(method=loess, span=.8)+
  theme_bw()

bank_paramgrid_RB <- expand.grid(sigma= .0134604, C=1)
RB_lc_data <- learing_curve_dat(dat=bank_train_set, outcome = "bank.y", test_prop = .25, method="svmRadial", metric="Accuracy", trControl=kcv_control, tuneGrid=bank_paramgrid_RB)
RB_lc <- RB_lc_data[RB_lc_data$Data!="Resampling",]
ggplot(RB_lc, aes(x=Training_Size, y=Accuracy, color=Data))+
  geom_smooth(method=loess, span=.8)+
  theme_bw()

#the svm
linear_train_split <- sample(1:nrow(bank_train_set), 1500)
linear_training_set <- bank_train_set[linear_train_split,]
linear_val_set <- bank_train_set[-linear_train_split,]

poly_train_split <- sample(1:nrow(bank_train_set), 2000)
poly_training_set <- bank_train_set[poly_train_split,]
poly_val_set <- bank_train_set[-poly_train_split,]

RB_train_split <- sample(1:nrow(bank_train_set), 1700)
RB_training_set <- bank_train_set[RB_train_split,]
RB_val_set <- bank_train_set[-RB_train_split,]

bank_svmLinear <- ksvm(bank.y~., linear_training_set, C=1, kernel="vanilladot")
bank_svmPoly <- ksvm(bank.y~., poly_training_set, degree=1, scale=.1, C=.1, kernel="polydot")
bank_svmRB <- ksvm(bank.y~., RB_training_set, sigma= .01317368, C=1, kernel="rbfdot")

#Confusion matrix
#training set
svmLinear.fitted <- predict(bank_svmLinear, linear_training_set)
confusionMatrix(linear_training_set$bank.y, svmLinear.fitted)

svmPoly.fitted <- predict(bank_svmPoly, poly_training_set)
confusionMatrix(poly_training_set$bank.y, svmPoly.fitted)

svmRB.fitted <- predict(bank_svmRB, RB_training_set)
confusionMatrix(RB_training_set$bank.y, svmRB.fitted)

#validation set
svmLinear.pred <- predict(bank_svmLinear, linear_val_set)
confusionMatrix(linear_val_set$bank.y, svmLinear.pred)

svmPoly.pred <- predict(bank_svmPoly, poly_val_set)
confusionMatrix(poly_val_set$bank.y, svmPoly.pred)

svmRB.pred <- predict(bank_svmRB, RB_val_set)
confusionMatrix(RB_val_set$bank.y, svmRB.pred)

#test set
svmLinear.pred <- predict(bank_svmLinear, bank_test_set)
confusionMatrix(bank_test_set$bank.y, svmLinear.pred)

svmPoly.pred <- predict(bank_svmPoly, bank_test_set)
confusionMatrix(bank_test_set$bank.y, svmPoly.pred)

svmRB.pred <- predict(bank_svmRB, bank_test_set)
confusionMatrix(bank_test_set$bank.y, svmRB.pred)
