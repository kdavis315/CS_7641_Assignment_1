#kdavis315
#Assignment1

#Required packages
require(caret)
require(RSNNS)

#citations
#http://topepo.github.io/caret/index.html
#https://rdrr.io/cran/RSNNS/man/mlp.html
citation(caret)
citation(RSNNS)



#Read data files
wine <- read.csv("winequality-white.csv")
bank <- read.csv("bank.csv")

###
#Wine Neural Network
###
#Wine classification data pre-processing
#Classification: high quality - "Yes" or "No"
HighQuality <- ifelse(wine$quality <= 6, "No", "Yes")
#Add it to the data
winedf <- data.frame(wine, HighQuality)
#remove numeric quality from wine
winedf$quality <- NULL

#Create train and test sets
set.seed(8)
wine_test_split <- sample(1:nrow(winedf), 1200)
wine_test_set <- winedf[wine_test_split,]
wine_train_set <- winedf[-wine_test_split,]

x <- wine_train_set[,1:11]
y <- wine_train_set[,12]

#tuning and learning curve
kcv_control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
nnfit1 <- caret::train(x, y, method = "mlp", trControl=kcv_control)
nnfit1
varImp(nnfit1)

nngrid <- expand.grid(size=1)
learning_curve_data <- caret::learing_curve_dat(wine_train_set, outcome = "HighQuality", test_prop = .25, method="mlp", metric="Accuracy", trControl=kcv_control, tuneGrid=nngrid)
new_learning_curve <- learning_curve_data[learning_curve_data$Data!="Resampling",]
ggplot(new_learning_curve, aes(x=Training_Size, y=Accuracy, color=Data))+
  geom_smooth(method=loess, span=.8)+
  theme_bw()

#the neural network
wine_train_split <- sample(1:nrow(wine_train_set), 2000)
wine_train <- wine_train_set[wine_train_split,]
wine_val <- wine_train_set[-wine_train_split,]

wine_train_and_val <-  both.dfs <- rbind(wine_train, wine_val) 
wine_test <-  both.dfs <- rbind(wine_train, wine_test_set)

wine_tv_values <- wine_train_and_val[,1:11]
wine_tv_targets <- decodeClassLabels(wine_train_and_val[,12])

wine_test_values <- wine_test[,1:11]
wine_test_targets <- decodeClassLabels(wine_test[,12])

wine_train_and_val <- splitForTrainingAndTest(wine_tv_values, wine_tv_targets, ratio = 0.46)
wine_train_and_val <- normTrainingAndTestSet(wine_train_and_val)

wine_test <- splitForTrainingAndTest(wine_test_values, wine_test_targets, ratio = 0.375)
wine_test <- normTrainingAndTestSet(wine_test)

wine_nn <- mlp(wine_train_and_val$inputsTrain, wine_train_and_val$targetsTrain, size=1, inputsTest = wine_train_and_val$inputsTest, targetsTest = wine_train_and_val$targetsTest)

#Confusion matrix
#training set
nn.fitted <- predict(wine_nn, wine_train_and_val$inputsTrain)
pred1 <- ifelse(nn.fitted[,1] >= nn.fitted[,2], "No", "Yes")
predictions <- decodeClassLabels(pred1)
pred <- encodeClassLabels(predictions)
targetsTrain <- encodeClassLabels(wine_train_and_val$targetsTrain)
confusionMatrix(targetsTrain, nn.fitted)

#validation set
nn.pred <- predict(wine_nn, wine_train_and_val$inputsTest)
pred1 <- ifelse(nn.pred[,1] >= nn.pred[,2], "No", "Yes")
predictions <- decodeClassLabels(pred1)
pred <- encodeClassLabels(predictions)
targetsTest <- encodeClassLabels(wine_train_and_val$targetsTest)
confusionMatrix(targetsTest, pred)

#test set
nn.pred <- predict(wine_nn, wine_test$inputsTest)
pred1 <- ifelse(nn.pred[,1] >= nn.pred[,2], "No", "Yes")
predictions <- decodeClassLabels(pred1)
pred <- encodeClassLabels(predictions)
targetsTest <- encodeClassLabels(wine_test$targetsTest)
confusionMatrix(targetsTest, pred)



###
#Bank Neural Network
###
#Create train and test sets
bank$job <- gsub("-", "", bank$job)
dummies <- dummyVars(y ~ ., data = bank)
newbank <- predict(dummies, newdata = bank)
bankdata <- data.frame(newbank, bank$y)
set.seed(8)
bank_test_split <- createDataPartition(bank$y, p=.75, list=FALSE, times=1)
bank_test_set <- bankdata[-bank_test_split,]
bank_train_set <- bankdata[bank_test_split,]

x <- bank_train_set[,1:51]
y <- bank_train_set[,52]

#tuning and learning curve
bankkcv_control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
banknnfit <- caret::train(x, y, method = "mlp", trControl=bankkcv_control)
banknnfit
varImp(banknnfit)

banknngrid <- expand.grid(size=1)
banklearning_curve_data <- caret::learing_curve_dat(dat=bank_train_set, outcome = "bank.y", test_prop = .25, method="mlp", metric="Accuracy", trControl=bankkcv_control, tuneGrid=banknngrid)
banknew_learning_curve <- banklearning_curve_data[banklearning_curve_data$Data!="Resampling",]
ggplot(banknew_learning_curve, aes(x=Training_Size, y=Accuracy, color=Data))+
  geom_smooth(method=loess, span=.8)+
  theme_bw()

#the neural network
bank_train_split <- createDataPartition(bank_train_set$bank.y, p=.44, list=FALSE, times=1)
bank_train <- bank_train_set[bank_train_split,]
bank_val <- bank_train_set[-bank_train_split,]

bank_train_and_val <-  both.dfs <- rbind(bank_train, bank_val) 
bank_test <-  both.dfs <- rbind(bank_train, bank_test_set)

bank_tv_values <- bank_train_and_val[,1:51]
bank_tv_targets <- decodeClassLabels(bank_train_and_val[,52])

bank_test_values <- bank_test[,1:51]
bank_test_targets <- decodeClassLabels(bank_test[,52])

bank_train_and_val <- splitForTrainingAndTest(bank_tv_values, bank_tv_targets, ratio = 0.56)
bank_train_and_val <- normTrainingAndTestSet(bank_train_and_val)

bank_test <- splitForTrainingAndTest(bank_test_values, bank_test_targets, ratio = 0.43)
bank_test <- normTrainingAndTestSet(bank_test)

bank_nn <- mlp(bank_train_and_val$inputsTrain, bank_train_and_val$targetsTrain, size=1, inputsTest = bank_train_and_val$inputsTest, targetsTest = bank_train_and_val$targetsTest)

#Confusion matrix
#training set
nn.fitted <- predict(bank_nn, bank_train_and_val$inputsTrain)
pred1 <- ifelse(nn.fitted[,1] >= nn.fitted[,2], "No", "Yes")
predictions <- decodeClassLabels(pred1)
pred <- encodeClassLabels(predictions)
targetsTrain <- encodeClassLabels(bank_train_and_val$targetsTrain)
confusionMatrix(targetsTrain, nn.fitted)

#validation set
nn.pred <- predict(bank_nn, bank_train_and_val$inputsTest)
pred1 <- ifelse(nn.pred[,1] >= nn.pred[,2], "No", "Yes")
predictions <- decodeClassLabels(pred1)
pred <- encodeClassLabels(predictions)
targetsTest <- encodeClassLabels(bank_train_and_val$targetsTest)
table(targetsTest, pred)

#test set
nn.pred <- predict(bank_nn, bank_test$inputsTest)
pred1 <- ifelse(nn.pred[,1] >= nn.pred[,2], "No", "Yes")
predictions <- decodeClassLabels(pred1)
pred <- encodeClassLabels(predictions)
targetsTest <- encodeClassLabels(bank_test$targetsTest)
table(targetsTest, pred)

