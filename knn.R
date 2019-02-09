#kdavis315
#Assignment1

#Required packages
require(caret)
require(class)
require(DMwR)

#citations
#http://topepo.github.io/caret/index.html
citation(caret)
citation(class)
citation(DMwR)


#Read data files
wine <- read.csv("winequality-white.csv")
bank <- read.csv("bank.csv")

###
#Wine KNN
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
knnfit1 <- caret::train(HighQuality~., data = wine_train_set, method = "knn", trControl=kcv_control)
knnfit1
varImp(knnfit1)

knngrid <- expand.grid(k=7)
learning_curve_data <- learing_curve_dat(dat=wine_train_set, outcome = "HighQuality", test_prop = .25, method="knn", metric="Accuracy", trControl=kcv_control, tuneGrid=knngrid)
new_learning_curve <- learning_curve_data[learning_curve_data$Data!="Resampling",]
ggplot(new_learning_curve, aes(x=Training_Size, y=Accuracy, color=Data))+
  geom_smooth(method=loess, span=.8)+
  theme_bw()


#knn
wine_train_split <- sample(1:nrow(wine_train_set), 2000)
wine_training_set <- wine_train_set[wine_train_split,]
wine_val_set <- wine_train_set[-wine_train_split,]

wine_knn1 <- kNN(HighQuality~., wine_training_set, wine_val_set, k=5)
wine_knn2 <- kNN(HighQuality~., wine_training_set, wine_val_set, k=6)
wine_knn3 <- kNN(HighQuality~., wine_training_set, wine_val_set, k=7)
wine_knn4 <- kNN(HighQuality~., wine_training_set, wine_val_set, k=8)
wine_knn5 <- kNN(HighQuality~., wine_training_set, wine_val_set, k=9)

winetest_knn1 <- kNN(HighQuality~., wine_training_set, wine_test_set, k=5)
winetest_knn2 <- kNN(HighQuality~., wine_training_set, wine_test_set, k=6)
winetest_knn3 <- kNN(HighQuality~., wine_training_set, wine_test_set, k=7)
winetest_knn4 <- kNN(HighQuality~., wine_training_set, wine_test_set, k=8)
winetest_knn5 <- kNN(HighQuality~., wine_training_set, wine_test_set, k=9)

#Confusion matrix
#validation set
confusionMatrix(wine_knn1, wine_val_set$HighQuality)
confusionMatrix(wine_knn2, wine_val_set$HighQuality)
confusionMatrix(wine_knn3, wine_val_set$HighQuality)
confusionMatrix(wine_knn4, wine_val_set$HighQuality)
confusionMatrix(wine_knn5, wine_val_set$HighQuality)

#test set
confusionMatrix(winetest_knn1, wine_test_set$HighQuality)
confusionMatrix(winetest_knn2, wine_test_set$HighQuality)
confusionMatrix(winetest_knn3, wine_test_set$HighQuality)
confusionMatrix(winetest_knn4, wine_test_set$HighQuality)
confusionMatrix(winetest_knn3, wine_test_set$HighQuality)



###
#Bank KNN
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
bankkcv_control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
bankknnfit <- caret::train(bank.y~., data = bank_train_set, method = "knn", trControl=bankkcv_control)
bankknnfit
varImp(bankknnfit)

bankknngrid <- expand.grid(k=9)
banklearning_curve_data <- learing_curve_dat(dat=bank_train_set, outcome = "bank.y", test_prop = .25, method="knn", metric="Accuracy", trControl=bankkcv_control, tuneGrid=bankknngrid)
banknew_learning_curve <- banklearning_curve_data[banklearning_curve_data$Data!="Resampling",]
ggplot(banknew_learning_curve, aes(x=Training_Size, y=Accuracy, color=Data))+
  geom_smooth(method=loess, span=.8)+
  theme_bw()

#knn
bank_train_split <- createDataPartition(bank_train_set$bank.y, p=.53, list=FALSE, times=1)
bank_training_set <- bank_train_set[-bank_train_split,]
bank_val_set <- bank_train_set[bank_train_split,]

bank_knn1 <- kNN(bank.y~., bank_training_set, bank_val_set, k=7)
bank_knn2 <- kNN(bank.y~., bank_training_set, bank_val_set, k=8)
bank_knn3 <- kNN(bank.y~., bank_training_set, bank_val_set, k=9)
bank_knn4 <- kNN(bank.y~., bank_training_set, bank_val_set, k=10)
bank_knn5 <- kNN(bank.y~., bank_training_set, bank_val_set, k=11)

banktest_knn1 <- kNN(bank.y~., bank_training_set, bank_test_set, k=7)
banktest_knn2 <- kNN(bank.y~., bank_training_set, bank_test_set, k=8)
banktest_knn3 <- kNN(bank.y~., bank_training_set, bank_test_set, k=9)
banktest_knn4 <- kNN(bank.y~., bank_training_set, bank_test_set, k=10)
banktest_knn5 <- kNN(bank.y~., bank_training_set, bank_test_set, k=11)

#Confusion matrix
#validation set
confusionMatrix(bank_val_set$bank.y, bank_knn1)
confusionMatrix(bank_val_set$bank.y, bank_knn2)
confusionMatrix(bank_val_set$bank.y, bank_knn3)
confusionMatrix(bank_val_set$bank.y, bank_knn4)
confusionMatrix(bank_val_set$bank.y, bank_knn5)

#test set
confusionMatrix(bank_test_set$bank.y, banktest_knn1)
confusionMatrix(bank_test_set$bank.y, banktest_knn2)
confusionMatrix(bank_test_set$bank.y, banktest_knn3)
confusionMatrix(bank_test_set$bank.y, banktest_knn4)
confusionMatrix(bank_test_set$bank.y, banktest_knn5)


