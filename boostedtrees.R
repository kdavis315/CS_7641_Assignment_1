#kdavis315
#Assignment1

#Required packages
require(caret)
require(ada)
require(plyr)

#citations
#http://topepo.github.io/caret/index.html
citation(caret)
citation(ada)
citation(plyr)



#Read data files
wine <- read.csv("winequality-white.csv")
bank <- read.csv("bank.csv")

###
#Wine Decision Tree
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
treefit1 <- caret::train(HighQuality~., data = wine_train_set, method = "ada", trControl=kcv_control)
treefit1
varImp(treefit1)

treegrid <- expand.grid(iter = 150, maxdepth = 3, nu =.1)
learning_curve_data <- learing_curve_dat(dat=wine_train_set, outcome = "HighQuality", test_prop = .25, method="ada", metric="Accuracy", trControl=kcv_control, tuneGrid=treegrid)
new_learning_curve <- learning_curve_data[learning_curve_data$Data!="Resampling",]
ggplot(new_learning_curve, aes(x=Training_Size, y=Accuracy, color=Data))+
  geom_smooth(method=loess, span=.8)+
  theme_bw()

#the tree
wine_train_split <- sample(1:nrow(wine_train_set), 2000)
wine_training_set <- wine_train_set[wine_train_split,]
wine_val_set <- wine_train_set[-wine_train_split,]

HighQuality2 <- ifelse(wine_training_set$HighQuality == "No", 0, 1)
wine_training_set <- data.frame(wine_training_set, HighQuality2)
wine_training_set$HighQuality <- NULL
HighQuality3 <- ifelse(wine_val_set$HighQuality == "No", 0, 1)
wine_val_set <- data.frame(wine_val_set, HighQuality3)
wine_val_set$HighQuality <- NULL
HighQuality4 <- ifelse(wine_test_set$HighQuality == "No", 0, 1)
wine_test_set <- data.frame(wine_test_set, HighQuality4)
wine_test_set$HighQuality <- NULL

wine_tree <- ada(HighQuality2~., wine_training_set, iter = 150, nu =.1, type="discrete")

#Confusion matrix
#training set
tree.fitted <- predict(wine_tree, wine_training_set)
confusionMatrix(wine_training_set$HighQuality2, tree.fitted)

#validation set
tree.pred <- predict(wine_tree, wine_val_set)
confusionMatrix(wine_val_set$HighQuality3, tree.pred)

#test set
wine_tree.pred <- predict(wine_tree, wine_test_set)
confusionMatrix(wine_test_set$HighQuality4, wine_tree.pred)


###
#Bank Decision Tree
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
banktreefit <- caret::train(bank.y~., data = bank_train_set, method = "ada", trControl=bankkcv_control)
banktreefit

banktreegrid <- expand.grid(iter = 150, maxdepth = 2, nu =.1)
banklearning_curve_data <- learing_curve_dat(dat=bank_train_set, outcome = "bank.y", test_prop = .25, method="ada", metric="Accuracy", trControl=bankkcv_control, tuneGrid=banktreegrid)
banknew_learning_curve <- banklearning_curve_data[banklearning_curve_data$Data!="Resampling",]
ggplot(banknew_learning_curve, aes(x=Training_Size, y=Accuracy, color=Data))+
  geom_smooth(method=loess, span=.8)+
  theme_bw()

#the tree
bank_train_split <- createDataPartition(bank_train_set$bank.y, p=.56, list=FALSE, times=1)
bank_training_set <- bank_train_set[-bank_train_split,]
bank_val_set <- bank_train_set[bank_train_split,]

y2 <- ifelse(bank_training_set$bank.y == "no", 0, 1)
bank_training_set <- data.frame(bank_training_set, y2)
bank_training_set$bank.y <- NULL
y3 <- ifelse(bank_val_set$bank.y == "no", 0, 1)
bank_val_set <- data.frame(bank_val_set, y3)
bank_val_set$bank.y <- NULL
y4 <- ifelse(bank_test_set$bank.y == "no", 0, 1)
bank_test_set <- data.frame(bank_test_set, y4)
bank_test_set$bank.y <- NULL

bank_tree <- ada(y2~., bank_training_set, iter = 100, nu =.1, type="discrete")

#Confusion matrix
#training set
tree.fitted <- predict(bank_tree, bank_training_set)
confusionMatrix(bank_training_set$y2, tree.fitted)

#validation set
tree.pred <- predict(bank_tree, bank_val_set)
confusionMatrix(bank_val_set$y3, tree.pred)

#test set
tree.pred <- predict(bank_tree, bank_test_set)
confusionMatrix(bank_test_set$y4, tree.pred)

