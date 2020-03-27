rm(list=ls())
options(scipen=999,digits=3)

# Pacman loads multiple libraries at the same time
library(pacman)
p_load('tidyverse','rpart','rpart.plot','Metrics','forecast','caret',
       'ggplot2', 'FNN', 'fastDummies','dataPreparation','reshape2','corrplot')

#Custom Summary Function for Cross-Validation
mape <- function(actual,pred){
  mape <- mean(abs((actual - pred)/actual))*100
  return (mape)
}
mapeSummary <- function (data,
                         lev = NULL,
                         model = NULL) {
  c(MAPE=mape(data$obs, data$pred),
    RMSE=sqrt(mean((data$obs-data$pred)^2)),
    Rsquared=summary(lm(pred ~ obs, data))$r.squared)
}


# Read in data
df <- read_csv('kc_house_data.csv')
set.seed(29)

# hist(df$price,breaks = 100)

# Visualization/ Exploration of Data
df %>% filter(price<2000000) %>% ggplot(aes(x=price)) + geom_histogram(bins=30,fill="#69b3a2", color="#e9ecef", alpha=0.9) +
  scale_x_continuous(name="Price",labels = function(x) format(x, scientific = FALSE))+
  scale_y_continuous(name="Count") +
  ggtitle("Density of Housing Prices Excluding Outliers") + theme_bw()+ theme(axis.title.x = element_text( size=18, face="bold"),
                                                                              axis.title.y = element_text(size=18, face="bold"),
                                                                              axis.text.x = element_text(size=18,face = "bold"),
                                                                              axis.text.y = element_text(size=18,face = "bold"),
                                                                              plot.title = element_text(size=30))

df %>% filter(sqft_living<7500) %>% ggplot(aes(x=sqft_living)) + geom_histogram(bins=30,fill="#69b3a2", color="#e9ecef", alpha=0.9) +
  scale_x_continuous(name="Sqft Living",labels = function(x) format(x, scientific = FALSE))+
  scale_y_continuous(name="Count") +
  ggtitle("Density of Square Footage Excluding Outliers") + theme_bw() + theme(axis.title.x = element_text( size=18, face="bold"),
                                                                               axis.title.y = element_text(size=18, face="bold"),
                                                                               axis.text.x = element_text(size=18,face = "bold"),
                                                                               axis.text.y = element_text(size=18,face = "bold"),
                                                                               plot.title = element_text(size=30))

ggplot(df, aes(fill=city,x=factor(city, levels=names(sort(table(city),
                                          decreasing=FALSE))))) + geom_bar(stat='count') +
  coord_flip() + scale_x_discrete(name="City") + scale_y_continuous(name="Count") +
  ggtitle("Count of Houses Per City") + theme_bw() + theme(axis.title.x = element_text( size=18, face="bold"),
                                                           axis.title.y = element_text(size=18, face="bold"),
                                                           axis.text.x = element_text(size=18,face = "bold"),
                                                           axis.text.y = element_text(size=18,face = "bold"),
                                                           plot.title = element_text(size=30))


ggplot(df, aes(fill=yr_built,x=factor(yr_built))) + geom_bar(stat='count') +
  scale_x_discrete(breaks=seq(1900, 2015, 10),name="Year Built") + scale_y_continuous(name="Count") +
  ggtitle("Number of Houses per Year Built") + theme_bw() + theme(axis.title.x = element_text( size=18, face="bold"),
                                                                  axis.title.y = element_text(size=18, face="bold"),
                                                                  axis.text.x = element_text(size=18, face='bold'),
                                                                  axis.text.y = element_text(size=18,face = "bold"),
                                                                  plot.title = element_text(size=30))

bx <- df %>% filter(price <2000000)%>% select(price,city) %>% melt(id.vars='city', measure.vars='price')
ggplot(bx)+geom_boxplot(aes(x=city, y=value, fill=city))+ scale_y_continuous(name="Price")+ theme_bw() + theme(axis.title.x = element_text( size=18, face="bold"), axis.title.y = element_text(size=18, face="bold"),
axis.text.x = element_text(size=18, face='bold',angle = 90, hjust = 1),
axis.text.y = element_text(size=18,face = "bold"))





# Outlier Detection & Adjustment
# After finding outliers, the following code makes adjustments 
# to bedroom and price
df$bedrooms[df$bedrooms==33] = 3
df <- df %>% filter(bedrooms != 0 & bedrooms <9)
df <- df %>% filter(df$price <2000000)

# Check for null values
sapply(df, function(x) sum(is.na(x)))

# Preprocessing Steps:
# Create variable for if house has been renovated
df <- df %>% mutate(renovated = if_else(yr_renovated>0,1,0))

# Create Age variable
df <- df %>% mutate(age = 2015 - yr_built)


# Create buckets for grade
df <- df %>% mutate(grade_new = case_when(grade >= 1  & grade <= 4 ~ "Low",
                                             grade >= 5  & grade <= 9 ~ "Average",
                                             grade >= 10  & grade <= 13 ~ "High")) 

# Build dataframe to check correlations
cor.check <- df %>% select(price, bathrooms,sqft_living,sqft_lot,floors,waterfront,view,condition,
                           sqft_basement,age, sqft_living15, sqft_lot15, renovated, bedrooms)%>% cor(.)

# Feature Selection: 
df <- df %>% select(price, bathrooms,sqft_living,sqft_lot,floors,waterfront,view,condition,grade_new,
                    sqft_basement,age,city, renovated)




##########################################################
#                 Linear Regression                      #
##########################################################

# Examine the correlation among the variables
# # # Pearson's Correlation (Parametric / Strength of Linear Rel.)
df.cor <- df %>% select(-c("grade_new",'city','view','condition'))%>% cor(.) %>% round(2)
corrplot(cor.check)
# 
# 
# # Coerce variables to factors
df.lr <- df %>% mutate(view = as.factor(view), condition = as.factor(condition),grade_new=factor(grade_new, levels=c("Low","Average","High")),
                       city = factor(city,levels=c("Seattle","Mercer Island",'Bellevue','Redmond','Sammamish','Issaquah','Woodinville','Vashon',
                                                   'Snoqualmie','Renton','North Bend','Maple Valley','Kirkland','Kent','Kenmore','Federal Way','Fall City',
                                                   'Enumclaw','Duvall','Carnation','Bothell','Black Diamond','Auburn')))


# Split dataframe to train and test 
df.lr <- df.lr %>% mutate(id=1:nrow(df.lr))

train.lr <- df.lr %>% sample_frac(.8)
test.lr <- df.lr %>% slice(-train.lr$id)


# Create the linear model
lrm <- lm(price~.-id, data=train.lr)

summary(lrm)


# Create predictions
a = predict(lrm, train.lr)
train <- train.lr %>% mutate(prediction.lr = a)

# Evaluate accuracy measures
forecast::accuracy(train$prediction.lr,train$price)

# KFolds Cross Validation / 10 Folds
cv <- trainControl(method="cv", number=10, savePredictions = TRUE, summary=mapeSummary)

model_caret <- train(price ~ .,data = df, trControl = cv, method = "lm") 

model_caret


##########################################################
#              Regression Decision Tree                  #
##########################################################

# Create regression decision tree model
dt = rpart(price~.-id, data=train.lr, method='anova', cp= 0.01, minsplit = 30, xval = 10)

# View the outputs for decision trees
prp(dt, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = 0, digits=-3, shadow.col='grey', box.palette='Blues')
printcp(dt)
plotcp(dt)


# Create cp.table and prune the tree with optimal cp value
cp.table = as_tibble(dt$cptable)
optimal.cp = cp.table %>% filter(nsplit==6)
pruned.ct = prune(dt, cp = optimal.cp$CP)
prp(pruned.ct, type=1, extra=1, under=TRUE, split.font = 2, varlen = 0, digits=-3, shadow.col='grey', box.palette='Blues')

# Plot pruned tree
rpart.plot(pruned.ct,
           box.palette = "GnBu", # color scheme
           branch.lty = 3, # dotted branch lines
           shadow.col = "gray", # shadows under the node boxes
           nn = TRUE, digits=-3) # display the node numbers, Turn off Scientific Notation

# Predict the results of model
results = predict(pruned.ct, test.lr, type="vector")
test.dt <- test.lr %>% mutate(prediction = results)

# Examine the accuracy measures
forecast::accuracy(test.dt$prediction,test.dt$price)


# KFolds Cross Validation
tGrid <- expand.grid(cp = seq(0, .02, .0001))
model.dt <- train(price~., data=df.lr[-c(14)], trControl=cv, method="rpart", na.action=na.exclude,tuneGrid = tGrid)

# Find Optimal CP under Cross Val
cv.cp <- model.dt$bestTune$cp
print(paste("10-Folds CV Best CP: ", cv.cp))


##########################################################
#                   KNN Regression                       #
##########################################################

# Create Dummy variables with fastDummies library
df <- df %>% mutate(condition = as.character(condition), view = as.character(view))
df <- dummy_cols(df, select_columns = c('view','condition','grade_new','city'))

# Remove unnecessary variables
df <- df %>% select(-c('view',"condition","city","grade_new"))

# Split the train and test dataframes
df.knn <- df %>% mutate(id=1:nrow(df))
train.knn <- df.knn %>% sample_frac(.8)
test.knn <- df.knn %>% slice(-train.knn$id)

# Standardize the numeric variables
cols <- c("sqft_lot",'sqft_living','sqft_basement','sqft_living15','sqft_lot15','age','bathrooms','floors')
scales <- build_scales(dataSet = train.knn, cols = cols, verbose = TRUE)

# Scale the Train and Test data with Train's Mean and Std Dev.
train.knn <- fastScale(dataSet = train.knn, scales = scales, verbose = TRUE)
output <- train.knn$price
train.knn <- train.knn %>% select(-c("price",'id'))
test.knn <- fastScale(dataSet = test.knn, scales = scales, verbose = TRUE) %>% select(-c('id'))


# Find optimal k_neighbors
for (i in c(1,3,5,7,9,11)){
  print(paste("Neighbors =",i))
  prediction <- knn.reg(train.knn, test.knn[,-c(1)], output, i)
  
  prediction <- prediction$pred

  test.eval <- test.knn %>% mutate(knn.pred = prediction)

  print(forecast::accuracy(test.eval$knn.pred,test.eval$price))
}


# KFolds Cross Validation (Scale during CV)
# Scales all numeric variables including dummy, so output will differ slightly
model.knn <- train(price~., data=df,trControl=cv, method="knn",preProcess=(c('center','scale')))

model.knn

