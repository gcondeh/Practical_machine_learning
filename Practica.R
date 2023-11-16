rm(list=ls())

## Librerías

library(caret)
library(tidyverse)
library(patchwork)


training <- read.csv("pml-training.csv")

# exactly according to the specification (Class A), 
# throwing the elbows to the front (Class B), 
# lifting the dumbbell only halfway (Class C), 
# lowering the dumbbell only halfway (Class D) 
# and throwing the hips to the front (Class E)


#### Limpiar juegos de datos ####

training %>%
  mutate(classe = as.factor(classe)) %>%
  mutate(user_name = as.factor(user_name)) %>%
  mutate(cvtd_timestamp = dmy_hm(cvtd_timestamp)) -> training

##Identificamos las columnas a eliminar
## Quitamos las columnas con NA
## Quitamos las columnas que son caracteres pq sólo están presentes si window=yes y tienen muchos datos erróneos 

which(colMeans(is.na(training))>0) -> quitar
which(sapply(training, class) == "character")  -> quitar_2


## Vemos la distribución de los timestamp, X y New window con los datos, habrá que quitarlos pq están ordenadas por fecha.

featurePlot(x = training[,c(1,3:4,7)], 
            y = training$classe, 
            plot = "pairs",
            ## Add a key at the top
             auto.key = list(columns = 5))


quitar_3 <- c(1, 3:5, 7)
## Quitamos las columnas seleccionadas y la X que es un índice.

training[, -c(quitar, quitar_2, quitar_3)]-> training_wk
str(training_wk)

#### Separamos juego de datos en train y test ####

set.seed(2004)
inTrain = createDataPartition(training_wk$classe, p = 0.7, list = FALSE)
training_wk_entr = training_wk[ inTrain,]
training_wk_test = training_wk[-inTrain,]

# Comprobamos la distribución

nrow(training_wk_entr) -> todo
training_wk_entr %>%
  group_by(classe) %>%
  summarise( n(), n()/todo) -> graf_1

ggplot(graf_1, aes(x=classe, y=`n()` , fill=classe))+
  geom_col() +
  geom_text(label = round(graf_1$`n()/todo`,2), vjust = -0.5)+
  ggtitle("Distribución entrenamiento\n")+
  theme_classic()+
  theme(legend.position = "none", axis.title = element_blank())-> p1
  
nrow(training_wk_test) -> todo
training_wk_test %>%
  group_by(classe) %>%
  summarise( n(), n()/todo) -> graf_2

ggplot(graf_2, aes(x=classe, y=`n()` , fill=classe))+
  geom_col() +
  geom_text(label = round(graf_2$`n()/todo`,2), vjust = -0.5) +
  ggtitle("Distribución test\n")+
  theme_classic()+
  theme(legend.position = "none", axis.title = element_blank())-> p2

p1+p2

#### modelo árbol de decisión ####

set.seed(2212)
modelo_rpart <- train(classe ~ ., method = "rpart", data = training_wk_entr)

print(modelo_rpart$finalModel)
plot(modelo_rpart$finalModel, uniform = TRUE)
text(modelo_rpart$finalModel, use.n = TRUE, all = TRUE, cex=1)

predic_rpart <- predict(modelo_rpart, newdata = training_wk_test)

confusionMatrix(predic_rpart, training_wk_test$classe)

#### modelo svm ####


set.seed(2212)
modelo_svm <- train(classe ~ ., method = "svmLinear", data = training_wk_entr)

print(modelo_svm$finalModel)

predic_svm <- predict(modelo_svm, newdata = training_wk_test)

confusionMatrix(predic_svm, training_wk_test$classe)



#### modelo random forest ####

set.seed(2212)
# modelo_rf <- train(classe ~ ., method = "rf", data = training_wk_entr)
modelo_rf <- train(classe ~ ., method = "rf", data = training_wk_entr, trControl = trainControl(method = "cv", number = 10))

print(modelo_rf$finalModel)

predic_rf <- predict(modelo_rf, newdata = training_wk_test)
predict_prob <- predict(modelo_rf, type = "prob")

confusionMatrix(predic_rf, training_wk_test$classe)

plot(modelo_rf)
plot(modelo_rf$finalModel)

varImp(modelo_rf)

plot(varImp(modelo_rf), top = 10)

featurePlot(x = training_wk_entr[,c(2,4,42,40)], 
            y = training_wk_entr$classe, 
            plot = "pairs",
            ## Add a key at the top
            auto.key = list(columns = 5)) 


