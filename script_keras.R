
library(plyr)
v<-c(0,2,0,1,0,0,0,-1,0,-2,   -1.5,1, -1,1, -1.5,0, -1,0, -1.5,-1, -1,-1, 1,1, 1.5,1,1,0, 1.5,0, 1,-1, 1.5,-1.0   )
k<-2#dimensiones
sn<-10000#numero de especimenes
ln<-17#numero de landmarks
pares<-as.matrix(read.csv("pairs.txt",header = FALSE,sep=" "))
n<-16
des<-0.05
  
B<-array(v,c(k,ln,sn))
for(j in 1:sn){
  lp<-sample(c(1:2,4:17),n,replace = FALSE)
  B[,lp,j]<-(rnorm(length(lp)*k,mean = B[,lp,j],sd=des))
}
B[,3,]<-rnorm(1*k*sn,mean=B[,3,],sd=0.05)#agrega perturbacion al punto central (3)


#rotacion
alfa<-runif(sn,min = 0,max = pi/20)
AR<-sapply(alfa,simplify = "array",function(x){matrix(c(cos(x),-sin(x),sin(x),cos(x)),2,2)})
TAR<-array(0,c(k,ln,sn))
for(s in 1:sn){
  TAR[,,s]<-AR[,,s]%*%B[,,s]
}

#traslacion
TAR<-aaply(TAR,3,function(x){
  x+matrix(runif(2),k,ln)})
TTAR<-aperm(TAR,c(3,2,1))

#----------seccion NN --------------------


#armar base sacando distancia euclidiana del cero al punto central (15,1:2) y coordenadas

asim<- apply(B,3,function(x){sqrt(sum((x[,3])^2))})#calcula la asim
#asim<- apply(B,3,function(x){sqrt(sum((x[1,3])^2))})#calcula la asim

base<-matrix(TTAR,sn,ln*2,byrow=T)#arma una matriz con las coordenadas
mat<-cbind(base,asim)#esta matriz de dimensiones c(10000,35) tiene la base completa (train y test), con x (,1:34) e y(,35)

#----------------





# Split
train = mat[1:9000,]
test = mat[9001:10000,]


# Cre Max y Min
maxs <- apply(train[,1:34], 2, max)
mins <- apply(train[,1:34], 2, min)

# escalado y vuelto a la matriz mat
train[,1:34] <- scale(train[,1:34],center = TRUE, scale = maxs - mins)
test[,1:34] <- scale(test[,1:34],center = TRUE, scale = maxs - mins)

x_train <- train[,1:34]
y_train <- train[,35]
x_test <- test[,1:34]
y_test <- test[,35]


#---------KERAS-----------


library(keras)

#define el modelo
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 10, activation = 'relu', input_shape = c(34)) %>%
  layer_dense(units = 10, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'relu') %>%
  layer_dense(units = 1, activation = 'linear')
 


#compila el modelo
model %>% compile(
  loss = "mean_squared_error",
  optimizer=optimizer_adam(lr = 0.001),
  metrics = "mean_absolute_error"
)


#entrena el modelo
inicio<-Sys.time()
history <- model %>% fit( x_train, y_train,
  epochs = 50,
  batch_size = 15,
  validation_split = 0.1
)
fin<-Sys.time()
fin-inicio



pp<-predict(model,x_test)
cbind(y_test,pp)


#save_model_hdf5(model, filepath="/home/flotto/Documentos/redes/goldman/modelo92", overwrite = TRUE, include_optimizer = TRUE)
#load_model_hdf5(filepath="/home/flotto/Documentos/redes/goldman/modelo92", custom_objects = NULL, compile = TRUE)

