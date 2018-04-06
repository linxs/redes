library(plyr)
#v<-c(0,2,0,1,0,0,0,-1,0,-2,   -1.5,1, -1,1, -1.5,0, -1,0, -1.5,-1, -1,-1, 1,1, 1.5,1,1,0, 1.5,0, 1,-1, 1.5,-1.0   )
w<-c(0,2.1,0,1,0,0,0,-1,0,-2,   -1.5,1, -1,1, -1.6,0, -1,0, -1.5,-1, -1,-1, 1,1.1, 1.5,1,1,0, 1.5,0, 1,-1, 1.5,-1.0   )
#x<-c(0,2,0,1,0.15,0,0,-1,0,-2,   -1.5,.97, -1,1, -1.5,0, -1,0, -1.5,-1, -1,-1, .86,1, 1.5,1,1,0, 1.5,0, 1,-1, 1.5,-1.0   )
#y<-c(0,2,0,1,0,0,0.08,-1,0,-2,   -1.5,1, -1,1, -1.39,0, -1,0, -1.5,-1, -1,-1, 1,1, 1.54,1,1,0, 1.5,0, 1,-1, 1.5,-1.0   )
#z<-c(0,2,0,1,0,0.13,0,-1,0,-2,   -1.5,1, -1,1, -1.57,0, -1,0, -1.5,-1, -1.1,-1, 1,1, 1.5,1,1,0, 1.5,0, 1.3,-1, 1.5,-1.0   )
#vp<-c(v,w,x,y,z)


k<-2#dimensiones
sn<-10000#numero de especimenes
ln<-17#numero de landmarks
n<-4
des<-0.05

B<-array(w,c(k,ln,sn))

O<-B

for(j in 1:sn){
  lp<-sample(c(1:17),n,replace = FALSE)
  B[,lp,j]<-(rnorm(length(lp)*k,mean = B[,lp,j],sd=des))
}

E<-B-O#saca el error por coord de cada landmark


#Aca guardar los originales

#

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

lx<-11
izq<-c(1:11)
der<-c(1:5,12:17)
basei<-matrix(0,sn,((lx*lx)-lx)/2,byrow=T)#arma una matriz donde quepan las distancias
based<-basei
for(i in 1:sn){
  basei[i,]<-c(dist(TTAR[izq,,i]))
  based[i,]<-c(dist(TTAR[der,,i]))
}

basedif<-basei-based

EP<-aperm(E,c(2,1,3))
orig<-matrix(EP,sn,ln*2,byrow=T)#errores por coord


mat<-cbind(basedif,orig)#esta matriz de dimensiones c(10000,55+34) tiene la base completa (train y test), con x (,1:55) e y(,56:89)

#----------------


# Split
train = mat[1:9000,]
test = mat[9001:10000,]


# Cre Max y Min
maxs <- apply(train[,1:55], 2, max)
mins <- apply(train[,1:55], 2, min)

# escalado y vuelto a la matriz mat
train[,1:55] <- scale(train[,1:55],center = TRUE, scale = maxs - mins)
test[,1:55] <- scale(test[,1:55],center = TRUE, scale = maxs - mins)

train[is.na(train)]<-0
test[is.na(test)]<-0



x_train <- train[,1:55]
y_train <- train[,56:89]
x_test <- test[,1:55]
y_test <- test[,56:89]


#---------KERAS-----------


library(keras)

#define el modelo
model <- keras_model_sequential()
model %>% 
  layer_dense(units = 110, activation = 'relu', input_shape = c(55)) %>%
  layer_dense(units = 110, activation = 'relu') %>%
  layer_dense(units = 110, activation = 'relu') %>%
  layer_dense(units = 34, activation = 'linear')



#compila el modelo
model %>% compile(
  loss = "mean_squared_error",
  optimizer=optimizer_adam(lr = 0.0001),
  metrics = "mean_squared_error"
)


#entrena el modelo
inicio<-Sys.time()
history <- model %>% fit( x_train, y_train,
                          epochs = 50,
                          batch_size = 10,
                          validation_split = 0.1)


pp<-predict(model,x_test)
rbind(pp[1,],y_test[1,])



#for(i in 1:30){
#  plot(t(O[,,i])+matrix(pp[i,],ncol = 2),col="red")
#  points(t(B[,,9000+i]))
#  readline(prompt="Press [enter] to continue")
#}
