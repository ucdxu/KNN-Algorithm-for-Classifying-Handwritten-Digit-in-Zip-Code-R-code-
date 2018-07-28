#Appendix A: Citations
#https://stackoverflow.com/questions/11641701/sum-a-list-of-matrices?rq=1: We used this website to add up several matrices for question 7
#https://stats.stackexchange.com/questions/156036/confusion-matrix-and-roc-curves: We used this website to figure out how to use the built-in confusion matrix function in R.

#Appendix B: Code

#1
read_digits = function(i){
  data.matrix(read.table(i))}
train=read_digits("train.txt")
test=read_digits("test.txt")
## loading the data “test” and “train”

#2
view_digit = function(a,i){
  image(matrix(a[i,-1],nrow=16)[,seq(from=16,to=1)],col=grey.colors(256))
}
##  create the greyscale image. 
##  input: view_digit(train,2) (“a” is dataset, “i” is the row of the digit you want displayed)

#3
result=sapply(2:257,function(i){tapply(train[,i],train[,1],mean)})
## this step is creating the overlaying numbers
par(mfrow=c(2,5))
sapply(1:10,function(i){
  image(matrix(result[i,],nrow=16)[,seq(from=16,to=1)],col=grey.colors(256))
})
## This step computes 10 overlayed images in a 2 by 5 plot.

#3b
result2=sapply(2:257,function(i){
  k=summary(aov(train[,i]~factor(train[,1])))
  return(k[[1]][1,2]/k[[1]][2,2])
})
## This function is using ANOVA, it returns (SSR/SSE)
which.max(result2)
result2[230]
which(result2 > 1.2)
## This output shows the useful classifications. 
which.min(result2)
result2[80]
which(result2 < 0.02)
## This output shows the not useful classifications. 
boxplot(train[,230]~train[,1])
boxplot(train[,213]~train[,1])
boxplot(train[,80]~train[,1])
## First plot are the useful pixels, by comparing the second plot, the first plot has less residual.
test=function(p,t,dm){
  sapply(1:nrow(t),function(i){dist(rbind(p,t[i,]),method=dm)})
}

#4
library(proxy)
predict_knn=function(p,t,dm,k,tl){
  num=order(proxy::dist(t,t(as.matrix(p)),method=dm))[1:k]
  tab=table(tl[num])
  tab=tab[which(tab==max(tab))]
  ifelse(length(tab)==1,return(tab),return(tab[sample(1:length(tab),1)]))
}  
## this function contain 5 input: 1. p(prediction point), t(training point), dm(distance metric), k
## and label.
predict_kkn(train[1,-1],train[c(2:7291),-1],"euclidean",9,train[c(2:7291),1])
## This is the function’s output. 
## train[1,-1] selects the first row, and ignores the first column because it is not pixel.
## train[c(2:7291),-1] is the rest of the train set
## dm="euclidean"; (can switch the method by replacing "euclidean" with "maximum", 
## "manhattan", "canberra", "binary" or "minkowski".)
## 9 is k
## train[c(2:7291),1] is the label for the train data’s digits 

#5
library(caret)
dist_matrix_tr_eu =dist(train[,-1], train[,-1], method = "euclidean")
#to make this work for problem 6, you have to compute another dist matrix with a different 
#distance metric
cv_error_knn=function(dat,dm,k,dist_matrix){
  num=caret::createFolds(1:nrow(dat),k=10,list=T)
  sapply(1:10,function(i){
    pre=sapply(num[[i]],function(z){
      ord1 = order(dist_matrix[-z,z])[1:k]
      tab1 = table(dat[-z,1][ord1])
      tab3 = tab1[which(tab1 == max(tab1))]
      ifelse(length(tab3)==1,
             a<-tab3,
             a<-tab3[sample(1:length(tab3),1)])
      
      return(attr(a,"names"))})    
    
    return(1-sum(pre==dat[num[[i]],1])/length(num[[i]]))})
  ## 1- correction rate. = error rate
}
temp = cv_error_knn(train, "euclidean", 10, dist_matrix_tr_eu)
#6
library(doParallel)
library(foreach)

ER1 = foreach(x =1:15,.combine ="rbind") %dopar% cv_error_knn(train,"euclidean",x, 
                                                              dist_matrix_tr_eu)
dist_matrix_tr_man = dist(train[,-1], train[,-1], method = "manhattan")
#distance matrix using manhattan distance
ER2 = foreach(x =1:15,.combine ="rbind") %dopar% cv_error_knn(train,"manhattan",x, 
                                                              dist_matrix_tr_man)
# The first ER1 is using euclidean method, and this function is similar to "sapply"
# but it is more efficient than "sapply" because "sapply" only using one CPU.
# there are 300 output, so I need to calculate their mean.
ER11=apply(ER1,1,mean)
ER22=apply(ER2,1,mean)
## 

plot(1:15,ER11,xlab="k",ylab="Error Rate", main= "Error Rate Using Euclidean and Manhattan Distance", type = "l", ylim = c(0.02,0.06))
points(1:15,ER22,type="l", col= "red")
legend("topleft", legend=c("Manhattan","Euclidean"),
       col=c("red","black"), lty=1:1, cex=0.8, bg = "white")
## makes the plot with the two color-coordinated lines corresponding to the legend.

#7                                               
cv_error_knn2=function(dat,dm,k,dist_matrix){
  num=caret::createFolds(1:nrow(dat),k=10,list=T)
  
  sapply(1:10,function(i){
    pre=sapply(num[[i]],function(z){
      ord1 = order(dist_matrix[-z,z])[1:k]
      tab1 = table(dat[-z,1][ord1])
      tab3 = tab1[which(tab1 == max(tab1))]
      ifelse(length(tab3)==1,
             a<-tab3,
             a<-tab3[sample(1:length(tab3),1)])
      # number of predictions
      return(attr(a,"names"))})
    return(cbind(pre,dat[num[[i]],1]))
    ## returns two col matrix, first col is prediction, second col is actual
  })}  

#confusion matrix
con_matrix = function(dat, dm, k, dist_matrix){
  result = cv_error_knn2(dat, dm, k, dist_matrix) 
  #creates the matrix of prediction and actual values
  mat = sapply(1:10, function(i){tab = caret::confusionMatrix(result[[i]][,1], result[[i]][,2])})
  #creates a confusion matrix for each of the 10 folds
  mat_list = lapply(1:10, function(i){mat[,i]$table})
  #extracts 
  return(Reduce('+', mat_list))
}
eu_k = lapply(c(3,4,5), function(k){con_matrix(train, "euclidean", k, dist_matrix_tr_eu)})
man_k = lapply(c(3,4,5), function(k){con_matrix(train, "manhattan", k, dist_matrix_tr_man)})

#9
error_rate = function(dat, dm, k, dist_matrix){
  pre=sapply(1:2007,function(z){
    ord1 = order(dist_matrix[-z,z])[1:k]
    tab1 = table(dat[-z,1][ord1])
    tab3 = tab1[which(tab1 == max(tab1))]
    ifelse(length(tab3)==1,
           a<-tab3,
           a<-tab3[sample(1:length(tab3),1)])
    
    return(attr(a,"names"))})
  return(1-sum(pre==dat[,1])/2007)  
}
dist_matrix_test_eu = proxy::dist(test[,-1], test[,-1], method = "euclidean")
ERtest1 = foreach(x =1:15,.combine ="rbind") %dopar% error_rate(test,"euclidean",x, 
                                                                dist_matrix_test_eu)
dist_matrix_test_man = proxy::dist(test[,-1], test[,-1], method = "manhattan")
ERtest2 = foreach(x =1:15,.combine ="rbind") %dopar% error_rate(test,"manhattan",x, dist_matrix_test_man)
plot(1:15,ERtest1,xlab="k",ylab="error rate", type="l", ylim = c(0.02,0.06),
     main = "Error Rate of 10-Fold CV on Testing Data \n Using Euclidean and Manhattan distance")
points(1:15,ERtest2, type="l", col="red")
# Same idea as Q6, but with the testing data and no 10-fold CV.
