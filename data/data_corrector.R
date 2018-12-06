library(readr)

train = read_csv("data/train.csv")
corrections = read_csv("data/corrections.csv")

print(paste(round(nrow(corrections) / nrow(train),4)*100,"% of pairs have been looked at now.",sep=""))
print(paste(round(sum(corrections$new_label!=corrections$resolution) / nrow(train),4)*100,"% of pairs in the training set have changed classes"))                    
train$percent = train$respEqual / train$n
#train$id_all = paste(train$idStudy, train$idStudyObject, train$idStatement1, train$idStatement2, sep='+')
#cat("Minimum percent under consideration (no decimals): ")
#low = as.numeric(readLines(con = stdin(),1))/100
low = .50
#cat("Maximum percent under consideration (no decimals): ")
#high = as.numeric(readLines(con = stdin(),1))/100
high = .85
sub_train = train[train$percent<high & train$percent>low  & !train$id_all %in% corrections$id_all,]
sub_train = sub_train[order(sub_train$percent,decreasing = T),]
i=0
go = T
while(go) {
  i = i + 1
  print(paste("Question:",sub_train$question[i]))
  print(paste("Statement1:",sub_train$statement1[i]))
  print(paste("Statement2:",sub_train$statement2[i]))
  res = as.numeric(readline("Match? (yes=1, no=2, not sure = 3, quit = 4): "))
  if(res == 4) {
    go = F
    print(paste(round(nrow(corrections) / nrow(train),4)*100,"% of pairs have been looked at now.",sep=""))
    print(paste(round(sum(corrections$new_label!=corrections$resolution) / nrow(train),4)*100,"% of pairs in the training set have changed classes"))
  } else {
    to_add = sub_train[i,c("id_all","question","statement1","statement2","resolution")]
    if(res == 3) {
      to_add$new_label = to_add$resolution
    } else {
      to_add$new_label = res
    }
    corrections = rbind(corrections, to_add)
  }
  if((i+1) > nrow(sub_train)) {
    go = F
    print(paste(round(nrow(corrections) / nrow(train),4)*100,"% of pairs have been looked at now.",sep=""))
    print(paste(round(sum(corrections$new_label!=corrections$resolution) / nrow(train),4)*100,"% of pairs in the training set have changed classes"))
  }
}
write_csv(corrections,"data/corrections.csv")
for(i in 1:nrow(train)) {
  if(train$id_all[i] %in% corrections$id_all) train$id_all[i] == corrections$new_label[corrections$id_all ==train$id_all[i]] 
}
write_csv(train,"data/corrected_train.csv")
