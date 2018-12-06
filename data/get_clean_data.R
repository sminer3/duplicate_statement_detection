#Purpose: Preprocess the data. Also works as an EDA on match file to understand basic patterns and identify red flags
#Program is meant to run at command line: Rscript [filename]
#Make sure the folder were Rscript.exe is part of path

#Libraries and data
library(readr)
set.seed(1)
dat = read_csv("../raw/final_matches.csv")
dat2 = read_csv("../raw/documentdb_matches.csv")

#Rstudio
#dat = read_csv("raw/final_matches.csv")
#dat2 = read_csv("raw/documentdb_matches.csv")


#Preprocessing before merge
dat2$idStudy = as.character(lapply(strsplit(as.character(dat2$idStudyStudyObject),'+',fixed=TRUE),function(x) x[1]))
dat2$idStudyObject = as.character(lapply(strsplit(as.character(dat2$idStudyStudyObject),'+',fixed=TRUE),function(x) x[2]))
dat2$n = rowSums(dat2[,c("respEqual","respUnequal","respNotSure")])
names(dat2)[names(dat2)=="questionName"] = "question"
order_columns = c("idStudy","idStudyObject","question","questionSimpleName","idStatement1","idStatement2","statement1","statement2","n","respEqual","respUnequal","respNotSure","resolution","matchProbability","firstSentiment","secondSentiment","firstStatementA","firstStatementI","firstStatementD","secondStatementA","secondStatementI","secondStatementD","aa","ai","ad","ia","ii","id","da","di","dd","uniqueScore1","uniqueScore2","gibberishScore1","gibberishScore2")
dat2 = dat2[,order_columns]
#Use rules instead of the default values, there are more equals this way

dat2$resolution = ifelse(dat2$respEqual/dat2$n>=6/7 & dat2$n>3,1,
                         ifelse(dat2$respUnequal+dat2$respNotSure>1,0,NA))

#Preprocess original data
dat$questionSimpleName = character(nrow(dat))
dat$respUnequal = NA
dat$respNotSure = NA
dat$matchProbability = NA
dat$firstSentiment = NA
dat$secondSentiment = NA
dat$firstStatementA = NA
dat$firstStatementI = NA
dat$firstStatementD = NA
dat$secondStatementA = NA
dat$secondStatementI = NA
dat$secondStatementD = NA
dat$uniqueScore1 = NA
dat$uniqueScore2 = NA
dat$gibberishScore1 = NA
dat$gibberishScore2 = NA
names(dat)[names(dat)=="comparison"] = "respEqual"
names(dat)[names(dat)=="match"] = "resolution"
dat = dat[,order_columns]

#Merge data
dat = rbind(dat,dat2)
dat$idStudy = tolower(dat$idStudy)
dat$idStudyObject = tolower(dat$idStudyObject)
dat$idStatement1 = tolower(dat$idStatement1)
dat$idStatement2 = tolower(dat$idStatement2)


#Remove duplicate rows ()
so_stmt1_stmt2 = paste(dat$idStudyObject,dat$idStatement1,dat$idStatement2)
paste("Number of duplicate pairs from original pulled data and documentdb data:",sum(duplicated(so_stmt1_stmt2)))
dat = dat[!duplicated(so_stmt1_stmt2),]

#Subset data by data that has enough observations to evaluate
unequals = dat$n-dat$respEqual
equals = dat$respEqual
old_criteria = unequals>1 | (equals>6 & unequals==0) | equals > 11 
new_criteria = unequals>1 | (equals>3 & unequals==0) | equals > 5
paste("Percent of observations that has enough data for new criteria (4/4, 6/7): ",round(100*mean(new_criteria,na.rm=T),1),"%",sep="")
paste("Percent of observations that has enough data for old criteria: (7/7, 12/13)",round(100*mean(old_criteria,na.rm=T),1),"%",sep="")

#Look at number of pairs per study object
dat = dat[new_criteria & !is.na(new_criteria),]
#dat = dat[old_criteria,]
cut = 500
so_table = as.numeric(table(dat$idStudyObject))
#hist(so_table) #histogram
paste("Numerical summary of number matched pairs per study object")
summary(so_table) #Numerical summary
paste("Percent of study objects with less than ",cut," pairs: ",round(mean(so_table<cut)*100,1),"%",sep="")
paste("Number of observations received from cutting at 500: ",sum(sapply(so_table,function(x) min(x,cut))),", which is ",round(100*sum(sapply(so_table,function(x) min(x,cut)))/sum(so_table),1),"% of the observations",sep="")

study_objects = table(dat$idStudyObject)
keep = names(study_objects[study_objects<cut])
to_sample = names(study_objects[study_objects>=cut])
sampled_dat = dat[dat$idStudyObject %in% keep,]
for (id in to_sample) {
  samp_to_add = dat[dat$idStudyObject==id,]
  sampled_dat<-rbind(sampled_dat,samp_to_add[sample(1:nrow(samp_to_add),cut),])
}
percent = sampled_dat$respEqual / sampled_dat$n
sampled_dat$resolution = ifelse(percent > .8499999, 1, 0)

#Study Objects that are focused on asking for a good name for a product, tv show, etc.
so_remove = c("7ef07d47-c17e-434d-836b-cfb268fd6e36", "4d30dba4-9b83-4823-986b-39a63f565852", "45cbfb53-b68c-4bfc-addf-fcb48c133e7d","401d4d5b-6eba-4d55-aada-261ec749e763")
paste(sum(sampled_dat$idStudyObject %in% so_remove), "pairs removed due to type of question (e.g. questions that ask about naming a new product or tv show, etc.")
sampled_dat = sampled_dat[!sampled_dat$idStudyObject %in% so_remove,]
#questions = unique(sampled_dat[,c("question","idStudyObject")])
#paste(questions$idStudyObject[grepl("name",questions$question)][c(4,11,16,19)],collapse = ", ")

#Look at duplicate statements
cut_stmt = 10
paste("Number of unique statements: ",length(unique(c(sampled_dat$statement1,sampled_dat$statement2))))
stmt_table = as.numeric(table(c(sampled_dat$statement1,sampled_dat$statement2)))
#hist(stmt_table)
paste("Numerical summary of the number of appearances for each statement")
summary(stmt_table)
paste("Percent of statements that appear less than ",cut_stmt," times: ",round(mean(stmt_table<cut_stmt)*100,1),"%",sep="")
paste("Percent of data that is observed to be a match: ", mean(dat$resolution))


sampled_dat$id_all = paste(sampled_dat$idStudy, sampled_dat$idStudyObject, sampled_dat$idStatement1, sampled_dat$idStatement2, sep = '+')
write_csv(sampled_dat,'train.csv')
#Next to do: remove statements that occur frequently?, Look at percent of words that appear in the embeddings, the number of statements that share words, etc.
