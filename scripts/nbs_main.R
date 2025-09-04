# Load packages -----------------------------------------------------------

require(NBR)

# Load data ---------------------------------------------------------------

demo.dat<-read.csv("/path/to/project/demo.csv")
demo.dat<-demo.dat[!is.na(demo.dat$Age),]
cmx<-list()
for(ii in c(1:nrow(demo.dat))){
  con_file <-sprintf("/path/to/project/data/mind-networks/%s/%s_aparc_atlas.csv", 
                     demo.dat$SUB_ID[ii], 
                     demo.dat$SUB_ID[ii])
  con_mat<-read.csv(con_file)
  con_mat<-as.matrix(con_mat[, c(2:ncol(con_mat))])
  cmx[[ii]]<-con_mat
}

cmx <- array(do.call(c, cmx), dim = c(68, 68, length(cmx)))

#  Run models with within each sex ----------------------------------------

mod = "~Cohort + Age + site + ICV"
thresholds<-c(0.001, 0.005, 0.01) # our different thresholds (p=0.005 is the main one)

## MALE
for(thrP in thresholds){
  
  set.seed(18900217) 
  before <- Sys.time()
  res.male<-nbr_lm(cmx[,,demo.dat$Gender=="M"], 
                   nnodes = 68, 
                   idata=demo.dat[demo.dat$Gender=="M",], 
                   mod = mod,        
                   nperm = 5000,
                   thrP = thrP,
                   thrT = NULL,
                   cores = 18,
                   verbose=TRUE
  )
  after <- Sys.time()
  file<-sprintf("~/Documentos/ace-autism/nbs/nbr_thrP%s_site-fixed_gender-male.RData",
                sub("0.", "", thrP)
                )
  save(res.male, file=file)
}


## FEMALE
for(thrP in c(0.001, 0.005, 0.01)){
  
  set.seed(18900217) 
  before <- Sys.time()
  res.female<-nbr_lm(cmx[,,demo.dat$Gender=="F"], 
                   nnodes = 68, 
                   idata=demo.dat[demo.dat$Gender=="F",], 
                   mod = mod,        
                   nperm = 5000,
                   thrP = thrP,
                   thrT = NULL,
                   cores = 18,
                   verbose=TRUE
  )
  after <- Sys.time()
  file<-sprintf("~/Documentos/ace-autism/nbs/nbr_thrP%s_site-fixed_gender-female.RData",
                sub("0.", "", thrP)
  )
  save(res.female, file=file)
}
