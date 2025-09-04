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

# Test normative sex differences (only within TDC group; main analysis setup) ----------------------------------------
mod <- "~Gender + Age + ICV + site"

set.seed(18900217) 
before <- Sys.time()
res.only.tdc<-nbr_lm(cmx[,,demo.dat$Cohort=="CON"], 
                 nnodes = 68, 
                 idata=demo.dat[demo.dat$Cohort=="CON",], 
                 mod = mod,        
                 nperm = 5000,
                 thrP = 0.005,
                 thrT=NULL,
                 cores = 20,
                 verbose=TRUE
)

file<-sprintf("~/Documentos/ace-autism/nbs/nbr_thrP005_site-fixed_genders_tdc.RData",
                sub("0.", "", thrP)
                )
save(res.only.tdc, file=file)
