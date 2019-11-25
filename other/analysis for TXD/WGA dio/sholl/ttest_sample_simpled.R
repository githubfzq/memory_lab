# delete one sample that is too simple

data_to_plot2<-data_to_plot[-13,]
data_melt2 <-
  melt(data_to_plot2, id.vars = 1, variable.name = "parameter")
#--------ks test----------
ksrslt<-list()
for(p in parameters_to_analysis) {
  ksrslt[[p]] <- ks.test(with(data_melt, value[type == 'GFP' &
                                                 parameter == p]),
                         with(data_melt, value[type == 'GFP_near' & parameter == p]))
}
ksp <- data.frame()
for (p in parameters_to_analysis) {
  ksp <- rbind(ksp,
               data.frame(parameter = p, pval = ksrslt[[p]]$p.value))
}
ksp<-data.frame(ksp,sample="full")
for(p in parameters_to_analysis) {
  ksrslt[[p]] <- ks.test(with(data_melt2, value[type == 'GFP' &
                                                  parameter == p]),
                         with(data_melt2, value[type == 'GFP_near' & parameter == p]))
}
for (p in parameters_to_analysis) {
ksp <- rbind(ksp,
data.frame(parameter = p, pval = ksrslt[[p]]$p.value,sample="simple"))
}
#-------norm test---------
normp<-data.frame()
for(t in c('GFP','GFP_near')){
  for(p in parameters_to_analysis){
    testdt<-with(data_melt,value[type==t&parameter==p])
    shprslt<-shapiro.test(testdt)
    normp<-rbind(normp,
                 data.frame(type=t,parameter=p,
                            pvalue=shprslt$p.value,testMethod=shprslt$method))
  }
}
#-----------f_test----------
ftest<-data.frame()
for(p in parameters_to_analysis){
  varrslt<-var.test(
    with(data_melt,value[type=='GFP'&parameter==p]),
    with(data_melt,value[type=='GFP_near'&parameter==p])
  )
  ftest<-rbind(ftest,
               data.frame(parameter=p,pvalue=varrslt$p.value,
                          testMethod=varrslt$method,sample="full")
  )
  varrslt<-var.test(
    with(data_melt2,value[type=='GFP'&parameter==p]),
    with(data_melt2,value[type=='GFP_near'&parameter==p])
  )
  ftest<-rbind(ftest,
               data.frame(parameter=p,pvalue=varrslt$p.value,
                          testMethod=varrslt$method,sample="simple"))
}
#-----------t_test---------
ttest<-data.frame()
for(p in with(normp,parameter[pvalue>0.01])){
  trslt<-t.test(
    with(data_melt,value[parameter==p&type=='GFP']),
    with(data_melt,value[parameter==p&type=='GFP_near'])
  )
  ttest<-rbind(ttest,
               data.frame(parameter=p,pvalue=trslt$p.value,
                          testMethod=trslt$method,sample="full"))
  trslt<-t.test(
    with(data_melt2,value[parameter==p&type=='GFP']),
    with(data_melt2,value[parameter==p&type=='GFP_near'])
  )
  ttest<-rbind(ttest,
               data.frame(parameter=p,pvalue=trslt$p.value,
                          testMethod=trslt$method,sample='simple'))
}
para_vareq_1<-intersect(
  with(normp,parameter[pvalue>0.01]),
  with(ftest,parameter[pvalue>0.01&sample=="full"])
)
para_vareq_2<-intersect(
  with(normp,parameter[pvalue>0.01]),
  with(ftest,parameter[pvalue>0.01&sample=="simple"])
)
for(p in para_vareq_1){
  trslt<-t.test(
    with(data_melt,value[parameter==p&type=='GFP']),
    with(data_melt,value[parameter==p&type=='GFP_near']),
    var.equal = TRUE
  )
  ttest<-rbind(ttest,
               data.frame(parameter=p,pvalue=trslt$p.value,
                          testMethod=trslt$method,sample="full"))
  trslt<-t.test(
    with(data_melt,value[parameter==p&type=='GFP']),
    with(data_melt,value[parameter==p&type=='GFP_near']),
    var.equal = TRUE
  )
  ttest<-rbind(ttest,
               data.frame(parameter=p,pvalue=trslt$p.value,
                          testMethod=trslt$method,sample='simple'))
}
for(p in with(normp,parameter[pvalue<=0.01])){
  wilrslt<-wilcox.test(
    with(data_melt,value[parameter==p&type=='GFP']),
    with(data_melt,value[parameter==p&type=='GFP_near'])
  )
  ttest<-rbind(ttest,
               data.frame(parameter=p,pvalue=wilrslt$p.value,
                          testMethod=wilrslt$method,sample="full"))
  wilrslt<-wilcox.test(
    with(data_melt2,value[parameter==p&type=='GFP']),
    with(data_melt2,value[parameter==p&type=='GFP_near'])
  )
  ttest<-rbind(ttest,
               data.frame(parameter=p,pvalue=wilrslt$p.value,
                          testMethod=wilrslt$method,sample="simple"))
}
#------visualization-------
data_vis1 <- subset(data_melt,
                   subset = is.element(parameter, unique(with(ttest, parameter[pvalue <
                                                                                 0.01 & sample == 'full']))))
data_vis2 <- subset(data_melt2,
                   subset = is.element(parameter, unique(with(ttest, parameter[pvalue <
                                                                                 0.01 & sample == 'simple']))))
data_vis<-rbind(data.frame(data_vis1,sample='full'),
                data.frame(data_vis2,sample='simple'))
