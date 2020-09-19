# Data file is saved as 'neuron_cluster_data.Rdata'
#
#---get data----
library(dplyr)
library(plyr)

ev.morphoData<-new.env()
load("morphology data.RData",envir = ev.morphoData)
obj.morphoAll<-ls(envir = ev.morphoData)
name.para<-get("Parameters",envir = ev.morphoData)
data.all<-get("DataAll",envir = ev.morphoData)
names(data.all)<-name.para
select_row<-function(tab,n) tab[n,]
select_col<-function(tab,n) tab[,n]
name.para.overall<-data.all %>% plyr::ldply(nrow) %>% filter(V1==27|V1==26) %>%
  select_col(1)
name.para.select<-name.para.overall[c(1,5:11,22)] # 14th can be considered
data.select<-data.all[name.para.select]
remove_row<-function(tab){
  if(nrow(tab)==27) return(slice(tab,-25))
  else return(tab)
}
data.select.noHead<-data.select %>% plyr::llply(select,c(2)) %>% 
  plyr::llply(remove_row) %>% bind_cols()
data.select.cat<-cbind(data.select[[1]] %>% select(c(1,8)) %>% select_row(-25),
                       data.select.noHead)
row.names(data.select.noHead)<-paste("neuron",1:26)

#----principal components analysis----
pr.morpho<-princomp(data.select.noHead,cor = T)
summary(pr.morpho,loadings = T)

#----linear model----
data.select.cat$treat<-as.factor(data.select.cat$treat)
glm.both<-glm(treat~.,binomial(),data.select.cat[-1])

cor.morpho<-cor(data.select.noHead)
k.morpho<-kappa(cor.morpho)
eig.morpho<-eigen(cor.morpho) # exist multicollinearity in 6th,7th,8th variant

cor2.morpho<-cor(data.select.noHead[,c(1:6,9)])
k2.morpho<-kappa(cor2.morpho,exact = T)
eig2.morpho<-eigen(cor2.morpho) #length,area,volume exist multicollinearity

cor3.morpho<-cor(data.select.noHead[,2:6])
k3.morpho<-kappa(cor3.morpho) # no multicollinearity

#----cluster analysis----

library("NbClust", lib.loc="~/R/win-library/3.5")
library("mvoutlier", lib.loc="~/R/win-library/3.5")

nc<-NbClust(data.scaled,method = "average")
data.scaled<-scale(data.select.noHead)
d<-dist(data.scaled)
hc<-hclust(d,method = "average")
clusters<-cutree(hc,k=3)
plot(hc);rect.hclust(hc,k=3)
data.clustered<-cbind(data.select.noHead,clusters)
data.clustered<-rename(data.clustered,cluster_average=clusters)

nc2<-NbClust(data.scaled,method = "centroid") #best is 4
hc2<-hclust(d,method = "centroid")
data.clustered$cluster_centroid<-cutree(hc2,k=4)
plot(hc2);rect.hclust(hc2,k=4)

nc3<-NbClust(data.scaled,method="single") #best is 5
hc3<-hclust(d,method="single")
data.clustered$cluster_single<-cutree(hc3,k=5)
plot(hc3);rect.hclust(hc3,k=5)

nc4<-NbClust(data.scaled,method="complete") #best is 2
hc4<-hclust(d,method="complete")
data.clustered$cluster_complete<-cutree(hc4,k=2)
plot(hc4);rect.hclust(hc4,k=2)

nc5<-NbClust(data.scaled,method="ward.D") #best is 2
hc5<-hclust(d,method="ward.D")
data.clustered$cluster_ward<-cutree(hc5,k=2)
plot(hc5);rect.hclust(hc5,k=2)

# find outliers
col.select<-c(3:8) %>% plyr::alply(1,combn,x=1:9,simplify=F)  %>% do.call(what=c)
proto.rm<-col.select %>% plyr::laply(is.element,el = 6:8) %>% plyr::aaply(1,all)
proto.select<-col.select[which(proto.rm==F)] #generate combinations without 6 to 8
names(proto.select)<-paste("protocol",seq_len(length(proto.select)))
select_list<-function(ls,ind) return(ls[[ind]])
outliers<-proto.select %>% plyr::llply(select_col,tab=data.scaled) %>%
  plyr::llply(aq.plot) %>% plyr::ldply(select_list,ind='outliers')
outliers.min<-min(rowSums(as.matrix(outliers[-1])))
proto.outmin<-which(rowSums(as.matrix(outliers[-1]))==outliers.min)
proto.id.best<-max(proto.outmin) # the protocol involved with more variables
proto.best<-select_list(proto.select,paste("protocol",proto.id.best))

# re-clustering
methds<-c("single","complete","average","centroid","ward.D")
names(methds)<-methds
dst<-dist(data.scaled[,proto.best])
hclusters<-list(all=plyr::alply(methds,1,hclust,d=dst,.dims = T))

#----clustering with different group:GFP pos/neg-----

# find multicollinearity
cr<-list(pos=cor(data.select.noHead[1:16,]),
          neg=cor(data.select.noHead[17:26,]))
kp<-plyr::laply(cr,kappa)
eig<-plyr::llply(cr,eigen)

# find outliers
out.pos<-proto.select %>% plyr::llply(select_col,tab=data.scaled[1:16,]) %>%
  plyr::llply(aq.plot) %>% plyr::ldply(select_list,ind='outliers')
outliers.pos.min<-min(rowSums(as.matrix(out.pos[-1])))
proto.pos.outmin<-which(rowSums(as.matrix(out.pos[-1]))==outliers.pos.min)
proto.pos.idbest<-max(proto.pos.outmin)
proto.pos.best<-select_list(proto.select,paste("protocol",proto.pos.idbest))

aq.plot.catch<-function(x){
  tryCatch(
    aq.plot(x)$'outliers',
    error=function(e) return(array(NA,nrow(x),list(row.names(x))))
  )
} # catch each error and return NA as result
out.neg<-proto.select %>% plyr::llply(select_col,tab=data.scaled[17:26,]) %>%
  plyr::ldply(aq.plot.catch) %>% na.omit()
outliers.neg.min<-min(rowSums(as.matrix(out.neg[-1])))
proto.neg.outmin<-which(rowSums(as.matrix(out.neg[-1]))==outliers.neg.min)
proto.neg.idbest<-max(proto.neg.outmin)
proto.neg.best<-select_list(proto.select,out.neg[proto.neg.idbest,1])

# clustering GFP positive/negtive

library("ggdendro")
library("ggplot2")
library("rpart")
require("magrittr")

data.pos.cluster<-data.select.noHead[1:16,proto.pos.best]
data.neg.cluster<-data.select.noHead[17:26,proto.neg.best]
dst.pos<-dist(scale(data.pos.cluster))
dst.neg<-dist(scale(data.neg.cluster))
hclusters$pos<-plyr::alply(methds,1,hclust,d=dst.pos,.dims = T)
hclusters$neg<-plyr::alply(methds,1,hclust,d=dst.neg,.dims = T)
data.pos.cluster$cluster_complete<-cutree(hclusters$pos$complete,k = 2)
data.neg.cluster$cluster_complete<-cutree(hclusters$neg$complete,k=2)
data.pos.cluster<-within(data.pos.cluster,cluster_complete<-as.factor(cluster_complete))
data.neg.cluster<-within(data.neg.cluster,cluster_complete<-as.factor(cluster_complete))

data.dendr<-llply(hclusters,with,dendro_data(complete)) # coerce model to dendrogram data
data.dendr$pos$labels<-data.pos.cluster %>% mutate(label=rownames(data.pos.cluster)) %>% 
  select_at(c("label","cluster_complete")) %>% 
  right_join(data.dendr$pos$labels,by = "label")
data.dendr$neg$labels<-data.neg.cluster %>% mutate(label=rownames(data.neg.cluster)) %>% 
  select_at(c("label","cluster_complete")) %>% 
  right_join(data.dendr$neg$labels,by = "label")

selected_color<-RColorBrewer::brewer.pal(9,"Set1")[c(4,5,7,8)]
change_legend_key_text<-function(fig,txt){
  fig_grobs<-ggplotGrob(fig)
  fig_grobs$grobs[[15]]$grobs[[1]]$grobs[[4]]$label<-txt
  fig_grobs$grobs[[15]]$grobs[[1]]$grobs[[6]]$label<-txt
  ggplotify::as.ggplot(fig_grobs)
}
fig.pos.clust<-ggplot(mapping=aes(x=x,y=y))+
  geom_segment(aes(xend=xend,yend=yend),segment(data.dendr$pos))+
  geom_text(aes(label=label,color=cluster_complete),label(data.dendr$pos),hjust=0)+
  coord_flip()+
  scale_y_reverse(name=NULL,labels=NULL,breaks=NULL,expand=c(0.2,0))+
  scale_x_continuous(name=NULL,labels=NULL,breaks=NULL)+
  scale_color_manual(values = selected_color[1:2],labels=c("Cluster 1","Cluster 2"),
                     guide=guide_legend(title="GFP+ clusters"))
fig.pos.clust %<>% change_legend_key_text("id")
fig.neg.clust<-ggplot(mapping=aes(x=x,y=y))+
  geom_segment(aes(xend=xend,yend=yend),segment(data.dendr$neg))+
  geom_text(aes(label=label,color=cluster_complete),label(data.dendr$neg),hjust=0)+
  coord_flip()+
  scale_y_reverse(name=NULL,labels=NULL,breaks=NULL,expand=c(0.2,0))+
  scale_x_continuous(name=NULL,labels=NULL,breaks=NULL)+
  scale_color_manual(values = selected_color[3:4],labels=c("Cluster 1","Cluster 2"),
                     name="GFP- clusters")
fig.neg.clust %<>% change_legend_key_text("id")
export_images<-function(file_name,fig,w,h,dpi=100,scale=3){
  fig_format<-c("png","jpg","pdf")
  fig_file_name<-paste(file_name,fig_format,sep = '.')
  a_ply(fig_file_name,1,ggsave,fig,width=w,height=h,units="mm",dpi=dpi,scale=scale)
}
export_images("GFP+ cluster tree",fig.pos.clust,50,32,100,3)
export_images("GFP- cluster tree",fig.neg.clust,50,20,100,3)


glm.pos<-glm(cluster_complete~.,data = data.pos.cluster,family = binomial())
glm.neg<-glm(cluster_complete~.,data = data.neg.cluster,family = binomial())
sqrt(vif(glm.pos))>2 # last 3 variables exists multicollinearity
sqrt(vif(glm.neg))>2
eig.neg<-eigen(cor(data.neg.cluster[-8]))

library(leaps)
leap.pos<-regsubsets(cluster_complete~.,data = data.pos.cluster) 
plot(leap.pos) # all-subset regression to choose variables
glm.pos2<-glm(
  cluster_complete~`Filament Distance from Origin`+`Filament Length (sum)`+
    `Filament No. Dendrite Terminal Pts`,
  data = data.pos.cluster,family = binomial()
)

dtree<-list(
  pos=rpart(cluster_complete~.,data=data.pos.cluster,method = "class",parms = list(split="information")),
  neg=rpart(cluster_complete~.,data=data.neg.cluster,method = "class",parms = list(split="information"))
)

# anova and visulization
data.pos.fig<-mutate(data.pos.cluster,id=1:16)
data.neg.fig<-mutate(data.neg.cluster,id=17:26)

para.pos<-names(data.pos.cluster)[1:5]
lm.pos<-paste0("`",para.pos,"`~cluster_complete") %>% as.quoted() %>%
  llply(lm,data=data.pos.cluster)
aov.pos<-llply(lm.pos,anova)
p_to_signif<-function(x){
  signifi<-cut(x,breaks = c(0,0.001,0.01,0.05,1),labels = c('***','**','*',''))
  return(levels(signifi)[signifi])
}
pval.aov.pos<-aov.pos %>% ldply(with,data.frame(p=`Pr(>F)`[1])) %>%
  dplyr::mutate(para=para.pos,significant=p_to_signif(p))

para.neg<-names(data.neg.cluster)[1:ncol(data.neg.cluster)-1]
lm.neg<-paste0("`",para.neg,"`~cluster_complete") %>% as.quoted() %>%
  llply(lm,data=data.neg.cluster)
aov.neg<-llply(lm.neg,anova)
pval.aov.neg<-aov.neg %>% ldply(with,data.frame(p=`Pr(>F)`[1])) %>%
  dplyr::mutate(para=para.neg,significant=p_to_signif(p))

fig.pos.dot<-ggplot(data.pos.fig,
  aes(x=`Filament Full Branch Depth`,
      y=`Filament Length (sum)`))+
  geom_point(aes(color=cluster_complete,shape=cluster_complete))+
  geom_text(aes(label=id),nudge_x=0.1)+
  scale_color_discrete(name="cluster")+
  scale_shape_discrete(name="cluster")+
  scale_x_continuous(name="# Branch Depth")+
  scale_y_continuous(name=expression(paste("Length(",mu,"m)")))+
  theme_classic()
fig.neg.dot<-ggplot(data.neg.fig,
  aes(x=`Filament No. Dendrite Terminal Pts`,
      y=`Filament Length (sum)`))+
  geom_point(aes(shape=cluster_complete,color=`Filament Area (sum)`,
                 size=`Filament Volume (sum)`))+
  geom_text(aes(label=id),nudge_x = 1)+
  scale_color_continuous(name=expression(paste("Area(",mu,m^2,")")),
                         low='green',high = 'red')+
  scale_x_continuous(name="# Dendrite Terminal Points")+
  scale_y_continuous(name=expression(paste("Sum of Length(",mu,"m)")))+
  scale_size_continuous(name=expression(paste("Volume(",mu,m^3,")")))+
  scale_shape_discrete(name="Cluster")+
  theme_classic()
