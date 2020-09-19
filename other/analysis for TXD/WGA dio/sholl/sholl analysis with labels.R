# ----read data-----
library(dplyr)
library(plyr)
gfp.data=read.csv('Sholl of GFP+.csv')
gfpNear.data=read.csv('Sholl of GFP-.csv')
sholl.data=rbind(
  mutate(gfp.data,group='GFP+'),mutate(gfpNear.data,group='GFP-'))
gfp.normalized.data=read.csv('Normalized Sholl of GFP+.csv')
gfpNear.normalized.data=read.csv('Normalized Sholl of GFP-.csv')
sholl.normalized.data=rbind(
  mutate(gfp.normalized.data,group='GFP+'),
  mutate(gfpNear.normalized.data,group='GFP-'))
#----data process-----
library(tidyr)

sholl.data.fill<-sholl.data %>% spread(radius,intersections,fill=0) %>%
  gather("radius","intersections",4:775) %>%
  dplyr::mutate(radius=as.integer(radius))
sholl.data.fill<-within(sholl.data.fill,intersections[radius==0]<-1)
data.normalized.fill<-sholl.normalized.data %>% dplyr::select(2:6) %>%
  spread(radius_normalized,intersections,fill=0) %>%
  gather("radius_normalized","intersections",4:104) %>%
  dplyr::mutate(radius_normalized=as.numeric(radius_normalized))
data.normalized.fill.filtered<-data.normalized.fill %>%
  dplyr::filter(as.character(radius_normalized) %in% as.character(seq(0,1,0.05)))
data.cut<-sholl.data %>% 
  dplyr::filter(radius %in% seq(0,max(radius),30))
data.cut.fill<-sholl.data.fill %>%
  dplyr::filter(radius %in% seq(0,max(radius),30)) %>%
  dplyr::mutate(radius=as.factor(radius),group=as.factor(group))
tally.tb<-with(data.cut,table(radius,group))
radius.selected<-as.integer(names(which(tally.tb[,1]>0)))
data.cut<-sholl.data %>% dplyr::filter(radius %in% radius.selected) %>%
  mutate(radius=as.factor(radius),group=as.factor(group))
data.label<-dlply(data.cut.fill,.(label))
data.label.cum<-llply(data.label,ddply,.(group,id),
                      mutate,cum_inters=cumsum(intersections))

data.noLabel.fill<-data.cut.fill %>% dplyr::group_by(radius,id,group) %>%
  dplyr::summarise(intersections=sum(intersections))
data.noLabel.fill<-within(data.noLabel.fill,intersections[radius==0]<-1)
data.noLabel<-data.cut %>% group_by(radius,id,group) %>% 
  dplyr::summarise(intersections=sum(intersections))
data.noLabel<-within(data.noLabel,intersections[radius==0]<-1)
data.noLabel.cum<-data.noLabel %>% dplyr::group_by(group,id) %>%
  dplyr::mutate(cum_inters=cumsum(intersections))
percent.selected<-seq(0,1,0.05)


data.normalized.filtered<-sholl.normalized.data %>%
  filter(as.character(radius_normalized) %in% as.character(percent.selected)) %>%
  dplyr::mutate(radius_normalized=as.factor(radius_normalized),
                group=as.factor(group))
data.normalized.label<-dlply(data.normalized.filtered,.(label))
data.normalized.noLabel<-data.normalized.filtered %>%
  group_by(group,id,radius_normalized) %>% 
  dplyr::summarise(intersections=sum(intersections))
data.normalized.noLabel<-within(
  data.normalized.noLabel,intersections[radius_normalized==0]<-1)
data.normalized.unrepeat<-data.normalized.filtered %>% 
  group_by(label,radius_normalized,group) %>% 
  dplyr::summarise(n=length(unique(intersections))) %>% 
  dplyr::filter(n==1) %>% 
  anti_join(x=data.normalized.filtered,
            by = c("label","radius_normalized","group")) # avoid all values identical
data.normalized.nolabel.unrepeat<-data.normalized.noLabel %>% 
  group_by(radius_normalized,group) %>% 
  dplyr::summarise(n=length(unique(intersections))) %>% 
  dplyr::filter(n==1) %>% 
  anti_join(x=data.normalized.noLabel,
            by = c("radius_normalized","group"))
data.normalized.label.cum<-llply(data.normalized.label,ddply,.(group,id),
                                 mutate,cum_inters=cumsum(intersections))
data.normalized.nolabel.cum<-ddply(data.normalized.noLabel,.(group,id),
                                   mutate,cum_inters=cumsum(intersections))
#----statistics----
library(WRS2)
library(coin)
library(methods)
# library(lmPerm)
# aovpfit<-aovp(intersections~radius*group,data.noLabel,perm='Prob',seqs=T)
# aovp.label<-llply(
#   data.label,aovp,formula=intersections~radius*group,
#   perm='Prob',seqs=T)

# origin data sholl test
# MANOVA
# library(MVN)
# 
# data.noLabel.spread<-spread(data.noLabel,radius,intersections,fill = 0)
# data.noLabel.manovaY<-as.matrix(data.noLabel.spread[,c(-1,-2,-3)])
# data.noLabel.manovaG<-data.noLabel.spread$group
# manova.noLabel<-manova(data.noLabel.manovaY~data.noLabel.manovaG)
# manova.qq<-qqplot(
#   qchisq(ppoints(nrow(data.noLabel.manovaY)),df=ncol(data.noLabel.manovaY)),
#   mahalanobis(data.noLabel.manovaY,colMeans(data.noLabel.manovaY),
#               cov(data.noLabel.manovaY)))
# multnorm.noLabel<-mvn(data.noLabel.manovaY,mvnTest = 'mardia',
#                       multivariatePlot = 'qq')

# analysis of apical+basal with normalized radius
normtest.normalized<-dlply(
  data.normalized.unrepeat,.(label,radius_normalized,group),
  with,shapiro.test(intersections))
normtest.normalized.sum<-ldply(normtest.normalized,with,
                               data.frame(pVal=p.value,method=method))
with(normtest.normalized.sum,all(pVal>0.05)) #false

normtest.normalized.nolabel<-dlply(
  data.normalized.nolabel.unrepeat,.(group,radius_normalized),
  with,shapiro.test(intersections)) 
normtest.normalized.nolabel.sum<-ldply(
  normtest.normalized.nolabel,with,
  data.frame(pVal=p.value,method=method))
with(normtest.normalized.nolabel.sum,all(pVal>0.05)) #false
with(normtest.normalized.nolabel.sum,all(pVal>0.01))

data.normalized.noLabel.tTest<-normtest.normalized.nolabel.sum %>%
  dplyr::filter(pVal>0.05) %>%
  dplyr::group_by(radius_normalized) %>%
  dplyr::summarise(typeG=n_distinct(group)) %>%
  dplyr::filter(typeG==2) %>%
  dplyr::select(1) %>%
  semi_join(x=data.normalized.noLabel,by=c("radius_normalized"))
var.normalized.nolabel<-dlply(
  data.normalized.noLabel.tTest,.(radius_normalized),
  with,var.test(intersections~group))
var.normalized.nolabel.sum<-ldply(
  var.normalized.nolabel,with,data.frame(pVal=p.value))
data.normalized.noLabel.tTest.varIsEq<-var.normalized.nolabel.sum %>%
  dplyr::filter(pVal>0.05) %>% dplyr::select(1) %>%
  semi_join(x=data.normalized.noLabel.tTest,by=c("radius_normalized"))
data.normalized.noLabel.tTest.varNotEq<-setdiff(
  data.normalized.noLabel.tTest,data.normalized.noLabel.tTest.varIsEq)
data.normalized.noLabel.wilcox<-setdiff(
  data.normalized.noLabel,data.normalized.noLabel.tTest)
method_p<-function(obj){
  m<-slot(obj,"method")
  p<-pvalue(obj)
  return(data.frame(method=m,pVal=p))
}
wilcox.normalized.nolabel.sum<-data.normalized.noLabel.wilcox %>%
  dlply(.(radius_normalized),with,wilcox_test(intersections~group)) %>%
  ldply(method_p)
tTest.normalized.nolabel.varIsEq.sum<-data.normalized.noLabel.tTest.varIsEq %>%
  dlply(.(radius_normalized),with,t.test(intersections~group,var.equal=TRUE)) %>%
  ldply(with,data.frame(method=method,pVal=p.value))
tTest.normalized.nolabel.varNotEq.sum<-data.normalized.noLabel.tTest.varNotEq %>%
  dlply(.(radius_normalized),with,t.test(intersections~group)) %>%
  ldply(with,data.frame(method=method,pVal=p.value))
result.normalized.noLabel<-rbind(
  tTest.normalized.nolabel.varIsEq.sum,tTest.normalized.nolabel.varNotEq.sum,
  wilcox.normalized.nolabel.sum)
# analysis of apical+basal
data.noLabel.unrepeat<-data.noLabel %>% dplyr::group_by(radius,group) %>%
  dplyr::summarise(n=n_distinct(intersections),N=length(intersections)) %>%
  dplyr::filter(n==1|N<=3) %>%
  anti_join(x=data.noLabel,by=c("radius","group"))
normtest.noLabel.sum<-data.noLabel.unrepeat %>% 
  dlply(.(radius,group),with,shapiro.test(intersections)) %>%
  ldply(with,data.frame(method=method,pVal=p.value))
data.noLabel.tTest<-normtest.noLabel.sum %>%
  dplyr::filter(pVal>0.05) %>%
  dplyr::group_by(radius) %>% dplyr::mutate(n=n_distinct(group)) %>%
  dplyr::filter(n==2) %>%
  semi_join(x=data.noLabel,by=c("radius","group"))
var.noLabel.sum<-data.noLabel.tTest %>%
  dlply(.(radius),with,var.test(intersections~group)) %>%
  ldply(with,data.frame(method=method,pVal=p.value))
data.noLabel.tTest.varIsEq<-var.noLabel.sum %>%
  dplyr::filter(pVal>0.05) %>%
  semi_join(x=data.noLabel.tTest,by="radius")
data.noLabel.tTest.varNotEq<-setdiff(
  data.noLabel.tTest,data.noLabel.tTest.varIsEq)
data.noLabel.wilcox<-setdiff(data.noLabel,data.noLabel.tTest)
tTest.noLabel.varIsEq<-data.noLabel.tTest.varIsEq %>%
  dlply(.(radius),with,t.test(intersections~group,var.equal=T)) %>%
  ldply(with,data.frame(method=method,pVal=p.value))
tTest.noLabel.varNotEq<-data.noLabel.tTest.varNotEq %>%
  dlply(.(radius),with,t.test(intersections~group)) %>%
  ldply(with,data.frame(method=method,pVal=p.value))

wilcox.noLabel<-data.noLabel.wilcox %>%
  dlply(.(radius),with,wilcox_test(intersections~group)) %>%
  ldply(method_p)
result.noLabel<-rbind(
  tTest.noLabel.varIsEq,tTest.noLabel.varNotEq,wilcox.noLabel)



# aov.normalized<-aov(intersections~radius_normalized*group,
#                     data.normalized.nolabel.unrepeat)
# aovp.normalized.label<-llply(
#   data.normalized.label,aovp,formula=intersections~radius_normalized*group,
#   perm='Prob',seqs=T)

# analysis of cumulative intersections
data.noLabel.cum.unrepeat<-data.noLabel.cum %>%
  dplyr::group_by(radius,group) %>%
  dplyr::summarise(n=n_distinct(cum_inters),N=length(cum_inters)) %>%
  filter(n==1|N<3) %>%
  anti_join(x=data.noLabel.cum,by = c("radius","group"))
normtest.cum<-dlply(data.noLabel.cum.unrepeat,.(radius,group),
                    with,shapiro.test(cum_inters))
normtest.cum.sum<-ldply(normtest.cum,with,
                        data.frame(pVal=p.value,method=method))
with(normtest.cum.sum,all(pVal>0.01))

data.noLabel.cum.unrepeat.tmp<-ldply(data.label.cum) %>% select_at(c(2:7))
data.label.cum.unrepeat<-data.noLabel.cum.unrepeat.tmp %>%
  group_by(label,radius,group) %>% 
  summarise(n=n_distinct(cum_inters),N=length(cum_inters)) %>%
  filter(n==1|N<3) %>% 
  anti_join(x=data.noLabel.cum.unrepeat.tmp,by=c("label","radius","group"))
normtest.label.cum<-dlply(data.label.cum.unrepeat,.(label,radius,group),
                          with,shapiro.test(cum_inters))
normtest.label.cum.sum<-ldply(normtest.label.cum,with,
                              data.frame(pVal=p.value,method=method))
# 63/72 entry p.value >0.05
lm.noLabel.cum<-lm(cum_inters~radius*group,data=data.noLabel.cum)

data.label.cum.bind<-bind_rows(data.label.cum$apical,data.label.cum$basal)
lm.label.cum<-lm(cum_inters~radius+group*label,data.label.cum.bind)
aov.noLabel.cum<-aov(cum_inters~radius*group,data=data.noLabel.cum)
aov.label.cum<-aov(cum_inters~radius+group*label,data.label.cum.bind)
multcomp.label.cum<-TukeyHSD(aov.label.cum)

# Sholl analysis with apical/basal

result.label.wilcox<-data.cut.fill %>% 
  dlply(.(label,radius),with,wilcox_test(intersections~group)) %>%
  ldply(method_p)
result.label.oneway<-data.cut.fill %>% 
  dlply(.(label,radius),with,oneway_test(intersections~group)) %>%
  ldply(method_p)
result.label<-full_join(
  result.label.wilcox,result.label.oneway,by=c("label","radius"))

# Sholl analysis with apical/basal with normalized radius

result.normalized.label<-data.normalized.fill.filtered %>%
  dplyr::mutate(group=as.factor(group)) %>%
  dlply(.(label,radius_normalized),with,wilcox_test(intersections~group)) %>%
  ldply(method_p)

#----visualization-----
library(ggplot2)
library(plotrix)
library(ggsignif)
library(stringr)
# plot Sholl intersections with cummulative radius
data.label.cum.fig<-ldply(data.label.cum) %>% select_at(2:7) %>%
  mutate(radius=as.integer(levels(radius)[radius]))
data.label.cum.fig.sum<-data.label.cum.fig %>% dplyr::group_by(label,radius,group) %>%
  dplyr::summarise(mu=mean(cum_inters),se=std.error(cum_inters),
                    ymin=mu-se,ymax=mu+se)
data.label.cum.signifpos<-ddply(
  data.label.cum.fig.sum,.(label,group),
  summarise,max_r=max(radius),max_mu=max(mu)) %>% mutate(line_id=1:4)
data.label.cum.signifline<-data.frame(
  id=c(rep(c(1,2,3,4),each=4)),
  x=c(540,550,555,545)[c(1,2,2,1,1,3,3,1,1,4,4,1,2,3,3,2)],
  y=data.label.cum.signifpos$max_mu[c(1,1,3,3,4,4,2,2,1,1,2,2,4,4,3,3)],
  label=c('apical','basal')[c(rep(NA,8),rep(c(1,2),each=4))],
  group=c('GFP+','GFP-')[c(rep(2,4),rep(1,4),rep(NA,8))]
)
data.label.cum.signifstar<-data.label.cum.signifline %>%
  dplyr::group_by(id,label,group) %>%
  dplyr::summarise(x=max(x),y=mean(y))
data.label.cum.signifstar$p<-
  with(multcomp.label.cum,c(group[1,c(4,4)],label[1,c(4,4)]))
p_to_signif<-function(x){
  signifi<-cut(x,breaks = c(0,0.001,0.01,0.05,1),labels = c('***','**','*',''))
  return(levels(signifi)[signifi])
}
data.label.cum.signifstar$signif<-p_to_signif(data.label.cum.signifstar$p)

fig.label.cum<-ggplot(data.label.cum.fig.sum,
                  aes(radius,mu))+
  geom_ribbon(aes(ymin=ymin,ymax=ymax,color=group,fill=group,linetype=label),
              alpha=0.1,color=NA)+
  geom_line(aes(color=group,linetype=label))+
  xlab(expression(paste("radius from soma(",mu,"m)")))+
  ylab(paste("cummulative intersections number"))+
  scale_x_continuous(breaks = seq(0,540,60))+
  theme_classic()
fig.label.cum.signif<-fig.label.cum+
  geom_path(aes(x=x,y=y,group=id),data=data.label.cum.signifline)+
  ggrepel::geom_text_repel(aes(x=x,y=y,label=signif),data.label.cum.signifstar)
fig.cum.facetLabel<-fig.label.cum+facet_grid(.~label)+
  geom_path(aes(x=x,y=y),data=dplyr::filter(data.label.cum.signifline,!is.na(label)))+
  geom_text(aes(x=x,y=y,label=signif),dplyr::filter(data.label.cum.signifstar,!is.na(label)))
fig.cum.facetGrp<-fig.label.cum+facet_grid(.~group)+
  geom_path(aes(x=x,y=y),data=dplyr::filter(data.label.cum.signifline,!is.na(group)))+
  geom_text(aes(x=x,y=y,label=signif),dplyr::filter(data.label.cum.signifstar,!is.na(group)))

# plot intersections without apical/basal units
data.noLabel.fig0<-sholl.data.fill %>% dplyr::group_by(radius,group,id) %>%
  dplyr::summarise(intersections=sum(intersections)) %>%
  dplyr::group_by(radius,group) %>% dplyr::summarise(mu=mean(intersections))
data.noLabel.fig<-data.noLabel.fill %>% dplyr::group_by(radius,group) %>%
  dplyr::summarise(mu=mean(intersections),se=std.error(intersections),
                   ymin=mu-se,ymax=mu+se)
data.normalized.fig0<-sholl.normalized.data %>%
  dplyr::group_by(radius_normalized,group,id) %>%
  dplyr::summarise(intersections=sum(intersections)) %>%
  dplyr::group_by(radius_normalized,group) %>%
  dplyr::summarise(mu=mean(intersections))
data.normalized.fig<-data.normalized.noLabel %>%
  dplyr::group_by(radius_normalized,group) %>% 
  dplyr::summarise(mu=mean(intersections),se=std.error(intersections),
                   ymin=mu-se,ymax=mu+se) %>% ungroup() %>%
  dplyr::mutate(radius_normalized=as.numeric(
    levels(radius_normalized)[radius_normalized]))
data.normalized.signif<-data.normalized.fig %>%
  dplyr::group_by(radius_normalized) %>%
  dplyr::summarise(ypos=max(ymax)) %>%
  mutate(radius_normalized=as.factor(radius_normalized)) %>%
  inner_join(result.normalized.noLabel,by="radius_normalized") %>%
  dplyr::mutate(signif=p_to_signif(pVal)) %>%
  mutate(x=as.numeric(levels(radius_normalized)[radius_normalized])) %>%
  dplyr::select(c(6,2,5))
data.noLabel.signif<-data.noLabel.fig %>%
  dplyr::group_by(radius) %>% dplyr::summarise(ypos=max(ymax)) %>%
  mutate(radius=as.factor(radius)) %>%
  inner_join(result.noLabel,by="radius") %>%
  mutate(signif=p_to_signif(pVal),x=as.numeric(radius)) %>%
  dplyr::select(c(6,2,5))

fig.nolabel<-ggplot(data.noLabel.fig,aes(radius,mu))+
  geom_line(aes(color=group),data.noLabel.fig0)+
  geom_errorbar(aes(ymin=ymin,ymax=ymax,color=group),width=12)+
  geom_text(aes(x=x,y=ypos,label=signif),data.noLabel.signif,size=8)+
  xlab(expression(paste("radius from soma(",mu,"m)")))+
  ylab(paste("Sholl intersection number"))+
  theme_classic()
fig.normalized.nolabel<-ggplot(data.normalized.fig,aes(radius_normalized,mu))+
  geom_line(aes(color=group),data.normalized.fig0)+
  geom_errorbar(aes(ymin=ymin,ymax=ymax,color=group),width=0.02)+
  xlab("max normalized radius")+
  ylab("Sholl intersection number")+
  scale_x_continuous(breaks = seq(0,1,0.2),limits = c(0,1))+
  theme_classic()

# plot intersections with apical/basal

data.label.fig0<-sholl.data.fill %>% dplyr::group_by(radius,group,label) %>%
  dplyr::summarise(mu=mean(intersections),se=std.error(intersections),
                   ymin=mu-se,ymax=mu+se)
data.label.fig<-data.cut.fill %>% dplyr::group_by(radius,group,label) %>%
  dplyr::summarise(mu=mean(intersections),se=std.error(intersections),
                   ymin=mu-se,ymax=mu+se) %>% ungroup() %>%
  dplyr::mutate(radius=as.integer(levels(radius)[radius]))
data.label.signif<-result.label.wilcox %>% 
  dplyr::mutate(radius=as.integer(levels(radius)[radius]),
                signifi=p_to_signif(pVal)) %>%
  right_join(data.label.fig,by=c("label", "radius")) %>% 
  dplyr::group_by(label,radius) %>%
  dplyr::mutate(ypos=max(ymax)) %>%
  dplyr::filter(!is.na(pVal)) %>% dplyr::select(c(1,2,11,5)) %>% unique()
  
fig.label<-ggplot(data.label.fig0,aes(radius,mu))+
  geom_line(aes(color=group,linetype=label))+
  geom_ribbon(aes(ymin=ymin,ymax=ymax,fill=group,linetype=label),alpha=0.1)+
  geom_errorbar(aes(ymin=ymin,ymax=ymax,color=group),
                data = data.label.fig,width=12)+
  geom_text(aes(radius,ypos,label=signifi),data.label.signif,size=5)+
  xlab(expression(paste("radius from soma(",mu,"m)")))+
  ylab(paste("Sholl intersection number"))+
  theme_classic()
fig.label.facet<-fig.label+facet_grid(label~.,scales="free_y")

data.normalized.label.fig0<-data.normalized.fill %>%
  dplyr::group_by(radius_normalized,group,label) %>%
  dplyr::summarise(mu=mean(intersections),se=std.error(intersections),
                   ymin=mu-se,ymax=mu+se)
data.normalized.label.fig<-data.normalized.fill.filtered %>%
  dplyr::group_by(radius_normalized,group,label) %>%
  dplyr::summarise(mu=mean(intersections),se=std.error(intersections),
                   ymin=mu-se,ymax=mu+se)
data.normalized.label.signif<-result.normalized.label %>% 
  dplyr::mutate(signifi=p_to_signif(pVal)) %>% 
  right_join(data.normalized.label.fig,by = c("label", "radalized")) %>% 
  dius_normplyr::group_by(label,radius_normalized) %>% 
  dplyr::mutate(ypos=max(ymax)) %>% dplyr::filter(!is.na(pVal)) %>% 
  dplyr::select(c(1,2,11,5)) %>% unique()

fig.normalized.label<-ggplot(data.normalized.label.fig0,
                             aes(radius_normalized,mu,color=group,
                                 linetype=label,ymin=ymin,ymax=ymax))+
  geom_line()+geom_ribbon(aes(color=NULL,fill=group),alpha=0.1)+
  geom_errorbar(aes(linetype=NULL),data=data.normalized.label.fig,width=0.02)+
  geom_text(aes(radius_normalized,ypos,label=signifi,linetype=NULL,color=NULL,
                ymin=NULL,ymax=NULL),
            data.normalized.label.signif,size=5,show.legend=F)+
  xlab("max normalized radius")+
  ylab("Sholl intersection number")+
  scale_x_continuous(breaks = seq(0,1,0.2),limits = c(0,1))+
  theme_classic()
fig.normalized.label.facet<-fig.normalized.label+
  facet_grid(label~.,scales='free_y')
