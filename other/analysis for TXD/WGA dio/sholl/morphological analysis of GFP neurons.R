# Analysis of morphology of neurons labeled by GFP(GFP+) or not(GFP-),
# by FanZuquan, at July 20,2018

# ----data concatenation----
rootDir<-getwd()
subDir<-list.dirs(recursive = F)
gfpParameterDirs<-list.dirs(subDir[1],recursive = F)
nearParameterDirs<-list.dirs(subDir[2],recursive = F)

require(stringr)
library(stringr)
col_select<-function(x,colNum) x[,colNum]

Parameters<-dir(gfpParameterDirs[2]) %>%
  str_split_fixed('[0-9]{8}_',2) %>% col_select(2) %>%
  str_split_fixed('Statistics_',2) %>% col_select(2) %>%
  str_split_fixed('.csv',2) %>% col_select(1)
ParametersPattern<-Parameters %>%str_replace_all("[\\(\\)=]",".")%>%
  str_c('.csv') #for regex
ParametersPattern<-str_c('[0-9]{8}_(Statistics_|)',ParametersPattern);

gfpTotalFilePath<-dir(subDir[1],recursive = T)
nearTotalFilePath<-dir(subDir[2],recursive = T)

gfpParaIndi<-ParametersPattern%>%
  sapply(str_which,string=gfpTotalFilePath,USE.NAMES = F)
nearParaIndi<-ParametersPattern%>%
  sapply(str_which,string=nearTotalFilePath,USE.NAMES = F)
names(gfpParaIndi)<-Parameters #File location indices
names(nearParaIndi)<-Parameters

library(plyr)
library(dplyr)

readCsvPlus<-function(fl,subDirInd){
  dt2<-file.path(subDir[subDirInd],fl)%>%
    lapply(read.csv,skip=3,check.names=F)
  names(dt2)<-str_match(fl,"/(.*) statis")[,2]
  return(dt2)
} # read.csv + filename extraction

catPara<-function (filelist,subDirInd){
  filelist%>%readCsvPlus(subDirInd)%>%
    ldply(data.frame,check.names=F)
  }

gfpData<-lapply(gfpParaIndi,
                function (i) gfpTotalFilePath[i]%>%catPara(1))
nearData<-lapply(nearParaIndi,
                 function (i) nearTotalFilePath[i]%>%catPara(2))

Datalist<-list(`GFP+`=gfpData,`GFP-`=nearData)

getPara<-function(para,datalst,str) datalst[[para]] %>%
  mutate(treat=str)
catParaAll<-function(para){
  data1<-getPara(para,Datalist[[1]],names(Datalist)[1])
  data2<-getPara(para,Datalist[[2]],names(Datalist)[2])
  rbind(data1,data2)
} #for each parameter : catenate gfp+/-
DataAll<-lapply(Parameters,catParaAll)
for (ind in seq(1,length(DataAll))) write.csv(DataAll[[ind]],paste("Sum of",Parameters[ind],".csv"))

# ------sholl analysis-----------
shollData<-DataAll[[which(Parameters=="Filament_No._Sholl_Intersections")]]
radiusLimit<-shollData %>%
  group_by(.id) %>% summarise(maxRadius=max(Radius))
shollData<-shollData %>% group_by(.id) %>%
  mutate(`max Radius`=max(Radius),
         `Radius Normalized`=Radius/`max Radius`)
shollData2<-shollData %>% 
  filter(Radius %in% seq(0,max(radiusLimit$maxRadius),30)) # 30um step
shollData3<-shollData2 %>% 
  transmute(Radius=factor(Radius),
            treat=factor(treat),
            `Filament No. Sholl Intersections`=`Filament No. Sholl Intersections`)
# shollData3: factor data for anova
anovaSholl<-aov(`Filament No. Sholl Intersections`~Radius+treat,data = shollData3)
# summary() to look result

shollData_temp<-dplyr::select(shollData2,treat,.id,Radius,`Filament No. Sholl Intersections`)
ids<-unique(shollData2$.id)
shollData_temp2<-data.frame(
  treat=c(rep("GFP+",16*26),rep("GFP-",10*26)),
  .id=rep(ids,each=26),
  Radius=seq(0,max(radiusLimit$maxRadius),30),
  extra=0)
shollData4<-shollData_temp %>% full_join(
  shollData_temp2,by=c("treat",".id","Radius"))
shollData4<-shollData4 %>%
  dplyr::select(c("treat",".id","Radius","Filament No. Sholl Intersections"))
shollData4$`Filament No. Sholl Intersections`[is.na(shollData4$`Filament No. Sholl Intersections`)]<-0
# for universal number of radius group
shollData5<-mutate(shollData4,Radius=factor(Radius),treat=factor(treat))
shollData6 <-
  ddply(
    shollData5,
    .(treat, Radius),
    summarise,
    interaction_mean = mean(`Filament No. Sholl Intersections`),
    interaction_sd = sd(`Filament No. Sholl Intersections`)
  )

tTestEach<- function(tab) {
  normRslt <- tryCatch(
    shapiro.test(tab$`Filament No. Sholl Intersections`),
    error=function(e){
      list(p.value=0.001,test.method="Shapiro-Wilk normality test")
    })
  if (normRslt$p.value < 0.05)
    wilcox.test(`Filament No. Sholl Intersections` ~ treat, data = tab)
  else
    t.test(`Filament No. Sholl Intersections` ~ treat, data = tab)
}
# normal distribution: t-test; non-normal : wilcox-test

tTestRslt<-dlply(shollData4,.(Radius),tTestEach)
tTestPval<-ldply(tTestRslt,function(x) data.frame(p.value=x$p.value,method=x$method))


library(ggplot2)
shollFigure1<-ggplot(shollData6,aes(Radius,interaction_mean,group=treat,color=treat))+
  geom_line(size=1)+geom_point()+
  geom_errorbar(
    aes(ymin=interaction_mean-interaction_sd,
        ymax=interaction_mean+interaction_sd,
        group=treat),
    width=0.3)+
  xlab(expression(paste("Distance from Soma(",mu,"m)")))+
  ylab("Number of intersections")+
  scale_x_discrete(breaks=seq(0,750,150))+
  scale_y_continuous(breaks = seq(0,30,5),limits = c(-2,30))+
  scale_color_manual(values = c("grey","red"),name=NULL)+
  theme_classic()+
  theme(legend.position = c(0.5,0.8),
        axis.line = element_line(size=0.8),
        axis.ticks = element_line(size=0.8))

shollData7<-transmute(
  shollData,treat=treat,
  `Radius Normalized`=as.integer(
    cut(`Radius Normalized`,25,include.lowest = T)),
  `Filament No. Sholl Intersections`=`Filament No. Sholl Intersections`)

shollData8 <-ddply(
    shollData7,
    .(treat, `Radius Normalized`),
    summarise,
    mean_intersection = mean(`Filament No. Sholl Intersections`),
    sd_intersection = sd(`Filament No. Sholl Intersections`))

tTestRslt2<-dlply(shollData7,.(`Radius Normalized`),tTestEach)
tTestPval2<-ldply(tTestRslt2,function(x) data.frame(p.value=x$p.value,method=x$method))

shollData8_temp<-ddply(
  shollData8,.(`Radius Normalized`),
  summarise,ymax=max(mean_intersection+sd_intersection))
tTestPval2<-left_join(mutate(tTestPval2,text=if_else(p.value<0.05,"*","")),
          shollData8_temp,by = "Radius Normalized")


shollFigure2<-ggplot(data = shollData8,
       aes(`Radius Normalized`,`mean_intersection`,group=treat,color=treat))+
  geom_line(size=1)+geom_point()+
  geom_errorbar(aes(
    ymax=mean_intersection+sd_intersection,
    ymin=mean_intersection-sd_intersection
  ),
  width=0.3)+
  geom_text(aes(`Radius Normalized`,ymax,label=text,group=NULL,color=NULL),
            tTestPval2,size=6,show.legend = F)+
  scale_x_continuous(breaks = seq(0,25,5),labels = as.character(seq(0,100,20)))+
  scale_y_continuous(breaks = seq(0,30,5),limits = c(0,30))+
  ylab("Number of intersections")+
  xlab("Radius Normalized(%)")+
  scale_color_manual(values = c("grey","red"),name=NULL)+
  theme_classic()+
  theme(legend.position = c(0.5,0.8),
        axis.line = element_line(size=0.8),
        axis.ticks = element_line(size=0.8))

# -----length,area,volume----

lengthData<-DataAll[[which(Parameters=='Dendrite_Length')]]
areaData<-DataAll[[which(Parameters=="Dendrite_Area")]]
volumeData<-DataAll[[which(Parameters=="Dendrite_Volume")]]

lengthData2<-lengthData %>%
  filter(`Default Labels`%in% c('apical dendrite','basal dendrite')) %>%
  ddply(.(treat,.id,`Default Labels`),summarize,
        Length=sum(`Dendrite Length`)/1000,unit="mm")
lengthData2_temp<-ddply(lengthData2,.(treat,.id),summarise,
                        `Default Labels`="total",Length=sum(Length),unit="mm")
lengthData2<-full_join(lengthData2,lengthData2_temp)

areaData2<-areaData %>%
  filter(`Default Labels`%in% c('apical dendrite','basal dendrite')) %>%
  ddply(.(treat,.id,`Default Labels`),summarise,
        Area=sum(`Dendrite Area`)/(10^3),unit="10^3 um^2")
areaData2_temp<-ddply(areaData2,.(treat,.id),summarise,
                      `Default Labels`="total",Area=sum(Area),unit="10^3 um^2")
areaData2<-full_join(areaData2,areaData2_temp)

volumeData2<-volumeData %>%
  filter(`Default Labels`%in% c('apical dendrite','basal dendrite')) %>%
  ddply(.(treat,.id,`Default Labels`),summarize,
        Volume=sum(`Dendrite Volume`)/10^3,unit="10^3 um^3")
volumeData2_temp<-ddply(volumeData2,.(treat,.id),summarise,
                        `Default Labels`="total",Volume=sum(Volume),unit="10^3 um^3")
volumeData2<-full_join(volumeData2,volumeData2_temp)

length.normTest<-dlply(lengthData2,.(`Default Labels`),with,shapiro.test(Length))
length.normTest.pVal<-ldply(length.normTest,function(t) data.frame(pVal=t$p.value))
length.varTest<-dlply(lengthData2,.(`Default Labels`),with,var.test(Length~treat))
length.varTest.pVal<-ldply(length.varTest,function(t) data.frame(pVal=t$p.value))
length.tTest<-dlply(lengthData2,.(`Default Labels`),with,t.test(Length~treat,var.equal=T))
length.tTest.pVal<-ldply(length.tTest,function(t) data.frame(pVal=t$p.value,method=t$method))

area.normTest<-dlply(areaData2,.(`Default Labels`),with,shapiro.test(Area))
area.normTest.pVal<-ldply(area.normTest,function(t) data.frame(pVal=t$p.value))
area.tTest<-dlply(areaData2,.(`Default Labels`),with,wilcox.test(Area~treat))
area.tTest.pVal<-ldply(area.tTest,function(t) data.frame(pVal=t$p.value,method=t$method))

volume.normTest<-dlply(volumeData2,.(`Default Labels`),with,shapiro.test(Volume))
volume.normTest.pVal<-ldply(volume.normTest,function(t) data.frame(pVal=t$p.value))
volume.tTest<-dlply(volumeData2,.(`Default Labels`),with,wilcox.test(Volume~treat))
volume.tTest.pVal<-ldply(volume.tTest,function(t) data.frame(pVal=t$p.value,method=t$method))

# plot length
lengthData.sum<-lengthData2%>%dplyr::group_by(treat,`Default Labels`) %>%
  dplyr::summarise(Length.mean=mean(Length),Length.sd=sd(Length))
length.signifPosi<-lengthData.sum %>% dplyr::group_by(`Default Labels`) %>% 
  dplyr::summarise(yPosition=max(Length.mean+Length.sd)) %>%
  right_join(length.tTest.pVal,by="Default Labels") %>%
  filter(pVal<0.05) %>%
  mutate(signifText="*")

length.fig<-ggplot(lengthData.sum,aes(`Default Labels`,Length.mean,group=treat))+
  geom_bar(aes(fill=treat),stat="identity",position = position_dodge(width=0.7),width = 0.6)+
  geom_errorbar(aes(ymax=Length.mean+Length.sd,ymin=Length.mean-Length.sd),
                position=position_dodge(width=0.7),width=0.1)+
  geom_errorbarh(aes(x=`Default Labels`,xmin=2-0.35/2,xmax=2+0.35/2,y=yPosition+0.2,group=NULL),
                 length.signifPosi,height=0)+
  geom_text(aes(x=`Default Labels`,y=yPosition+0.2,label=signifText,group=NULL),
            length.signifPosi,size=7)+
  ylab("Dendrite Length(mm)")+xlab(NULL)+
  scale_fill_manual(values = c("gray","red"),name=NULL)+
  scale_x_discrete(labels=c("Apical","Basal","Total"))+
  scale_y_continuous(breaks=0:6)+
  theme_classic()+
  theme(axis.text.x = element_text(color="black",size=11))

length.signifPosi2<-lengthData2 %>% dplyr::group_by(`Default Labels`) %>% 
  dplyr::summarise(yPosition=max(Length)) %>%
  right_join(length.tTest.pVal,by="Default Labels") %>%
  filter(pVal<0.05) %>%
  mutate(signifText="*")

length.fig2<-ggplot(lengthData2,aes(
  x=`Default Labels`,y=Length,color=treat))+
  geom_dotplot(position = position_dodge(0.8),stackratio = 0.5,fill="white",
               binaxis = "y",stackdir = "center",binwidth=0.2)+
  geom_crossbar(aes(y=Length.mean,
                    ymin=Length.mean-Length.sd,
                    ymax=Length.mean+Length.sd,
                    fill=treat),
                lengthData.sum,position = position_dodge(0.8),alpha=0.3,
                width=0.7,fatten = 2,show.legend = F)+
  geom_errorbarh(aes(x=`Default Labels`,xmin=2-0.4/2,xmax=2+0.4/2,y=yPosition+0.2,color=NULL),
                 length.signifPosi2,height=0,show.legend = F)+
  geom_text(aes(x=`Default Labels`,y=yPosition+0.2,label=signifText,color=NULL),
            length.signifPosi2,size=7,show.legend = F)+
  ylab("Dendrite Length(mm)")+xlab(NULL)+
  scale_fill_manual(values = c("gray","red"),name=NULL)+
  scale_x_discrete(labels=c("Apical","Basal","Total"))+
  scale_y_continuous(breaks = 0:8)+
  scale_color_manual(values = c("gray","red"),name=NULL)+
  theme_classic()+
  theme(axis.text.x = element_text(color="black",size=11))

# plot area
areaData.sum<-areaData2%>%dplyr::group_by(treat,`Default Labels`) %>%
  dplyr::summarise(Area.mean=mean(Area),Area.sd=sd(Area))
area.fig<-ggplot(areaData2,aes(
  x=`Default Labels`,y=Area,color=treat))+
  geom_dotplot(position = position_dodge(0.8),stackratio = 0.5,fill="white",
               binaxis = "y",stackdir = "center",binwidth = 1.5)+
  geom_crossbar(aes(y=Area.mean,
                    ymin=Area.mean-Area.sd,
                    ymax=Area.mean+Area.sd,
                    fill=treat),
                areaData.sum,position = position_dodge(0.8),alpha=0.3,
                width=0.7,fatten = 2,show.legend = F)+
  ylab(expression(paste("Dendrite Area(",10^3,mu,m^2,")")))+xlab(NULL)+
  scale_fill_manual(values = c("gray","red"),name=NULL)+
  scale_x_discrete(labels=c("Apical","Basal","Total"))+
  scale_y_continuous(breaks = seq(0,70,10))+
  scale_color_manual(values = c("gray","red"),name=NULL)+
  theme_classic()+
  theme(axis.text.x = element_text(color="black",size=11))

# plot volume
volumeData.sum<-volumeData2%>%dplyr::group_by(treat,`Default Labels`) %>%
  dplyr::summarise(Volume.mean=mean(Volume),Volume.sd=sd(Volume))
volume.fig<-ggplot(volumeData2,aes(
  x=`Default Labels`,y=Volume,color=treat))+
  geom_dotplot(position = position_dodge(0.8),stackratio = 0.5,fill="white",
               binaxis = "y",stackdir = "center",binwidth = 1.5)+
  geom_crossbar(aes(y=Volume.mean,
                    ymin=Volume.mean-Volume.sd,
                    ymax=Volume.mean+Volume.sd,
                    fill=treat),
                volumeData.sum,position = position_dodge(0.8),alpha=0.3,
                width=0.7,fatten = 2,show.legend = F)+
  ylab(expression(paste("Dendrite Volume(",10^3,mu,m^3,")")))+xlab(NULL)+
  scale_fill_manual(values = c("gray","red"),name=NULL)+
  scale_x_discrete(labels=c("Apical","Basal","Total"))+
  scale_y_continuous(breaks = seq(0,60,10))+
  scale_color_manual(values = c("gray","red"),name=NULL)+
  theme_classic()+
  theme(axis.text.x = element_text(color="black",size=11))

# -----detailed Sholl analysis---

detailedShollFile<-list.files(pattern = "Sholl Result.*csv")
names(detailedShollFile)<-c('GFP-','GFP+')
detailedShollData<-adply(detailedShollFile,1,read.csv,check.names=F,.id = 'treat')
detailedShollData<-mutate(detailedShollData,id=str_split_fixed(Image,".swc",2)[,1])
detailedShollData<-select(detailedShollData,c(1,43,5,10:28,31:32,34:35,37:38,40:41))
library(reshape2)
Sholl.detailed.data.melted<-melt(detailedShollData,c(1,2))
Sholl.detailed.data.summary<-ddply(Sholl.detailed.data.melted,c(1,3),summarise,Mean=mean(value),Sd=sd(value))
# norm test
Sholl.detailed.normTest<-dlply(Sholl.detailed.data.melted,.(treat,variable),with,shapiro.test(value))
Sholl.detailed.normTest.pVal<-ldply(Sholl.detailed.normTest,function(t) data.frame(pVal=t$p.value))
Sholl.detailed.normTest.pVal<-mutate(Sholl.detailed.normTest.pVal,is.norm=pVal>=0.05)
Sholl.detailed.normTest.sum<-ddply(.data = Sholl.detailed.normTest.pVal,.(variable),summarize,norm=all(is.norm))
# t test/wilcox test
Sholl.detail.tTest.data<-semi_join(Sholl.detailed.data.melted,
                Sholl.detailed.normTest.sum %>% dplyr::filter(norm==T),
                by="variable")
Sholl.detailed.wilcox.data<-semi_join(Sholl.detailed.data.melted,
                Sholl.detailed.normTest.sum %>% dplyr::filter(norm==F),
                by="variable")
Sholl.detailed.VarTest<-dlply(Sholl.detail.tTest.data,.(variable),with,var.test(value~treat))
Sholl.detailed.VarTest.pVal<-ldply(Sholl.detailed.VarTest,function(t) data.frame(pVal=t$p.value))
Sholl.detailed.VarTest.pVal<-mutate(Sholl.detailed.VarTest.pVal,
                                    is.var.equal=pVal>=0.05) # all var equal
Sholl.detailed.tTest<-dlply(Sholl.detail.tTest.data,.(variable),
            with,t.test(value~treat,var.equal=TRUE))
Sholl.detailed.tTest.pVal<-ldply(Sholl.detailed.tTest,with,
                                 data.frame(pVal=p.value,method=method))
Sholl.detailed.wilcox<-dlply(Sholl.detailed.wilcox.data,.(variable),
                             with,wilcox.test(formula=value~treat))
Sholl.detailed.wilcox.pVal<-ldply(Sholl.detailed.wilcox,with,
                                  data.frame(pVal=p.value,method=method))
Sholl.detailed.pVal<-bind_rows(Sholl.detailed.tTest.pVal,Sholl.detailed.wilcox.pVal)
Sholl.detailed.pVal<-mutate(Sholl.detailed.pVal,
                            significant=ifelse(pVal<0.05,'*',''))
