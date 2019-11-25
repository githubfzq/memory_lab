roi<-read.csv("Results_of_ROI.csv")
roi$dye<-c('AAV1','DAPI')
roi$slice_id<-rep(1:3,each=2)

library(tidyverse)
roi<-as_tibble(roi)
data_all<-roi %>% mutate(data_path=paste0("Results_of_",dye,"_slice",slice_id,".csv")) %>%
  mutate(signal_pos=map(data_path,read.csv))
bin_continuous<-function(s,bin,bin_position="middle") {
  if(bin_position=="middle") (s-min(s)) %/% bin *bin+min(s)+bin/2
  else if(bin_position=="left") (s-min(s)) %/% bin *bin+min(s)
  else if(bin_position=="right") (s-min(s)) %/% bin *bin+min(s)+bin
}
dat_AAV<-data_all %>% filter(dye=="AAV1") %>% unnest() %>% select_at(c(2:7,17,18)) %>%
  rename_at(7:8,~c('X','Y'))
dat_DAPI<-data_all %>% filter(dye=="DAPI") %>% unnest() %>% select_at(c(2:7,12:13)) %>%
  rename_at(7,~'X')
d<-bind_rows(dat_AAV,dat_DAPI) %>% mutate(X0=X-BX,Y0=Y-BY,X_bin=bin_continuous(X0,200)) %>%
  group_by(dye,slice_id,X_bin) %>% tally() %>% ungroup() %>% spread(dye,n,fill = 0) %>%
  mutate(percent=AAV1/DAPI,slice_id=as.factor(slice_id))
d<-d %>% expand(slice_id) %>% crossing(tibble(X_bin=seq(min(d$X_bin),1904,200))) %>%
  full_join(d,.,by=c("slice_id","X_bin")) %>% arrange_at(1) %>%
  mutate_if(is.double,replace_na,0)

g<-ggplot(d,aes(x=X_bin,y=percent))+stat_summary(geom="ribbon",fill="red",alpha=0.6)+
  geom_line(aes(group=slice_id))+
  scale_x_continuous(name=expression(paste("Distance from pia (",mu,"m)")))+
  scale_y_continuous(name="mCherry / DAPI")+
  theme_classic()
# ggsave("mCherry percent.png",g,scale = 2,width = 40,height = 30,units = "mm")
ggsave("mCherry percent 2.png",g,scale = 2,width = 40,height = 30,units = "mm")

bind_rows(dat_AAV,dat_DAPI) %>% mutate(X0=X-BX,Y0=Y-BY,X_bin=cut(X0,breaks = seq(0,1200,200))) %>%
  group_by(dye,slice_id,X_bin) %>% tally() %>% ungroup() %>% spread(dye,n,fill = 0) %>%
  mutate(percent=AAV1/DAPI,slice_id=as.factor(slice_id)) %>% write.csv("bin data.csv")
bind_rows(dat_AAV,dat_DAPI) %>% mutate(X0=X-BX,Y0=Y-BY,X_bin=cut(X0,breaks = seq(0,1200,200))) %>%
  group_by(dye,slice_id,X_bin) %>% tally() %>% ungroup() %>% spread(dye,n,fill = 0) %>%
  mutate(percent=AAV1/DAPI,slice_id=as.factor(slice_id)) %>% group_by(X_bin) %>%
  summarise(percent_mean=mean(percent),percent_se=plotrix::std.error(percent)) %>%
  write.csv("bin data summarise.csv")
