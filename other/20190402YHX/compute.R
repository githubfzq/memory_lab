library(openxlsx)
library(tidyverse)
library(broom)
dat=read.xlsx("./data analyse excel cfos Лузм.xlsx",sheet="summation")
dat=dat %>% filter(!is.na(density) | !is.na(grey))
dat_sum<-dat %>% group_by(region,treatment) %>%
  summarise(mean_density=mean(density,na.rm = T),se_density=plotrix::std.error(density,na.rm = T),
            mean_grey=mean(grey,na.rm = T),se_grey=plotrix::std.error(grey,na.rm = T),
            normality_density=possibly(shapiro.test,list(p.value=NA))(density)$p.value>0.05,
            normality_grey=possibly(shapiro.test,list(p.value=NA))(grey)$p.value>0.05)
res<-dat %>% as_tibble() %>% nest(-region) %>% 
  mutate(anova_density=map(data,~aov(density~treatment,data=.)),
         anova_grey=map(data,~aov(grey~treatment,data=.)),
         anova_density_p=map(anova_density,~tidy(.)[[1,"p.value"]]),
         anova_grey_p=map(anova_grey,~tidy(.)[[1,"p.value"]]),
         Tukey_density=map(anova_density,~tidy(TukeyHSD(.))),
         Tukey_grey=map(anova_grey,~tidy(TukeyHSD(.))))
res_density<-res %>% select_at(c(1,7)) %>% unnest(Tukey_density)
res_grey<-res %>% select(c(1,8)) %>% unnest(Tukey_grey)

write.xlsx(dat_sum,"stat result.xlsx",sheetName="data_summary")
wb=loadWorkbook("./stat result.xlsx")
addWorksheet(wb,"ANOVA test")
res %>% select_at(c(1,5:6)) %>% unnest() %>%
  writeData(wb=wb,sheet = "ANOVA test",x=.)
addWorksheet(wb,"Tukey test for density")
res_density %>% select(c(1,3,7)) %>% writeData(wb,3,x=.)
addWorksheet(wb,"Tukey test for grayvalue")
res_grey %>% select(c(1,3,7)) %>% writeData(wb,4,x=.)
saveWorkbook(wb,"./stat result.xlsx",overwrite = T)

library(ggplot2)
fig_density<-ggplot(dat_sum,aes(x=region,y=mean_density))+
  stat_identity(aes(fill=treatment),geom="bar",position = "dodge")+
  geom_errorbar(aes(group=treatment,ymin=mean_density+se_density,ymax=mean_density-se_density),
                position = position_dodge(width = 0.8),width=0.2)+
  scale_y_continuous(name="mean density")+
  scale_fill_brewer(type="div")
fig_gray<-ggplot(dat_sum,aes(x=region,y=mean_grey))+
  stat_identity(aes(fill=treatment),geom="bar",position="dodge")+
  geom_errorbar(aes(group=treatment,ymin=mean_grey+se_grey,ymax=mean_grey-se_grey),
                position=position_dodge(width = 0.8),width=0.2)+
  scale_y_continuous(name="mean gray value")+
  scale_fill_brewer(type="div")
