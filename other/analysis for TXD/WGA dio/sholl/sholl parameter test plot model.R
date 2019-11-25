m<-aes(x=type,y=I.branches..user.,fill=type)
g<-ggplot(data=data_mean,m)
bar<-geom_bar(stat="identity")
txtm<-aes(y=I.branches..user.,label=txt)
labl<-geom_text(txtm,size=10,fontface="bold")
thm<-theme_classic()
ylb<-ylab("primary branches")
f<-g+bar+labl+thm+ylb
f
