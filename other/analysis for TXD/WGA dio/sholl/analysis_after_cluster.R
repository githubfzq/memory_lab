# Data saved as "data_of_analysis_after_cluster.Rdata"
# ---load data before-----

require(tidyverse)
require(plyr)
require(modelr)
require(magrittr)
require(purrr)
require(xlsx)

ev1<-new.env()
ev2<-new.env()
ev3<-new.env()

load("morphology data.RData",envir = ev1)
load("sholl with label.RData",envir = ev2)
load("neuron_cluster_data.Rdata",envir = ev3)


neuron_ids<-get("data.select.cat",envir = ev3) %>% select_at(1:2)
neuron_ids$.id[17]<-"(GFP+ near)slice_2_red neuron1"
neuron_ids$.id[12]<-"slice_1_green neuron3"
data.pos.cluster<-get("data.pos.cluster",envir = ev3) %>% 
  select_at("cluster_complete")
data.neg.cluster<-get("data.neg.cluster",envir = ev3) %>% 
  select_at("cluster_complete")
data.cluster<-bind_rows(data.pos.cluster,data.neg.cluster) %>% 
  {bind_cols(neuron_ids,.)} %>% as_tibble() %>% mutate_at("treat",as.character)
data.cluster[[1,3]]<-NA # neuron 1 as outlier

# universe neuron id to match cluster id
universe_neuron_id<-function(df){
  df %>% mutate_at(".id",funs(if_else(.=="(GFP+ near)slice_2_red_1 neuron1",
                                      "(GFP+ near)slice_2_red neuron1",.))) %>%
    mutate_at(".id",funs(if_else(.=="slice_1_green neuron2",
                                 "slice_1_green neuron3",.)))
}

data.length<-get("lengthData2",envir = ev1) %>% as_tibble() %>% universe_neuron_id() %>%
  left_join(data.cluster,by=c(".id","treat")) %>% filter(!is.na(cluster_complete))
data.area<-get("areaData2",envir = ev1) %>% as_tibble() %>% universe_neuron_id() %>%
  left_join(data.cluster,by=c(".id","treat")) %>% filter(!is.na(cluster_complete))
data.volume<-get("volumeData2",envir = ev1) %>% as_tibble() %>% universe_neuron_id() %>%
  left_join(data.cluster,by=c(".id","treat")) %>% filter(!is.na(cluster_complete))
data.sholl<-get("sholl.data.fill",envir = ev2) %>% as_tibble() %>%
  mutate_if(is.factor,as.character) %>% left_join(data.cluster,by=c(id=".id",group="treat")) %>%
  filter(!is.na(cluster_complete))
data.sholl_normed<-get("sholl.normalized.data",envir = ev2) %>% as_tibble() %>%
  mutate_if(is.factor,as.character) %>% left_join(data.cluster,by=c(id=".id",group="treat")) %>%
  filter(!is.na(cluster_complete)) %>% {
    dat<-.
    select_at(.,c(3,4,6,7)) %>% unique() %>% {
      modl<-.
      dat %>% data_grid(radius_normalized,label,id) %>% left_join(.,modl,by=c("label","id"))
    }
  } %>% full_join(data.sholl_normed,.,by=names(.)) %>% 
  replace_na(list(intersections=0))


depth_length_data<-get("DataAll",envir = ev1) %>% .[[6]] %>% as_tibble() %>%
  select_at(c(1:2,5,8,10,12))
depth_area_data<-get("DataAll",envir = ev1) %>% .[[1]] %>% as_tibble() %>%
  select_at(c(1:2,5,8,10,12)) 
depth_volume_data<-get("DataAll",envir = ev1) %>% .[[15]] %>% as_tibble() %>%
  select_at(c(1:2,5,8,10,12))
depth_data<-depth_length_data %>% full_join(depth_area_data,by=names(.)[-2]) %>%
  full_join(depth_volume_data,by=names(.)[c(-2,-7)]) %>%
  rename_at(c(2,4,6:8),~c("Length","label","group","Area","Volume")) %>%
  group_by_at(vars(1,3:4,6)) %>% summarise_at(vars(2,7:8),sum) %>% ungroup() %>%
  mutate_if(is.factor,as.character) %>% universe_neuron_id() %>%
  left_join(data.cluster,by=".id") %>%
  filter(!is.na(cluster_complete))

# ----analysis-------


# analysis of geometric data

data.length.clustered<-data.length %>% mutate(subgroup=paste0(treat,":cluster_",cluster_complete))
compares<-data.length.clustered %>% expand(subgroup) %>% 
  expand(subgroup[1:2],subgroup[3:4]) %>% rename_all(funs(paste0("compares",1:2)))
nest_cluster_compares<-. %>% crossing(compares,.) %>%
  filter(compares1==subgroup | compares2==subgroup) %>% nest(-c(compares1,compares2))

res.length<-group_test(data.length,"`Default Labels`","Length ~ treat")
res.length.clustered<-data.length.clustered %>% nest_cluster_compares() %>%
  dplyr::mutate(result=map(data,~group_test(.,"`Default Labels`","Length ~ treat")))

data.area.clustered<-data.area %>% mutate(subgroup=paste0(treat,":cluster_",cluster_complete))
res.area<-group_test(data.area,"`Default Labels`","Area ~ treat")
res.area.clustered<-data.area.clustered %>% nest_cluster_compares() %>%
  dplyr::mutate(result=map(data,~group_test(.,"`Default Labels`","Area ~ treat")))

data.volume.clustered<-data.volume %>% mutate(subgroup=paste0(treat,":cluster_",cluster_complete))
res.volume<-group_test(data.volume,"`Default Labels`","Volume ~ treat")
res.volume.clustered<-data.volume.clustered %>% nest_cluster_compares() %>%
  dplyr::mutate(result=map(data,~group_test(.,"`Default Labels`","Volume ~ treat")))

# Sholl analysis with absolute radius

data.sholl.clustered<-data.sholl %>% mutate(subgroup=paste0(group,":cluster_",cluster_complete))
compute.cum.nolabel<-. %>% group_by_at(c("id","group","radius")) %>% 
  summarise_at("intersections",sum) %>% 
  cut_gruop_test("radius",30,"intersections~group",compute_cummulative = T,
                 method="anova")
res.cum.nolabel<-compute.cum.nolabel(data.sholl)
res.cum.nolabel.clustered<-data.sholl.clustered %>% nest_cluster_compares() %>%
  dplyr::mutate(result=map(data,compute.cum.nolabel))

res.cum.label<-cut_gruop_test(data.sholl,"radius",20,
               "intersections~radius+group*label",compute_cummulative = T)
res.cum.label.clustered<-data.sholl.clustered %>% nest_cluster_compares() %>%
  dplyr::mutate(result=map(data,cut_gruop_test,"radius",20,
                           "intersections~radius+group*label",compute_cummulative = T))

compute.nolabel<-. %>% group_by_at(c("id","group","radius")) %>% 
  summarise_at("intersections",sum) %>%
  cut_gruop_test("radius",10,"intersections~group")
res.nolabel<-compute.nolabel(data.sholl)
res.nolabel.clustered<-data.sholl.clustered %>% nest_cluster_compares() %>%
  dplyr::mutate(result=map(data,compute.nolabel))

res.label<-data.sholl %>% cut_gruop_test("radius",10,"intersections~group+label")
res.label.clustered<-data.sholl.clustered %>% nest_cluster_compares() %>%
  dplyr::mutate(result=map(data,~cut_gruop_test(.,"radius",10,"intersections~group+label")))

# Sholl analysis with normalized radius

data.sholl_normed.clustered<-data.sholl_normed %>%
  mutate(subgroup=paste0(group,":cluster_",cluster_complete))
compute.normed.cum.nolabel<-. %>% group_by_at(c("id","group","radius_normalized")) %>%
  summarise_at("intersections",sum) %>% ungroup() %>%
  cut_gruop_test("radius_normalized",0.05,"intersections~group",compute_cummulative = T,
                 method = "anova")
res.normed.cum.nolabel<-compute.normed.cum.nolabel(data.sholl_normed.clustered)
res.normed.cum.nolabel.clustered<-data.sholl_normed.clustered %>% nest_cluster_compares() %>%
  dplyr::mutate(result=map(data,compute.normed.cum.nolabel))

res.normed.cum.label<-cut_gruop_test(data.sholl_normed.clustered,"radius_normalized",0.05,
                                     "intersections~group*label",compute_cummulative = T)
res.normed.cum.label.clustered<-data.sholl_normed.clustered %>% nest_cluster_compares() %>%
  dplyr::mutate(result=map(data,cut_gruop_test,"radius_normalized",0.05,
                           "intersections~group*label",compute_cummulative = T))

compute.normed.nolabel<-. %>% group_by_at(c("id","group","radius_normalized")) %>%
  summarise_at("intersections",sum) %>% ungroup() %>%
  cut_gruop_test("radius_normalized",0.05,"intersections~group")
res.normed.nolabel<-compute.normed.nolabel(data.sholl_normed.clustered)
res.normed.nolabel.clustered<-data.sholl_normed.clustered %>% nest_cluster_compares() %>%
  dplyr::mutate(result=map(data,compute.normed.nolabel))

res.normed.label<-cut_gruop_test(data.sholl_normed,"radius_normalized",0.02,
                                 "intersections~group+label")
res.normed.label.clustered<-data.sholl_normed.clustered %>% nest_cluster_compares() %>%
  dplyr::mutate(result=map(data,cut_gruop_test,"radius_normalized",0.02,
                           "intersections~group+label"))

# analysis of geometric data with depth

depth_data.clustered<-depth_data %>% mutate(subgroup=paste0(group,":cluster_",cluster_complete))
res.length_depth<-group_test(depth_data,"Depth","Length~group",plot_type = "bar")
res.length_depth.clustered<-depth_data.clustered %>% nest_cluster_compares() %>%
  dplyr::mutate(result=map(data,group_test,"Depth","Length~group",plot_type = "bar"))

res.area_depth<-group_test(depth_data,"Depth","Area~group",plot_type = "bar")
res.area_depth.clustered<-depth_data.clustered %>% nest_cluster_compares() %>%
  dplyr::mutate(result=map(data,group_test,"Depth","Area~group",plot_type = "bar"))

res.volume_depth<-group_test(depth_data,"Depth","Volume~group",plot_type = "bar")
res.volume_depth.clustered<-depth_data.clustered %>% nest_cluster_compares() %>%
  dplyr::mutate(result=map(data,group_test,"Depth","Volume~group",plot_type = "bar"))

write_data<-function(){
  if(file.exists("Sholl data.xlsx")) file.exists("Sholl data.xlsx")
  all_table<-tibble(
    tables<-list(data.cluster,res.length$source_data,res.area$source_data,
                 res.volume$source_data,res.length_depth$source_data,
                 res.cum.nolabel$source_data,res.cum.label$source_data,
                 res.normed.cum.nolabel$source_data,res.normed.cum.label$source_data),
    sheats<-c("cluster","length","area","volume","depth","Sholl(sum)","Sholl",
              "r-normalized Sholl(sum)","r-normalized Sholl")
  )
  pwalk(all_table,~write.xlsx(as.data.frame(.x),"Sholl data.xlsx",
                              sheetName = .y,append = T,row.names = F))
}
# write_data()

# -----visualize-----

require(ggplot2)
require(customLayout)

selected_color<-c(RColorBrewer::brewer.pal(9,"Greys")[8],
                  RColorBrewer::brewer.pal(9,"Greens")[8])
clus_titles<-paste(compares$compares1,'~',compares$compares2)
cluster_for_file_name<-clus_titles %>% str_replace("~","vs") %>% str_replace_all(":","_")
add_clust_fig_titles<-function(figs) {
  c(list(figs[[1]]),
    map2(figs[-1],clus_titles,~.x+ggtitle(.y)+theme(plot.title = element_text(hjust = 0.5))))
}
stitch_figs<-function(figs){
  lay1<-lay_new(matrix(1))
  lay2<-lay_new(matrix(1:4,nrow = 2,byrow = T))
  lay3<-lay_bind_col(lay1,lay2,widths = c(2,3))
  lay4<-lay_new(matrix(1:3),heights = c(1,3,1))
  lay5<-lay_split_field(lay3,lay4,1)
  blank<-ggplot()+theme_classic()
  lay_grid(c(list(blank),figs[1],list(blank),figs[2:5]),lay5) %>% ggplotify::as.ggplot()
}

adjust_max_y_tick<-function(fig){
  fig_build<-ggplot_build(fig)
  y_range<-fig_build$layout$panel_params[[1]]$y.range
  y_ticks<-fig_build$layout$panel_params[[1]]$y.major_source
  if((y_range[2]-max(y_ticks))/y_ticks[2]>1/3){
    max_y<-max(y_ticks)+y_ticks[2]
    fig+scale_y_continuous(limits = c(0,max_y))
  }
  else fig
}
export_images<-function(file_name,fig,w,h,dpi=600,scale=2){
  fig_format<-c("png","jpg","tiff","pdf")
  fig_file_name<-paste(file_name,fig_format,sep = '.')
  a_ply(fig_file_name,1,ggsave,fig,width=w,height=h,units="mm",dpi=dpi,scale=scale)
}
shift_signif_y<-function(fig,delta_y){
  line_layer<-get_layer_id(fig,"GeomPath")
  star_layer<-get_layer_id(fig,"GeomText")
  fig$layers[[line_layer]]$data<-fig$layers[[line_layer]]$data %>% mutate(y=y+delta_y)
  fig$layers[[star_layer]]$data<-fig$layers[[star_layer]]$data %>% mutate(y=y+delta_y)
}
figs.length<-c(list(res.length),res.length.clustered$result) %>% transpose() %>% {.$figure}
modify_length_figs<-function(fig) fig+scale_fill_manual(values = selected_color,name=NULL)+
  scale_color_manual(values = selected_color,name=NULL,labels=c('Near','MD-projecting neuron'))+
  ylab("Dendrite length (mm)")+xlab(NULL)+
  scale_x_discrete(labels=c("Apical","Basal","Total")) # 2019-1-19 change legend labels
figs.length<-figs.length %>% map(modify_length_figs)
shift_signif_y(figs.length[[1]],0.5) # 2019-1-17 
figs.length[[1]]<-figs.length[[1]]+theme(legend.position=c(0.35,0.9),
                                         axis.title.y = element_text(margin = margin(r = 5)),
                                         axis.text.x = element_text(margin = margin(t=5)))
figs.length.titled<-add_clust_fig_titles(figs.length)
figs.length.overview<-stitch_figs(figs.length)
figs.length.titled.overview<-stitch_figs(figs.length.titled)

figs.area<-c(list(res.area),res.area.clustered$result) %>% transpose() %>% {.$figure}
modify_area_figs<-function(fig) fig+scale_fill_manual(values = selected_color,name=NULL)+
  scale_color_manual(values = selected_color,name=NULL)+
  ylab(expression(paste("Dendrite area (¡Á",10^3,mu,m^2,")")))+xlab(NULL)+
  scale_x_discrete(labels=c("Apical","Basal","Total"))+
  scale_y_continuous(breaks = function(x) seq(0,x[2],10))
figs.area<-figs.area %>% map(modify_area_figs)
figs.area[[1]]<-figs.area[[1]]+theme(legend.position=c(0.35,0.9),
                                     legend.direction = "horizontal",
                                     axis.title.y = element_text(margin = margin(r = 5)),
                                     axis.text.x = element_text(margin = margin(t=5)))
figs.area.titled<-add_clust_fig_titles(figs.area)
figs.area.overview<-stitch_figs(figs.area)
figs.area.titled<-stitch_figs(figs.area.titled)

figs.volume<-c(list(res.volume),res.volume.clustered$result) %>% transpose() %>% {.$figure}
modify_volume_figs<-function(fig) fig+scale_fill_manual(values = selected_color,name=NULL)+
  scale_color_manual(values = selected_color,name=NULL)+
  ylab(expression(paste("Dendrite volume (¡Á",10^3,mu,m^3,")")))+xlab(NULL)+
  scale_x_discrete(labels=c("Apical","Basal","Total"))+
  scale_y_continuous(breaks = function(x) seq(0,x[2],10),expand = expand_scale(mult = c(0.05,0)))
figs.volume<-figs.volume %>% map(modify_volume_figs)
figs.volume[[1]]<-figs.volume[[1]]+theme(legend.position=c(0.35,0.9),
                                         legend.direction = "horizontal",
                                         axis.title.y = element_text(margin = margin(r = 5)),
                                         axis.text.x = element_text(margin = margin(t=5)))
figs.volume.titled<-add_clust_fig_titles(figs.volume)
figs.volume.overview<-stitch_figs(figs.volume)
figs.volume.titled.overview<-stitch_figs(figs.volume.titled)

figs.sholl_cum_nolabel<-c(list(res.cum.nolabel),res.cum.nolabel.clustered$result) %>%
  transpose() %>% {.$figure}
modify_sholl_cum_figs<-function(fig) fig+scale_fill_manual(values = selected_color,name=NULL)+
  scale_color_manual(values = selected_color,name=NULL)+
  xlab(expression(paste("Radius from soma (",mu,"m)")))+
  ylab(paste("# of cumulative Sholl intersection"))+
  theme(axis.title.y = element_text(margin = margin(0,10,0,0)),
        axis.title.x = element_text(margin=margin(t=8)),
        legend.position = c(0.9,0.35))
signif_text_nudge<-function(fig,x=10){
  testdata<-fig$layers[[5]]$data
  remove_geom(fig,"GeomText")+
    geom_text(aes(x=x,y=y,label=signif_txt),testdata,nudge_x = x)
}
remove_fill_legend<-function(fig){
  layer_id<-get_layer_id(fig,"GeomRibbon")
  fig$layers[[layer_id]]$show.legend<-FALSE
  fig
}
figs.sholl_cum_nolabel<-map(figs.sholl_cum_nolabel,modify_sholl_cum_figs)
figs.sholl_cum_nolabel[[1]] %<>% signif_text_nudge() %>% remove_fill_legend() %>% 
  adjust_max_y_tick()
figs.sholl_cum_nolabel.titled<-add_clust_fig_titles(figs.sholl_cum_nolabel)
figs.sholl_cum_nolabel.overview<-stitch_figs(figs.sholl_cum_nolabel)
figs.sholl_cum_nolabel.titled.overview<-stitch_figs(figs.sholl_cum_nolabel.titled)

remove_linetype_mapping<-function(fig){
  remove_geom(fig,"GeomLine")+geom_line(aes(color=group))
}
adjust_text_repel<-function(fig,lim){
  id_layer<-get_layer_id(fig,"GeomTextRepel")
  text_data<-fig$layers[[id_layer]]$data
  fig1<-fig %>% remove_geom("GeomTextRepel")
  fig1+geom_text_repel(aes(x=x,y=y,label=signif_txt),text_data,xlim = lim)
}
figs.sholl_cum_label<-c(list(res.cum.label),res.cum.label.clustered$result) %>%
  transpose() %>% {.[4:6]}
figs.sholl_cum_label<-map(figs.sholl_cum_label,map,
                          ~modify_sholl_cum_figs(.)+guides(linetype="none")+theme(strip.background = element_blank()))
figs.sholl_cum_label$figure_facet2<-
  figs.sholl_cum_label$figure_facet2 %>%
  map(~.%+% facet_grid(.~label,labeller = as_labeller(first_letter_to_upper))) %>%
  map(remove_linetype_mapping) %>% map(remove_fill_legend) %>%
  map(~adjust_text_repel(.,c(780,810))+xlim(c(0,810)))
figs.sholl_cum_label.titled<-map(figs.sholl_cum_label,add_clust_fig_titles)
figs.sholl_cum_label.overview<-map(figs.sholl_cum_label,stitch_figs)
figs.sholl_cum_label.titled.overview<-map(figs.sholl_cum_label.titled,stitch_figs)
figs.sholl_cum_label$figure_facet2[[1]]<-
  figs.sholl_cum_label$figure_facet2[[1]]+theme(legend.position = c(0.9,0.3))

figs.sholl_nolabel<-c(list(res.nolabel),res.nolabel.clustered$result) %>% 
  transpose() %>% {.$figure}
modify_sholl_figs<-function(fig) fig+xlab(expression(paste("Radius from soma (",mu,"m)")))+
  ylab(paste("# of Sholl intersection"))+
  scale_color_manual(values = selected_color,guide=guide_legend(title = NULL))+
  scale_fill_manual(values = selected_color,guide=guide_legend(title = NULL))+
  theme(axis.title.y = element_text(margin = margin(r=5)),
        axis.title.x = element_text(margin = margin(t=5)))
signif_point_to_region<-function(fig,signif_tb,threshod_r){
  x_range<-signif_tb$result %>% dplyr::filter(p.value<0.05) %>% {.$radius} %>%
    factor_to_numeric()
  if(length(x_range)>0) {
    x_range %<>% {tibble(x=.)} %>% {
      break_i<-which(diff(.$x)>threshod_r)
      dplyr::mutate(.,segs=cut(x,c(min(x)-1,x[break_i],max(x))))
      } %>% 
      group_by(segs) %>% dplyr::summarise(start=range(x)[1],end=range(x)[2],mid=(start+end)/2)

    fig %>% remove_geom("GeomErrorbar") %>% remove_geom("GeomText") %>% {
    .+geom_segment(aes(x=start,xend=end),data=x_range,y=0,yend=0)+
      geom_text(aes(x=mid),data=x_range,y=0.2,label='*')}
    }
  else fig %>% remove_geom("GeomErrorbar")
  }
figs.sholl_nolabel<-figs.sholl_nolabel %>% map(modify_sholl_figs) %>%
  map2(c(list(res.nolabel),res.nolabel.clustered$result),
       signif_point_to_region,20) %>%
  map(remove_fill_legend)
figs.sholl_nolabel.titled<-add_clust_fig_titles(figs.sholl_nolabel)
figs.sholl_nolabel[[1]]<-figs.sholl_nolabel[[1]]+theme(legend.position = c(0.9,0.5))
figs.sholl_nolabel[[1]]$layers[[4]]$aes_params$y<-10

figs.sholl_label<-c(list(res.label),res.label.clustered$result) %>%
  transpose() %>% {.$figure}
signif_point_to_region2<-function(fig,threshod_r){
  cut_segment<-. %>% {
    break_i<-c(min(.)-0.002,.[which(diff(.)>threshod_r)],max(.))
    cut(.,break_i) %>% as.character()
  }
  dat_line_star<-{fig+facet_grid(.~label)} %>% ggplot_build() %>% {.$data[[4]]} %>% as_tibble() %>%
    dplyr::filter(!is.na(label)&label!="")
  if(nrow(dat_line_star)){
    dat_line_star %<>% group_by(PANEL) %>%
      dplyr::mutate(seg=cut_segment(x)) %>% group_by(PANEL,seg) %>%
      dplyr::summarise(start=range(x)[1],end=range(x)[2],mid=(start+end)/2) %>% ungroup() %>% 
      dplyr::mutate(label=c('apical','basal')[PANEL])
    remove_geom(fig,"GeomText")+
      geom_segment(aes(x=start,xend=end,group=label),dat_line_star,y=-0.5,yend=-0.5)+
      geom_text(aes(x=mid,group=label),dat_line_star,y=-0.2,label='*')+
      facet_grid(.~label,labeller = as_labeller(first_letter_to_upper))
  }
  else
    fig+facet_grid(.~label,labeller = as_labeller(first_letter_to_upper))
}
modify_sholl_label_figs<-function(fig) fig %>% signif_point_to_region2(30) %>%
  modify_sholl_figs() %>% remove_geom("GeomErrorbar") %>% {
    .+theme(strip.background = element_blank())+guides(linetype="none")
  }
put_signif_upside<-function(fig){
  fig1<-fig %>% remove_geom("GeomText") %>% remove_geom("GeomSegment")
  signif_data<-fig$layers[[3]]$data %>% dplyr::mutate(y=c(7,7,13),y2=y+0.2)
  fig1+geom_segment(aes(x=start,xend=end,group=label,y=y,yend=y),signif_data)+
    geom_text(aes(x=mid,group=label,y=y2),signif_data,label="*")
}
figs.sholl_label<-figs.sholl_label %>% map(modify_sholl_label_figs) %>%
  map(adjust_max_y_tick)
figs.sholl_label.titled<-add_clust_fig_titles(figs.sholl_label)
figs.sholl_label.overview<-stitch_figs(figs.sholl_label)
figs.sholl_label.titled.overview<-stitch_figs(figs.sholl_label.titled)
figs.sholl_label[[1]]<-figs.sholl_label[[1]] %+% 
  {figs.sholl_label[[1]]$data %>% dplyr::filter(mu!=0)} %+% 
  facet_grid(.~label,scales = "free",labeller = as_labeller(first_letter_to_upper))+
  theme(legend.position = "none") # 2019-1-17 remove legend
figs.sholl_label[[1]] %<>% put_signif_upside() %>% remove_fill_legend()

modify_norm_cum_figs<-function(fig) adjust_max_y_tick(fig)+
  scale_fill_manual(values = selected_color,name=NULL)+
  scale_color_manual(values = selected_color,name=NULL)+
  xlab(expression("Normalized radius"))+
  ylab(paste("# of cumulative Sholl intersection"))+
  guides(linetype="none")+
  theme(legend.position = c(0.2,0.8),strip.background = element_blank(),
        legend.background = element_rect(fill = "transparent"),
        axis.title.y = element_text(margin = margin(r=5)),
        axis.title.x = element_text(margin = margin(t=5)))
figs.norm_cum_nolabel<-c(list(res.normed.cum.nolabel),
                         res.normed.cum.nolabel.clustered$result) %>%
  transpose() %>% {.$figure}
figs.norm_cum_nolabel<-figs.norm_cum_nolabel %>% map(modify_norm_cum_figs) %>%
  map(remove_fill_legend)
figs.norm_cum_nolabel.titled<-add_clust_fig_titles(figs.norm_cum_nolabel)

figs.norm_cum_label<-c(list(res.normed.cum.label),
                         res.normed.cum.label.clustered$result) %>%
  transpose() %>% {.[4:6]}
figs.norm_cum_label<-figs.norm_cum_label %>% map(map,~modify_norm_cum_figs(.))
figs.norm_cum_label$figure_facet2 %<>% 
  map(~. %+%facet_grid(.~label,labeller = as_labeller(first_letter_to_upper))+
        theme(axis.text.x = element_text(angle = 30))+
        scale_x_continuous(limits = c(0,1.1))) %>%
  map(adjust_text_repel,c(1,1.25)) %>% map(remove_fill_legend)
figs.norm_cum_label.titled<-map(figs.norm_cum_label,add_clust_fig_titles)

modify_norm_figs<-function(fig) fig+scale_fill_manual(values = selected_color,name=NULL)+
  scale_color_manual(values = selected_color,name=NULL)+
  xlab(expression("Normalized radius"))+
  ylab(paste("# of Sholl intersection"))+
  guides(linetype="none")+
  theme(legend.position = c(0.8,0.8),strip.background = element_blank(),
        legend.background = element_rect(fill = "transparent"),
        axis.title.y = element_text(margin = margin(r=5)),
        axis.title.x = element_text(margin = margin(t=5)))
signif_point_to_region3<-function(fig,threshod_r){
  cut_segment<-. %>% {
    break_i<-c(min(.)-0.002,.[which(diff(.)>threshod_r)],max(.))
    cut(.,break_i) %>% as.character()
  }
  dat_line_star<-fig %>% ggplot_build() %>% {as_tibble(.$data[[4]])} %>%
    filter(!is.na(label)&label!="")
  if(nrow(dat_line_star)){
    dat_line_star %<>% group_by(PANEL) %>%
      dplyr::mutate(seg=cut_segment(x)) %>% group_by(PANEL,seg) %>%
      dplyr::summarise(start=range(x)[1],end=range(x)[2],mid=(start+end)/2) %>% ungroup()
    remove_geom(fig,"GeomText")+
      geom_segment(aes(x=start,xend=end),dat_line_star,y=0.5,yend=0.5)+
      geom_text(aes(x=mid),dat_line_star,y=0.7,label='*')
  }
  else
    fig
}
figs.norm_nolabel<-c(list(res.normed.nolabel),res.normed.nolabel.clustered$result) %>%
  transpose() %>% {.$figure}
figs.norm_nolabel<-figs.norm_nolabel %>% map(modify_norm_figs) %>%
  map(signif_point_to_region3,0.25) %>%
  map(remove_geom,"GeomErrorbar") %>% map(remove_fill_legend)
figs.norm_nolabel.titled<-add_clust_fig_titles(figs.norm_nolabel)

signif_point_to_region4<-function(fig,threshod_r){
  cut_segment<-. %>% {
    break_i<-c(min(.)-0.002,.[which(diff(.)>threshod_r)],max(.))
    cut(.,break_i) %>% as.character()
  }
  dat_line_star<-{fig+facet_grid(.~label)} %>% ggplot_build() %>% {.$data[[4]]} %>% as_tibble() %>%
    dplyr::filter(!is.na(label)&label!="")
  if(nrow(dat_line_star)){
    dat_line_star %<>% group_by(PANEL) %>%
      dplyr::mutate(seg=cut_segment(x)) %>% group_by(PANEL,seg) %>%
      dplyr::summarise(start=range(x)[1],end=range(x)[2],mid=(start+end)/2) %>% ungroup() %>% 
      dplyr::mutate(label=c('apical','basal')[PANEL])
    remove_geom(fig,"GeomText")+
      geom_segment(aes(x=start,xend=end,group=label),dat_line_star,y=0.5,yend=0.5)+
      geom_text(aes(x=mid,group=label),dat_line_star,y=0.7,label='*')+
      facet_grid(.~label,labeller = as_labeller(first_letter_to_upper))
  }
  else
    fig+facet_grid(.~label,labeller = as_labeller(first_letter_to_upper))
}
put_signif_upside2<-function(fig,y_manul=7.5){
  fig$layers[[3]]$aes_params$y<-y_manul
  fig$layers[[3]]$aes_params$yend<-y_manul
  fig$layers[[4]]$aes_params$y<-y_manul+0.2
  fig$layers[[4]]$aes_params$yend<-y_manul+0.2
  fig
}
figs.norm_label<-c(list(res.normed.label),res.normed.label.clustered$result) %>%
  transpose() %>% {.$figure}
figs.norm_label<-figs.norm_label %>% map(modify_norm_figs) %>%
  map(signif_point_to_region4,0.25) %>% map(remove_geom,"GeomErrorbar") %>%
  map(~.+theme(axis.text.x = element_text(angle=30))) %>% map(remove_fill_legend)
figs.norm_label[[1]]<-put_signif_upside2(figs.norm_label[[1]]) %>% adjust_max_y_tick()
figs.norm_label.titled<-add_clust_fig_titles(figs.norm_label)

figs.length_depth<-c(list(res.length_depth),res.length_depth.clustered$result) %>%
  transpose() %>% {.$figure}
limit_depth<-function(fig){
  fig<-fig %+% dplyr::filter(fig$data,Depth %in% 1:8)
  fig$layers[[3]]$data %<>% subset(Depth %in% 1:8)
  fig$layers[[4]]$data %<>% subset(Depth %in% 1:8)
  fig
} # subset layers of figures into 1~8
modify_length_depth_figs<-function(fig) fig+scale_fill_manual(values = selected_color,name=NULL)+
  scale_color_manual(values = selected_color,name=NULL)+
  ylab("Dendrite length (mm)")+
  xlab("Dendrite depth")+
  theme(legend.position = c(0.2,0.8),
        axis.title.y = element_text(margin = margin(r=5)),
        axis.title.x = element_text(margin = margin(t=5)))
figs.length_depth %<>% map(limit_depth) %>% map(modify_length_depth_figs) %>%
  map(adjust_max_y_tick)
figs.length_depth.titled<-add_clust_fig_titles(figs.length_depth)
figs.length_depth.overview<-stitch_figs(figs.length_depth)
figs.length_depth.titled.overview<-stitch_figs(figs.length_depth.titled)

figs.area_depth<-c(list(res.area_depth),res.area_depth.clustered$result) %>%
  transpose() %>% {.$figure}
modify_area_depth_figs<-function(fig) fig+scale_fill_manual(values = selected_color,name=NULL)+
  scale_color_manual(values = selected_color,name=NULL)+
  ylab(expression(paste("Dendrite area (",mu,m^2,")")))+
  xlab("Dendrite depth")+
  theme(legend.position = c(0.2,0.8),
        axis.title.y = element_text(margin = margin(r=5)),
        axis.title.x = element_text(margin = margin(t=5)))
figs.area_depth %<>% map(limit_depth) %>% map(modify_area_depth_figs) %>%
  map(adjust_max_y_tick)
figs.area_depth.titled<-add_clust_fig_titles(figs.area_depth)
figs.area_depth.overview<-stitch_figs(figs.area_depth)
figs.area_depth.titled.overview<-stitch_figs(figs.area_depth.titled)

figs.volume_depth<-c(list(res.volume_depth),res.volume_depth.clustered$result) %>%
  transpose() %>% {.$figure}
modify_volume_depth_figs<-function(fig) fig+scale_fill_manual(values = selected_color,name=NULL)+
  scale_color_manual(values = selected_color,name=NULL)+
  ylab(expression(paste("Dendrite volume (",mu,m^3,")")))+
  xlab("Dendrite depth")+
  theme(legend.position = c(0.2,0.8),
        axis.title.y = element_text(margin = margin(r=5)),
        axis.title.x = element_text(margin = margin(t=5)))
figs.volume_depth %<>% map(limit_depth) %>% map(modify_volume_depth_figs) %>%
  map(adjust_max_y_tick)
figs.volume_depth.titled<-add_clust_fig_titles(figs.volume_depth)
figs.volume_depth.overview<-stitch_figs(figs.volume_depth)
figs.volume_depth.titled.overview<-stitch_figs(figs.volume_depth.titled)

figs.cluster<-map(c("fig.pos.clust","fig.neg.clust"),get,envir=ev3)

export_required_images<-function(){
  all_figs<-tibble(
    filename=c("length",
               "area",
               "volume",
               "cumulative Sholl sum",
               "cumulative Sholl",
               "Sholl sum",
               "Sholl",
               "cumulative Sholl sum normalized",
               "cumulative Sholl normalized",
               "Sholl sum normalized",
               "Sholl normalized",
               "length depth",
               "area depth",
               "volume depth",
               "GFP+ cluster",
               "GFP- cluster"),
    figure=list(figs.length,
                figs.area,
                figs.volume,
                figs.sholl_cum_nolabel,
                figs.sholl_cum_label$figure_facet2,
                figs.sholl_nolabel,
                figs.sholl_label,
                figs.norm_cum_nolabel,
                figs.norm_cum_label$figure_facet2,
                figs.norm_nolabel,
                figs.norm_label,
                figs.length_depth,
                figs.area_depth,
                figs.volume_depth,
                list(figs.cluster[[1]]),
                list(figs.cluster[[2]])),
    size=list(c(40,30),
              c(40,30),
              c(40,30),
              c(40,40),
              c(40,40),
              c(45,35),
              c(45,35),
              c(40,35),
              c(45,35),
              c(40,30),
              c(45,35),
              c(40,30),
              c(40,30),
              c(40,30),
              c(85,35),
              c(85,30))
  )
  pwalk(all_figs,~export_images(..1,..2[[1]],..3[1],..3[2]))
}
# export_required_images()