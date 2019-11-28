require(tidyverse)
require(plyr)
require(broom)
require(plotrix)
require(ggsignif)
require(ggrepel)
require(rlang)

# load data frame, do normal test, var-test, t-test and wilcox rank test, and return test p-value
# data frame. 
# formula_str: y~f(X) [x=x1,x2,...] group by "group_var"
# if compute_cummulative is TRUE, formula is Y=∑f(group_var)~X group by "group var"
# sample_var: the variable which compute errorbar
group_test<-function(data,group_var,formula_str,test_method="auto",plot_type="point",
                     plot_data=NULL){
  group_var_sym<-sym(group_var)
  group_var_trim<-ifelse(str_detect(group_var,"^`.*`$"),
                         str_match(group_var,"^`(.*)`$")[,2],group_var)
  if(!is.factor(data[[group_var]])) data<-data %>% mutate_at(group_var_trim,as.factor)
  formula_varName=names(get_all_vars(formula_str,data))
  quote_vars<-c(group_var,formula_varName[-1]) %>% unique()
  if(test_method=="auto"){
    normTest<-data %>% ddply(as.quoted(quote_vars),norm_test_each,formula_varName[1]) %>%
      as_tibble()
    data.normal<-normTest %>% dplyr::mutate(Signif=p.value>0.05) %>% 
      ddply(as.quoted(quote_vars[1]),plyr::summarise,Signif=all(Signif)) %>% 
      dplyr::filter(Signif==TRUE) %>% semi_join(x=data,by=group_var_trim) %>% as_tibble()
    tTest<-data.normal %>% ddply(as.quoted(group_var),test_each,formula_str,TRUE) %>%
      as_tibble()
    data.wilcox<-anti_join(data,data.normal,by=names(data))
    wilcoxTest<-data.wilcox %>% ddply(as.quoted(group_var),test_each,formula_str,FALSE) %>%
      as_tibble()
    if(nrow(data.normal)==0) res<-wilcoxTest
    else{
      if(nrow(data.wilcox)==0) res<-tTest
      else res<-bind_rows(tTest,wilcoxTest) %>% arrange_at(group_var_trim)
    }
    aggregate_data<-data %>% group_by_at(c(group_var_trim,formula_varName[-1])) %>% 
      summarise_at(formula_varName[1],funs(mu=mean,se=std.error,ymin=mu-se,ymax=mu+se)) %>%
      ungroup()
    if(plot_type=="point"){
      signif_tab<-stat_signif_data(res,data,group_var_trim,ref_tb_aggregated = F,
                                 y_var = formula_varName[1],hv_adjust=0.2,show_ns=F)
      fig<-ggplot(data,aes(x=!!sym(group_var_trim),
                           y=!!sym(formula_varName[1])))+
        geom_point(aes(color=!!sym(formula_varName[2])),size=2,
                   position = position_jitterdodge(jitter.width = .1,dodge.width = .8),
                   alpha=0.7)+
        geom_crossbar(aes(y=mu,ymin=ymin,ymax=ymax,fill=!!sym(formula_varName[2]),
                          color=!!sym(formula_varName[2])),
                      aggregate_data,position = position_dodge(0.8),
                      alpha=0.3,width=0.7,fatten = 2,show.legend = F,na.rm = T)+
        geom_path(aes(x=x,y=y,group=line_id),signif_tab$signif_line,inherit.aes = F)+
        geom_text(aes(x=x,y=y,label=signif_txt),signif_tab$signif_level,
                  inherit.aes = F,nudge_y = 0.2,na.rm = T)+
        theme_classic()+
        theme(axis.text.x = element_text(color="black",size=11))
    }
    else if(plot_type=="bar"){
      signif_tab<-stat_signif_data(res,aggregate_data,group_var_trim,y_var = "mu",
                                   hv_adjust=0.2)
      
      fig<-ggplot(aggregate_data,
                  aes(x=!!sym(group_var_trim),y=mu,color=!!sym(formula_varName[2])))+
        geom_bar(aes(fill=!!sym(formula_varName[2])),
                 position = position_dodge(0.8),stat = "identity",
                 alpha=0.6,width = 0.7,na.rm = T)+
        geom_errorbar(aes(ymin=ymin,ymax=ymax),position = position_dodge(0.8),na.rm = T,
                      width=0.3)+
        geom_path(aes(x=x,y=y,group=line_id),signif_tab$signif_line,inherit.aes = F)+
        geom_text(aes(x=x,y=y,label=signif_txt),signif_tab$signif_level,
                  inherit.aes = F,nudge_y = 0.2,na.rm = T)+
        theme_classic()
    }
    else{
      # plot_type="line_error"
      aggregate_data<-aggregate_data %>% mutate_at(group_var_trim,funs(factor_to_numeric))
      signif_tab<-res %>% mutate_at(group_var,factor_to_numeric) %>% 
        stat_signif_data(aggregate_data,c(group_var_trim,formula_varName[c(-1,-2)]),has_line = F)
      if(length(formula_varName)==2) linetype_var<-NULL else linetype_var<-sym(formula_varName[3])
      tip_width<-data %>% mutate_at(group_var,factor_to_numeric) %>% 
        {range(.[[group_var]])} %>% {(.[2]-.[1])/70}
      fig<-ggplot(plot_data,aes(x=!!sym(group_var_trim),y=mu))+
        geom_line(aes(color=!!sym(formula_varName[2]),linetype=!!linetype_var))+
        geom_ribbon(aes(ymin=ymin,ymax=ymax,fill=!!sym(formula_varName[2]),
                        linetype=!!linetype_var),alpha=0.4)+
        geom_errorbar(aes(ymin=ymin,ymax=ymax,color=!!sym(formula_varName[2])),
                      aggregate_data,width=tip_width,na.rm = T)+
        geom_text(aes(x=!!sym(group_var),y=y,label=signif_txt),signif_tab,na.rm = T)+
        theme_classic()
    }

    list(source_data=data,
         aggregate_data=aggregate_data,
         result=res,
         figure=fig) %>%
      return()
    }
  else{
    result<-data %>% aov(formula = as.formula(formula_str),data = .) %>% TukeyHSD() %>% tidy() %>%
      filter_at("term",any_vars(.!=group_var))
    aggr<-data %>% group_by_at(quote_vars) %>% 
      summarise_at(formula_varName[1],funs(mu=mean,se=std.error,ymin=mu-se,ymax=mu+se)) %>%
      ungroup() %>% mutate_at(group_var,factor_to_numeric)
    signif_res<-stat_signif_data(result,plot_data,group_var_trim,line_orient = "vertical")
    if(length(formula_varName)>=4){
      fig_base<-ggplot(plot_data,aes(!!sym(group_var_trim),mu))+
        geom_line(aes(color=!!sym(quote_vars[2]),linetype=!!sym(quote_vars[3])))+
        geom_ribbon(aes(ymin=ymin,ymax=ymax,fill=!!sym(quote_vars[2]),
                        linetype=!!sym(quote_vars[3])),alpha=0.4)+
        geom_point(data=aggr,aes(color=!!sym(quote_vars[2])),size=1)+
        theme_classic()
      fig<-fig_base+geom_path(aes(x=x,y=y,group=line_id),
                              signif_res$signif_line)+
        geom_text_repel(aes(x=x,y=y,label=signif_txt),
                        signif_res$signif_level,show.legend = F,na.rm = T)
      fig_facet<-function(var_sym) fig_base+
        geom_path(aes(x=x,y=y,group=line_id),
                  signif_res$signif_line %>% filter(!is.na(!!var_sym)))+
        geom_text_repel(aes(x=x,y=y,label=signif_txt),
                        signif_res$signif_level %>% filter(!is.na(!!var_sym)),
                        show.legend = F,na.rm = T)+
        facet_grid(cols =vars(!!var_sym))
      fig_facet_1<-fig_facet(sym(quote_vars[2]))
      fig_facet_2<-fig_facet(sym(quote_vars[3]))
  
      list(source_data=data %>% mutate_at(group_var,factor_to_numeric),
           aggregate_data=aggr,
           result=result,
           figure=fig,
           figure_facet1=fig_facet_1,
           figure_facet2=fig_facet_2) %>%
        return()
    }
    else{
      fig<-ggplot(plot_data,aes(!!sym(group_var_trim),mu))+
        geom_line(aes(color=!!sym(quote_vars[2])))+
        geom_ribbon(aes(ymin=ymin,ymax=ymax,fill=!!sym(quote_vars[2])),alpha=0.4)+
        geom_point(aes(color=!!sym(quote_vars[2])),aggr,size=1)+
        geom_path(aes(x=x,y=y),signif_res$signif_line)+
        geom_text(aes(x=x,y=y,label=signif_txt),signif_res$signif_level,show.legend = F)+
        theme_classic()
      list(source_data=data %>% mutate_at(group_var,factor_to_numeric),
           aggregate_data=aggr,
           result=result,
           figure=fig) %>%
        return()
    }
  }
}
norm_test_each<-function(d,quote_var){
  res_tab<-d[[quote_var]] %>% 
    possibly(shapiro.test,
             shapiro.test(runif(10)) %>% modify_at("p.value",~0))(.) %>%
    with(tibble(method,p.value))
  return(res_tab)
}

# test_each: automaticaly choose test method between t.test and wilcox.test according to 
# normality of data.
# if RHS of formula has two or more variable, the 2nd and later variables is/are regarded as
# grouping variables.
test_each<-function(d,formula_str,is_norm){
  if(!is_tibble(d)) d<-as_tibble(d)
  formula_varname<-get_formula_varnames(formula_str,d)
  names_rhs<-formula_varname[-1]
  if(length(names_rhs)==1){
    if(is_norm){
      auto_t_test<-function(dt){
        res.varTest<-dt %>% var.test(formula=as.formula(formula_str),data=.)
        if(res.varTest$p.value>0.05)
          res.Test<-dt %>% t.test(formula=as.formula(formula_str),data=.,var.equal=T)
        else res.Test<-dt %>% t.test(formula=as.formula(formula_str),data=.)
        return(res.Test)
      }
      res.Test<-safely(auto_t_test)(d) %>% {.$result}
    }
    else res.Test<-quietly(possibly(wilcox.test,NA))(as.formula(formula_str),data=d) %>% .$result
    res_tab<-res.Test %>% 
      possibly(with,tibble(method=NA_character_,p.value=NA_real_))(.,tibble(method,p.value))
    return(res_tab)
  }
  else{
    group_var<-names_rhs[-1]
    formula_str<-paste(formula_varname[1],formula_varname[2],sep = '~')
    res_tab<-d %>% ddply(as.quoted(group_var),test_each,formula_str,is_norm) %>% as_tibble()
    return(res_tab)
  }
}

# load data frame, cut continous variable to discrite groups, and then do group test
# By default, if RHS in formula has two or more variables, test method is changed to anova automatically.
cut_gruop_test<-function(dat,cut_variable,interval,
                         formula_str,sample_var='id',compute_cummulative=FALSE,
                         method="auto"){
  formula_varName<-names(get_all_vars(formula_str,dat))
  if(length(formula_varName)>=3&compute_cummulative) method<-"anova"
  if(method=="anova"&!str_detect(formula_str,cut_variable)){
    formula_str<-formula_str %>% str_replace('~',paste0('~',cut_variable,'+'))
    formula_varName<-names(get_all_vars(formula_str,dat))
  }
  y_var<-formula_varName[1]
  if(compute_cummulative){
    grouby_var<-c(formula_varName[c(-1,-2)],sample_var)
    dat<-dat %>% group_by_at(grouby_var) %>% mutate_at(y_var,funs(cum=cumsum)) %>% 
      dplyr::arrange(.by_group=T) %>% rename_at("cum",function(x) paste(y_var,x,sep = '_')) %>%
      ungroup()
    formula_str<-str_replace(formula_str,y_var,paste0(y_var,'_cum'))
    formula_varName<-names(get_all_vars(formula_str,dat))
  }
  data_cut<-cut_data_continous(dat,cut_variable,interval)
  plt_dat<-dat %>% group_by_at(c(cut_variable,formula_varName[-1])) %>%
    summarise_at(formula_varName[1],funs(mu=mean,se=std.error,ymin=mu-se,ymax=mu+se)) %>%
    ungroup()
  group_test(data_cut,cut_variable,formula_str,test_method=method,plot_type = "line_errorbar",
             plot_data = plt_dat)
}
cut_data_continous <- function(d, v, delta) {
  d %>% dplyr::filter(as.character(eval(sym(v))) %in% as.character(seq(0, max(eval(
    sym(v))), delta)))
}
p_to_signif<-. %>%
  cut(.,breaks = c(0,0.001,0.01,0.05,1),labels = c('***','**','*','')) %>% levels(.)[.]
factor_to_numeric<-.%>% levels(.)[.] %>% as.numeric()

stat_signif_data<-function(result_tb,ref_tb,group_var,line_orient="horizontal",
                           ref_tb_aggregated=T,y_var=NA_character_,tip_length=0,
                           hv_adjust=0,show_ns=FALSE,has_line=TRUE){
  if(line_orient=="vertical"){
    tip_length<-range(ref_tb[[group_var]]) %>% {.[2]-.[1]} %>% {./80}
    result_tb<-result_tb %>% select_at(vars(term, comparison, adj.p.value))
    is_multi_term<-result_tb$term %>% str_detect(':') %>% any()
    creat_vertical_line<-function(x){
      if(is_multi_term){
        p12<-ref_tb %>% rename_at(vars(c(2,3)),~c("V2","V3")) %>% semi_join(x,by=c("V2","V3")) %>% 
          rename_at(vars(2:3),~names(ref_tb)[2:3]) %>% filter_at(group_var,all_vars(.==max(.)))
        p34<-ref_tb %>% rename_at(vars(c(2,3)),~c("V4","V5")) %>% semi_join(x,by=c("V4","V5")) %>% 
          rename_at(vars(2:3),~names(ref_tb)[2:3]) %>% filter_at(group_var,all_vars(.==max(.)))
      }
      else {
        p12<-ref_tb %>% rename_at(vars(2),~"V2") %>% semi_join(x,by="V2") %>%
          rename_at(vars(2),~names(ref_tb)[2]) %>% filter_at(group_var,all_vars(.==max(.)))
        p34<-ref_tb %>% rename_at(vars(2),~"V3") %>% semi_join(x,by="V3") %>%
          rename_at(vars(2),~names(ref_tb)[2]) %>% filter_at(group_var,all_vars(.==max(.)))
      }
      bind_rows(p12,p12,p34,p34) %>% dplyr::rename(x=!!sym(group_var),y=mu) %>%
        select_at(c("x","y")) %>% mutate_at("x",funs(.+c(0,tip_length,tip_length,0)))
    }
    if(is_multi_term) {
        ref_tb<-mutate_if(ref_tb,is.factor,as.character)
        signif_tb<-result_tb %>% filter(str_detect(term,':')) %>% .[["comparison"]] %>%
          str_match("([A-Za-z+-]+):([A-Za-z+-]+)-([A-Za-z+-]+):([A-Za-z+-]+)") %>% as_tibble() %>% 
          filter(V2==V4|V3==V5) %>% inner_join(result_tb,.,by=c("comparison"="V1")) %>% {
            term_group<-.[["term"]] %>% str_split(':',simplify = T) %>% as_tibble()
            dplyr::mutate(.,comparison_key=if_else(V2==V4,term_group$V1,term_group$V2))
          } %>% dplyr::mutate(comparison_value=if_else(V2==V4,V2,V3)) %>% 
            spread(comparison_key,comparison_value) %>% nest(cols=c(comparison,V2:V5)) %>%
          mutate_at("data",map,creat_vertical_line) %>% rowid_to_column(var = "line_id") %>%
          dplyr::mutate(signif_txt=p_to_signif(adj.p.value))
    }
    else
      signif_tb<-result_tb %>% .[["comparison"]] %>% 
        str_match("([A-Za-z+-]+)-([A-Za-z+-]+)") %>%
        as_tibble() %>% inner_join(result_tb,.,by=c("comparison"="V1")) %>%
        nest(cols=c(comparison,V2:V3)) %>% mutate_at("data",map,creat_vertical_line) %>% 
        rowid_to_column(var = "line_id") %>%
        dplyr::mutate(signif_txt=p_to_signif(adj.p.value))
    signif_level<-signif_tb %>% 
      dplyr::mutate(x=map(data,~max(.$x)+hv_adjust),y=map(data,~mean(.$y)+hv_adjust)) %>%
      select_at(vars(-data)) %>% unnest()
    signif_line<-signif_tb %>% unnest()
    if(show_ns) signif_level<-signif_level %>% mutate_at("signif_txt",funs(if_else(.=="","NS",.)))
    else signif_line<-signif_line %>% filter_at("signif_txt",all_vars(.!=""))
    list(signif_line=signif_line,
         signif_level=signif_level) %>% return()
  } else {
    if(!has_line) {
      signif_level_tb<-ref_tb %>% group_by_at(group_var) %>%
        dplyr::summarise(y=max(ymax)+tip_length+hv_adjust) %>%
        left_join(result_tb,.,by=group_var) %>% dplyr::mutate(signif_txt=p_to_signif(p.value))
      if(show_ns) signif_level_tb<-signif_level_tb %>%
        mutate_at("signif_txt",any_vars(if_else(.=="","NS",.)))
      return(signif_level_tb)
    }
    else{
      tip_length<-range(ref_tb[[y_var]]) %>% {.[2]/60}
      creat_signif_line<-. %>% .[c(2,2,3,3,1,1,1,1)] %>% matrix(4,2) %>% as_tibble() %>% 
        dplyr::rename(x=V1,y=V2) %>% unnest() %>% 
        mutate_at("y",funs(.+c(-tip_length,0,0,-tip_length)))
      if(ref_tb_aggregated) signif_tb<-ref_tb %>% group_by_at(group_var) %>% 
          dplyr::summarise(y=max(ymax)+tip_length+hv_adjust) 
      else signif_tb<-ref_tb %>% group_by_at(group_var) %>% 
          dplyr::summarise(y=max(!!sym(y_var))+tip_length+hv_adjust)
      signif_tb<-signif_tb %>%
          dplyr::mutate(xmin=1:nrow(.)-.2,xmax=1:nrow(.)+.2) %>%
          left_join(result_tb,by=group_var) %>% mutate_if(is.factor,as.character) %>%
          nest(cols=y:xmax,.key = "pos") %>% mutate_at("pos",funs(map(.,creat_signif_line))) %>% 
          rowid_to_column(var = "line_id") %>% 
          dplyr::mutate(signif_txt=p_to_signif(p.value)) %>% unnest()
      signif_level_tb<-signif_tb %>% nest(x,y) %>% 
        mutate_at("data",funs(map(.,dplyr::summarise,x=mean(x),y=max(y)))) %>% unnest()
      if(show_ns) signif_level_tb<-signif_level_tb %>%
        mutate_at("signif_txt",any_vars(if_else(.=="","NS",.)))
      else signif_tb<-signif_tb %>% filter_at("signif_txt",any_vars(.!=""))
      list(signif_line=signif_tb,signif_level=signif_level_tb)
    }
  }
}

get_formula_varnames<-function(formula_str,data) names(get_all_vars(formula_str,data))

remove_geom <- function(ggplot2_object, geom_type) {
  # Delete layers that match the requested type.
  layers <- lapply(ggplot2_object$layers, function(x) {
    if (class(x$geom)[1] == geom_type) {
      NULL
    } else {
      x
    }
  })
  # Delete the unwanted layers.
  layers <- layers[!sapply(layers, is.null)]
  ggplot2_object$layers <- layers
  ggplot2_object
}
get_layer_id<-function(fig,geom){
  for(i_layer in 1:length(fig$layers)){
    if(class(fig$layer[[i_layer]]$geom)[1]==geom) return(i_layer)
  }
}

first_letter_to_upper <- function(x) {
  substr(x, 1, 1) <- toupper(substr(x, 1, 1))
  x
}

bin_continuous<-function(s,bin,bin_position="middle") {
  if(bin_position=="middle") (s-min(s)) %/% bin *bin+min(s)+bin/2
  else if(bin_position=="left") (s-min(s)) %/% bin *bin+min(s)
  else if(bin_position=="right") (s-min(s)) %/% bin *bin+min(s)+bin
}
