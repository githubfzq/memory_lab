test<-tibble(
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
             "volume depth"),
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
              figs.volume_depth),
  size=list(c(40,30),
            c(40,30),
            c(40,30),
            c(40,40),
            c(40,40),
            c(45,35),
            c(45,35),
            c(40,35),
            c(40,35),
            c(40,30),
            c(40,30),
            c(40,30),
            c(40,30),
            c(40,30))
)
pwalk(test,~export_images(..1,..2[[1]],..3[1],..3[2]))


export_required_images<-function(){
  export_images("length total",figs.length[[1]],40,30)
  for(id in 1:4) export_images(paste("length",cluster_for_file_name)[id],
                               figs.length[[id+1]],40,30)
  export_images("area total",figs.area[[1]],40,30)
  for(id in 1:4) export_images(paste("area",cluster_for_file_name)[id],
                               figs.area[[id+1]],40,30)
  export_images("volume total",figs.volume[[1]],40,30)
  for (id in 1:4) export_images(paste("volume",cluster_for_file_name)[id],
                                figs.volume[[id+1]],40,30)
  export_images("cumulative Sholl apical and basal",figs.sholl_cum_nolabel[[1]],
                40,40)
  export_images("cumulative Sholl apical or basal",figs.sholl_cum_label$figure_facet2[[1]],
                40,40)
  export_images("Sholl apical and basal",figs.sholl_nolabel[[1]],45,35)
  export_images("Sholl apical or basal",figs.sholl_label[[1]],45,35)
  export_images("Sholl apical and basal with normalized",figs.norm_cum_nolabel[[1]],45,35)
  
  export_images("length with depth",figs.length_depth[[1]],40,30,scale = 2)
  for(id in 1:4) export_images(paste("length with depth",cluster_for_file_name)[id],
                               figs.length_depth[[id+1]],40,30)
  export_images("area with depth",figs.area_depth[[1]],40,30)
  for(id in 1:4) export_images(paste("area with depth",cluster_for_file_name)[id],
                               figs.area_depth[[id+1]],40,30)
  export_images("volume with depth",figs.volume_depth[[1]],40,30)
  for(id in 1:4) export_images(paste("volume with depth",cluster_for_file_name)[id],
                               figs.volume_depth[[id+1]],40,30)
}