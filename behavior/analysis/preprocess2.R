# -*- coding: utf-8 -*-
# ####### Some useful functions to preprocess the stop signal task result ####

######## This function is used to preprocess the data obtained from ###################
######## stop signal training program which is a self-written python software -- StopSignal Task v2.0 (blink)  ###################
######## ONe parameter is needed --- the txt file produced by the software above  ################
library(coin)

loadData = function(filename){
  con = file(filename, encoding = 'utf-8')
  result = readLines(con)
  close(con)
  info=result[2]
  print(info)
  ## save data
  for(i in seq(3, length(result)-1, 2)){
    event  = tolower(result[i])
    event = gsub(' ', '', event)#去掉空格
    data = result[i+1]
    if(data == '[]'){
      assign(event, NA)
      next
    }
    if(event=='whoknows'&data != '[]'){
      print('Warning!!!!!! Some trials may be missed !!!!!')
      print(data)
      next
    }
    data = strsplit(data, ',')
    data = data[[1]]
    data[1] = substring(data[1],2)
    data[length(data)] = substring(data[length(data)],1,nchar(data[length(data)])-1)
    data = as.double(data)
    assign(event, data)
  }
 
  # the training program interruptted in an ongoing trial, remove the last unfinished trial.
  L1 = length(pokeinl)
  L3 = length(pokeinm)
  if(L1>L3){
    pokeinl = pokeinl[1:L3]
    pokeoutl = pokeoutl[1:L3]
    pokeinr = pokeinr[1:L3]
    pokeoutr = pokeoutr[1:L3]
    trialtype = trialtype[1:L3]
  }
  
  ####length check####
  l=c(length(pokeinl),length(pokeoutl),length(pokeinm),#length(pokeOutM),
      length(pokeinr),length(pokeoutr),length(rewardstart),
      length(trialtype),length(isrewarded))
    
  if(mean(l)==l[1]){
    #### construct returned data frame ######
    returnData = data.frame(pokeInL = pokeinl, pokeOutL = pokeoutl, pokeInM = pokeinm, #pokeOutM = pokeOutM,
                            pokeInR = pokeinr, pokeOutR = pokeoutr, rewardStart = rewardstart, 
                            trialType = trialtype, isRewarded = isrewarded, 
                            stopSignalStart = rep(0,length(pokeinl)), 
                            SSDs = rep(0,length(pokeinl)), trialsSkipped = rep(0,length(pokeinl)))
    if(!is.na(trialsskipped[1])){
      returnData$trialType[trialsskipped]=1
      returnData$trialsSkipped[1:length(trialsskipped)] = trialsskipped
    }
    returnData$SSDs[returnData$trialType == 2] = ssds
    returnData$stopSignalStart[returnData$trialType==2] = stopsignalstart
    print('DataFrame Returned')
  }else{
    returnData = list(pokeInL = pokeinl, pokeOutL = pokeoutl, pokeInM = pokeinm, pokeOutM = pokeoutm,
                      pokeInR = pokeinr, pokeOutR = pokeoutr, rewardStart = rewardstart,
                      trialType = trialtype, isRewarded = isrewarded, 
                      stopSignalStart =  stopsignalstart, 
                      SSDs = ssds, trialsSkipped = trialsskipped)
    print('List Returned')
  }
  return(returnData)
}





#####Accuracy Calculation#####
calCorRate=function(data){
  rat.data=data.frame(type=data$trialType,correct=data$isRewarded)
  type1=rat.data[rat.data$type==1,'correct']
  type2=rat.data[rat.data$type==2,'correct']
  type1.correct=length(which(type1==1))
  type2.correct=length(which(type2==1))
  type1.cr=round(type1.correct/length(type1),2)
  type2.cr=round(type2.correct/(length(type2)+0.001),2)
  return(c(type1.cr,type2.cr))
}

### Go RT calculation#####
calGoRT = function(data){
  current.go = data[data$trialType==1,]
  n=which(current.go$isRewarded==1)[1]
  if(!is.na(n)){
    if((current.go[n,'pokeInL']-current.go[n, 'pokeInR'])>0){
      goRT = current.go[current.go$isRewarded == 1, 'pokeInL']-current.go[current.go$isRewarded == 1,'pokeOutR']
    }else{
      goRT = current.go[current.go$isRewarded == 1, 'pokeInR']-current.go[current.go$isRewarded == 1,'pokeOutL']
    }
   return(goRT)
  }
}

####SEM && SD CALCULATION FUNCTION####
sem = function(data){
  if(is.null(dim(data))){
    n = length(data)
    s = sd(data)
    return(s/sqrt(n))
  }else{
    sems = c()
    for(i in 1:dim(data)[2]){
      n = length(data[,i])
      s = sd(data[,i])
      sems=c(sems,s/sqrt(n))
    }
    return(sems)
  }
}



####SSRT CALCULATION#####
calSSRT = function(data, baseline=20, end=120){
  if(dim(data)[1]<end){
    end=dim(data)[1]
  }
  if(end>baseline){
    data=data[(baseline+1):end,]
  }else{
    print('Arguments Error! END should be bigger than BASELINE!')
  }
  goTrial = data[data$trialType == 1,]
  n=which(goTrial$isRewarded==1)[1]
  if(!is.na(n)){
    if((goTrial[n,'pokeOutL']-goTrial[n,'pokeOutR'])>0){
      goRT = goTrial[goTrial$isRewarded == 1, 'pokeInL']-goTrial[goTrial$isRewarded == 1,'pokeOutR']
    }else{
      goRT = goTrial[goTrial$isRewarded == 1, 'pokeInR']-goTrial[goTrial$isRewarded == 1,'pokeOutL']
    }
    SSD = data$SSDs[data$SSDs>0]
    stopAccuracy=calCorRate(data)[2]
    #ssrt=round(median(goRT)-mean(SSD),2)
    if(stopAccuracy<1){
      ssrt=round(sort(goRT)[ceiling(length(goRT)*(1-stopAccuracy))]-mean(SSD),2)
    }else{
      ssrt=0
    }
    return(list(gort=goRT, ssd=SSD, mrt=mean(goRT), ssrt=ssrt, 
                cr=calCorRate(data),stop_num=length(which(data$stopSignalStart>0))))
  }else{
    return(list(gort=0, ssd=0, mrt=0, ssrt=0, 
                cr=c(0,0),stop_num=0))
  }
}

calSSRT2 = function(data, baseline=20,block_num=5,block_length=60,correct=F){
  if(dim(data)[1]<block_num*block_length+baseline){
    print('Data integrity check failed!')
    while(block_num>0){
      block_num=block_num-1
      if(dim(data)[1]>block_num*block_length+baseline){
        break
      }
    }
  }else if(dim(data)[1]<baseline+block_length){
    return
  }
  # data selection and subdivision
  data=data[(baseline+1):(baseline+block_num*block_length),]
  blocked_ssrt=0
  block_ssrts=c()
  good_blocks_num=seq(block_num)
  for(i in 1:block_num){
    subdata=data[((i-1)*block_length+1):(i*block_length),]
    result=calSSRT(subdata,baseline=0,end=block_length)
#     allcrgo<<-c(allcrgo,result$cr[1])
#     allcrstop<<-c(allcrstop,result$cr[2])
#     allstop_num<<-c(allstop_num,result$stop_num)
#     if(result$stop_num>10){
#       crazystopblock<<-c(crazystopblock,list(subdata))
#     }
    if(correct){
      if(result$cr[2]<0.4 | result$cr[2]>0.8 | result$cr[1]<0.8 | result$ssrt<=80 | result$stop_num<=6){
        good_blocks_num=setdiff(good_blocks_num,i)
      }
    }
    block_ssrts=c(block_ssrts, result$ssrt)
  }
  print(block_ssrts)
  print(good_blocks_num)
  if(length(good_blocks_num)>0){
    if(correct){
      newdata=data[1,]
      newid=good_blocks_num
      for(i in newid){
        newdata=rbind(newdata,data[((i-1)*block_length+1):(i*block_length),])
      }
      newdata=newdata[-1,]
     
      result=calSSRT(newdata,baseline=0,end=dim(newdata)[1])
      result$blocked_ssrt=mean(block_ssrts[good_blocks_num])
    }else{
      result=calSSRT(data,baseline=0,end=block_num*block_length)
      result$blocked_ssrt=mean(block_ssrts)
    }
  }else{
    print('NO GOOD BLOCKS!!!')
    result=calSSRT(data,baseline=0,end=block_num*block_length)
    result$blocked_ssrt=result$ssrt
  }
  result$all_block_ssrts = block_ssrts
  #remove ssrts calculated from blocks unqaulified.
  if(length(good_blocks_num)<block_num){
    bad_blocks_num = setdiff(seq(block_num), good_blocks_num)
    result$all_block_ssrts[bad_blocks_num] = -1
  }
  
  print(result$blocked_ssrt)
  return(result)
}

#function to get preliminary analysis result returned in a dataframe
getRDF2=function(directory, baseline=20,blocked=T, block_num=3, block_length=100, correct=T,
                 remove_ssrt_smaller_than_threhold = 50){
  
  # file names start with "_ex_" should be exculded from analysis.
  name=dir(directory, pattern = '^[^_ex_]')
  
  ssrt=data.frame(ratid=substr(name,1,nchar(name)-4),
                  SSRT=rep(0,length(name)))
  mrt=c()
  crgo=c()
  crstop=c()
  for(i in 1:length(name)){
    b=loadData(paste(directory,name[i],sep=''))
    if(class(b)=='data.frame'){
      if(length(baseline)>1){
        br=calSSRT2(b, baseline[i], block_num, block_length, correct=correct)
       
      }else{
        br=calSSRT2(b, baseline, block_num, block_length, correct=correct)
      }
    }else{
      print('!!!!!!!!!DATA IS NOT DATA.FRAME!!!!!!!!!!')
      br = list(blocked_ssrt=0,mrt=0,cr=c(0,0))
    }
    
    ssrt[i,2]=br$blocked_ssrt
    mrt=c(mrt, br$mrt)
    crgo=c(crgo, br$cr[1])
    crstop=c(crstop, br$cr[2])
    
  }
  returned_data_frame = cbind(ssrt,data.frame(mrt=mrt,crgo=crgo,crstop=crstop))
  if(remove_ssrt_smaller_than_threhold>0){
    return(returned_data_frame[returned_data_frame$SSRT>remove_ssrt_smaller_than_threhold, ])
  }
  else{
    return(returned_data_frame)
  }
}


getRDF3=function(directory, baseline=20,blocked=T, block_num=3, block_length=100, correct=T){
  # In this function, all SSRTs from each block are also returned.
  
  # file names start with "_ex_" should be exculded from analysis.
  name=dir(directory, pattern = '^[^_ex_]')
  
  ssrt=data.frame(ratid=substr(name,1,nchar(name)-4),
                  SSRT=rep(0,length(name)))
  mrt=c()
  crgo=c()
  crstop=c()
  ssrtColumnNames = c()
  for(i in 1:block_num){
    columnName = paste('ssrt',i,sep = '')
    assign(columnName, c())
    ssrtColumnNames = c(ssrtColumnNames, columnName)
  }
  for(i in 1:length(name)){
    b=loadData(paste(directory,name[i],sep=''))
    if(class(b)=='data.frame'){
      if(length(baseline)>1){
        br=calSSRT2(b, baseline[i], block_num, block_length, correct=correct)
      }else{
        br=calSSRT2(b, baseline, block_num, block_length, correct=correct)
      }
    }else{
      print('!!!!!!!!!DATA IS NOT DATA.FRAME!!!!!!!!!!')
      br = list(blocked_ssrt=0,mrt=0,cr=c(0,0))
    }
    
    ssrt[i,2]=br$blocked_ssrt
    mrt=c(mrt, br$mrt)
    crgo=c(crgo, br$cr[1])
    crstop=c(crstop, br$cr[2])
    for(i in 1:block_num){
      temp = eval(parse(text = ssrtColumnNames[i]))
      assign(ssrtColumnNames[i], c(temp, br$all_block_ssrts[i]))
    }
  }
  blockSSRTDataFrame = data.frame(matrix(ncol = block_num, nrow = length(name)))
  colnames(blockSSRTDataFrame) = ssrtColumnNames
  for(i in 1:block_num){
    blockSSRTDataFrame[,ssrtColumnNames[i]] = eval(parse(text = ssrtColumnNames[i]))
  }
  return(cbind(ssrt,cbind(data.frame(mrt=mrt,crgo=crgo,crstop=crstop), blockSSRTDataFrame)))
}



# Plot SSRT Evolution
plotEvolution = function(data, title='SSRT Evolution', treatment=NULL){
  require(reshape2)
  require(ggplot2)
  data$id=seq_along(data[,1])
  if(is.null(data$treatment)){
    if(!is.null(treatment)){
      data$treatment=treatment
      df = melt(data[,-1], id.var=c('id','treatment'))
      p=ggplot(df,aes(variable,value,group=id,color=treatment))
    }else{
      df = melt(data[,-1], id.var='id')
      p=ggplot(df,aes(variable,value,group=id))
    }
  }else{
    df = melt(data[,-1], id.var=c('id','treatment'))
    p=ggplot(df,aes(variable,value,group=id,color=treatment))
  }
  p+geom_path()+
    ggtitle(title)+xlab('')+ylab('SSRT (ms)')+
    theme(axis.title = element_text(size = rel(2)), 
          axis.text = element_text(size = 20, color='black'),
          axis.ticks = element_line(size=rel(2), color='black'),
          axis.ticks.length = unit(.3, 'cm'),
          plot.title = element_text(size = rel(2.5)))
}


reshapeMyData = function(data,treatment_title,median_divide = T, middle_value=150){
  #Transform the data to a data.frame adjusted for plotting.
  if(dim(data)[2]!=10){
    print('Data is NOT Complete!!')
    return()
  }
  if(median_divide){
    median_ssrt = median(data$SSRT)
  }else{
    median_ssrt=middle_value
  }
  # rename columns
  colnames(data) = c('ratid','SSRT','mrt','crgo','crstop',
                     'ratid.1','SSRT.1','mrt.1','crgo.1','crstop.1')
  small_baseline = data[data$SSRT<median_ssrt,]
  big_baseline = data[data$SSRT>=median_ssrt,]
  df = data.frame(ssrt = c(mean(small_baseline$SSRT),mean(small_baseline$SSRT.1),
                           mean(big_baseline$SSRT),mean(big_baseline$SSRT.1)),
                  ssrt_p = c(t.test(small_baseline$SSRT,small_baseline$SSRT.1,paired=T,var.equal = T)$p.value,
                             t.test(big_baseline$SSRT,big_baseline$SSRT.1,paired=T,var.equal = T)$p.value,
                             t.test(small_baseline$SSRT,big_baseline$SSRT,var.equal = T)$p.value,
                             t.test(small_baseline$SSRT.1,big_baseline$SSRT.1,var.equal = T)$p.value),
                  ssrt_se = c(sem(small_baseline$SSRT),sem(small_baseline$SSRT.1),
                              sem(big_baseline$SSRT),sem(big_baseline$SSRT.1)),
                  mrt = c(mean(small_baseline$mrt),mean(small_baseline$mrt.1),
                          mean(big_baseline$mrt),mean(big_baseline$mrt.1)),
                  mrt_p = c(t.test(small_baseline$mrt, small_baseline$mrt.1,paired=T,var.equal = T)$p.value,
                            t.test(big_baseline$mrt, big_baseline$mrt.1,paired=T,var.equal = T)$p.value,
                            t.test(small_baseline$mrt,big_baseline$mrt,var.equal = T)$p.value,
                            t.test(small_baseline$mrt.1,big_baseline$mrt.1,var.equal = T)$p.value),
                  mrt_se = c(sem(small_baseline$mrt),sem(small_baseline$mrt.1),
                             sem(big_baseline$mrt),sem(big_baseline$mrt.1)),
                  crgo = c(mean(small_baseline$crgo),mean(small_baseline$crgo.1),
                           mean(big_baseline$crgo),mean(big_baseline$crgo.1)),
                  crgo_p = c(t.test(small_baseline$crgo, small_baseline$crgo.1,paired=T,var.equal = T)$p.value,
                             t.test(big_baseline$crgo,big_baseline$crgo.1,paired=T,var.equal = T)$p.value,
                             t.test(small_baseline$crgo,big_baseline$crgo,var.equal = T)$p.value,
                             t.test(small_baseline$crgo.1,big_baseline$crgo.1,var.equal = T)$p.value),
                  crgo_se = c(sem(small_baseline$crgo),sem(small_baseline$crgo.1),
                              sem(big_baseline$crgo),sem(big_baseline$crgo.1)),
                  crstop = c(mean(small_baseline$crstop),mean(small_baseline$crstop.1),
                             mean(big_baseline$crstop),mean(big_baseline$crstop.1)),
                  crstop_p = c(t.test(small_baseline$crstop, small_baseline$crstop.1,paired=T,var.equal = T)$p.value,
                               t.test(big_baseline$crstop, big_baseline$crstop.1,paired=T,var.equal = T)$p.value,
                               t.test(small_baseline$crstop,big_baseline$crstop,var.equal = T)$p.value,
                               t.test(small_baseline$crstop.1,big_baseline$crstop.1,var.equal = T)$p.value),
                  crstop_se = c(sem(small_baseline$crstop),sem(small_baseline$crstop.1),
                                sem(big_baseline$crstop),sem(big_baseline$crstop.1)),
                  treatments = factor(c('Baseline',treatment_title,'Baseline',treatment_title),levels=c('Baseline',treatment_title)),
                  Groups = factor(c('Fast Responders','Fast Responders','Slow Responders','Slow Responders'))
  )
  return(df)
}

# Reshape data2
reshapeMyData2 = function(data,treatment_title,sub_group_index=NA, median_divide = T, middle_value=150){
  ##Transform the data to a data.frame adjusted for plotting.
  ## 'treatment_title' is the treatment used as a factor in the returned dataframe
  ## 'sub_group_index' is list of row indexes indicate the first elements of subgroups and 
  # median divide all the subgroups and then combine all the samll groups to be final small group,
  # all the big groups to be the final big group.
  # eg: the #rows of data is 100. sub_group_index=c(10,30,50), then subgroups are [1:9,], [10:29,], [30:49,], [50:100,]
  if(dim(data)[2]!=10){
    print('Data is NOT Complete!!')
    return()
  }
  
  # rename columns  
  colnames(data) = c('ratid','SSRT','mrt','crgo','crstop',
                     'ratid.1','SSRT.1','mrt.1','crgo.1','crstop.1')
  if(is.na(sub_group_index)){
    if(median_divide){
      median_ssrt = median(data$SSRT)
    }else{
      median_ssrt=middle_value
    }
    small_baseline = data[data$SSRT<median_ssrt,]
    big_baseline = data[data$SSRT>=median_ssrt,]
  }else{
    small_baseline = data[1,]
    big_baseline = data[1,]
    sub_group_index = c(sub_group_index, dim(data)[1])
    count = 1
    for(i in sub_group_index){
      if(count==1){
        sub_group=data[1:i-1,]
      }else if(count == length(sub_group_index)){
        sub_group = data[sub_group_index[count-1]:i,]
      }else{
        sub_group = data[sub_group_index[count-1]:i-1,]
      }
      count = count+1
      median_ssrt = median(sub_group$SSRT)
      small_baseline = rbind(small_baseline, sub_group[sub_group$SSRT<median_ssrt,])
      big_baseline = rbind(big_baseline, sub_group[sub_group$SSRT>=median_ssrt,])
    }
    small_baseline = small_baseline[-1,]
    big_baseline = big_baseline[-1,]
  }
  
  df = data.frame(ssrt = c(mean(small_baseline$SSRT),mean(small_baseline$SSRT.1),
                           mean(big_baseline$SSRT),mean(big_baseline$SSRT.1)),
                  ssrt_p = c(t.test(small_baseline$SSRT,small_baseline$SSRT.1,paired=T,var.equal = T)$p.value,
                             t.test(big_baseline$SSRT,big_baseline$SSRT.1,paired=T,var.equal = T)$p.value,
                             t.test(small_baseline$SSRT,big_baseline$SSRT,var.equal = T)$p.value,
                             t.test(small_baseline$SSRT.1,big_baseline$SSRT.1,var.equal = T)$p.value),
                  ssrt_se = c(sem(small_baseline$SSRT),sem(small_baseline$SSRT.1),
                              sem(big_baseline$SSRT),sem(big_baseline$SSRT.1)),
                  mrt = c(mean(small_baseline$mrt),mean(small_baseline$mrt.1),
                          mean(big_baseline$mrt),mean(big_baseline$mrt.1)),
                  mrt_p = c(t.test(small_baseline$mrt, small_baseline$mrt.1,paired=T,var.equal = T)$p.value,
                            t.test(big_baseline$mrt, big_baseline$mrt.1,paired=T,var.equal = T)$p.value,
                            t.test(small_baseline$mrt,big_baseline$mrt,var.equal = T)$p.value,
                            t.test(small_baseline$mrt.1,big_baseline$mrt.1,var.equal = T)$p.value),
                  mrt_se = c(sem(small_baseline$mrt),sem(small_baseline$mrt.1),
                             sem(big_baseline$mrt),sem(big_baseline$mrt.1)),
                  crgo = c(mean(small_baseline$crgo),mean(small_baseline$crgo.1),
                           mean(big_baseline$crgo),mean(big_baseline$crgo.1)),
                  crgo_p = c(t.test(small_baseline$crgo, small_baseline$crgo.1,paired=T,var.equal = T)$p.value,
                             t.test(big_baseline$crgo,big_baseline$crgo.1,paired=T,var.equal = T)$p.value,
                             t.test(small_baseline$crgo,big_baseline$crgo,var.equal = T)$p.value,
                             t.test(small_baseline$crgo.1,big_baseline$crgo.1,var.equal = T)$p.value),
                  crgo_se = c(sem(small_baseline$crgo),sem(small_baseline$crgo.1),
                              sem(big_baseline$crgo),sem(big_baseline$crgo.1)),
                  crstop = c(mean(small_baseline$crstop),mean(small_baseline$crstop.1),
                             mean(big_baseline$crstop),mean(big_baseline$crstop.1)),
                  crstop_p = c(t.test(small_baseline$crstop, small_baseline$crstop.1,paired=T,var.equal = T)$p.value,
                               t.test(big_baseline$crstop, big_baseline$crstop.1,paired=T,var.equal = T)$p.value,
                               t.test(small_baseline$crstop,big_baseline$crstop,var.equal = T)$p.value,
                               t.test(small_baseline$crstop.1,big_baseline$crstop.1,var.equal = T)$p.value),
                  crstop_se = c(sem(small_baseline$crstop),sem(small_baseline$crstop.1),
                                sem(big_baseline$crstop),sem(big_baseline$crstop.1)),
                  treatments = factor(c('Baseline',treatment_title,'Baseline',treatment_title),levels=c('Baseline',treatment_title)),
                  Groups = factor(c('Fast Group','Fast Group','Slow Group','Slow Group'))
  )
  return(df)
}




reshapeMyData3 = function(data,treatment_title,sub_group_index=NA, median_divide = T, middle_value=150){
  ##  Wilcoxon Signed Rank Test substitute paired t-test
  if(dim(data)[2]!=10){
    print('Data is NOT Complete!!')
    return()
  }
  
  # rename columns  
  colnames(data) = c('ratid','SSRT','mrt','crgo','crstop',
                     'ratid.1','SSRT.1','mrt.1','crgo.1','crstop.1')
  if(is.na(sub_group_index)){
    if(median_divide){
      median_ssrt = median(data$SSRT)
    }else{
      median_ssrt=middle_value
    }
    small_baseline = data[data$SSRT<median_ssrt,]
    big_baseline = data[data$SSRT>=median_ssrt,]
  }else{
    small_baseline = data[1,]
    big_baseline = data[1,]
    sub_group_index = c(sub_group_index, dim(data)[1])
    count = 1
    for(i in sub_group_index){
      if(count==1){
        sub_group=data[1:i-1,]
      }else if(count == length(sub_group_index)){
        sub_group = data[sub_group_index[count-1]:i,]
      }else{
        sub_group = data[sub_group_index[count-1]:i-1,]
      }
      count = count+1
      median_ssrt = median(sub_group$SSRT)
      small_baseline = rbind(small_baseline, sub_group[sub_group$SSRT<median_ssrt,])
      big_baseline = rbind(big_baseline, sub_group[sub_group$SSRT>=median_ssrt,])
    }
    small_baseline = small_baseline[-1,]
    big_baseline = big_baseline[-1,]
  }
  
  df = data.frame(ssrt = c(mean(small_baseline$SSRT),mean(small_baseline$SSRT.1),
                           mean(big_baseline$SSRT),mean(big_baseline$SSRT.1)),
                  ssrt_p = c(wilcox.test(small_baseline$SSRT,small_baseline$SSRT.1,paired=T)$p.value,
                             wilcox.test(big_baseline$SSRT,big_baseline$SSRT.1,paired=T)$p.value,
                             wilcox.test(small_baseline$SSRT,big_baseline$SSRT)$p.value,
                             wilcox.test(small_baseline$SSRT.1,big_baseline$SSRT.1)$p.value),
                  ssrt_se = c(sem(small_baseline$SSRT),sem(small_baseline$SSRT.1),
                              sem(big_baseline$SSRT),sem(big_baseline$SSRT.1)),
                  mrt = c(mean(small_baseline$mrt),mean(small_baseline$mrt.1),
                          mean(big_baseline$mrt),mean(big_baseline$mrt.1)),
                  mrt_p = c(wilcox.test(small_baseline$mrt, small_baseline$mrt.1,paired=T)$p.value,
                            wilcox.test(big_baseline$mrt, big_baseline$mrt.1,paired=T)$p.value,
                            wilcox.test(small_baseline$mrt,big_baseline$mrt)$p.value,
                            wilcox.test(small_baseline$mrt.1,big_baseline$mrt.1)$p.value),
                  mrt_se = c(sem(small_baseline$mrt),sem(small_baseline$mrt.1),
                             sem(big_baseline$mrt),sem(big_baseline$mrt.1)),
                  crgo = c(mean(small_baseline$crgo),mean(small_baseline$crgo.1),
                           mean(big_baseline$crgo),mean(big_baseline$crgo.1)),
                  crgo_p = c(wilcox.test(small_baseline$crgo, small_baseline$crgo.1,paired=T)$p.value,
                             wilcox.test(big_baseline$crgo,big_baseline$crgo.1,paired=T)$p.value,
                             wilcox.test(small_baseline$crgo,big_baseline$crgo)$p.value,
                             wilcox.test(small_baseline$crgo.1,big_baseline$crgo.1)$p.value),
                  crgo_se = c(sem(small_baseline$crgo),sem(small_baseline$crgo.1),
                              sem(big_baseline$crgo),sem(big_baseline$crgo.1)),
                  crstop = c(mean(small_baseline$crstop),mean(small_baseline$crstop.1),
                             mean(big_baseline$crstop),mean(big_baseline$crstop.1)),
                  crstop_p = c(wilcox.test(small_baseline$crstop, small_baseline$crstop.1,paired=T)$p.value,
                               wilcox.test(big_baseline$crstop, big_baseline$crstop.1,paired=T)$p.value,
                               wilcox.test(small_baseline$crstop,big_baseline$crstop)$p.value,
                               wilcox.test(small_baseline$crstop.1,big_baseline$crstop.1)$p.value),
                  crstop_se = c(sem(small_baseline$crstop),sem(small_baseline$crstop.1),
                                sem(big_baseline$crstop),sem(big_baseline$crstop.1)),
                  treatments = factor(c('Baseline',treatment_title,'Baseline',treatment_title),levels=c('Baseline',treatment_title)),
                  Groups = factor(c('Fast Group','Fast Group','Slow Group','Slow Group'))
  )
  return(df)
}


# Reshape data2
reshapeMyData4.four.quarter.division = function(data,treatment_title){
  ##Transform the data to a data.frame adjusted for plotting.
  ## 'treatment_title' is the treatment used as a factor in the returned dataframe
  ##  return the first quarter and last quarter.
   if(dim(data)[2]!=10){
    print('Data is NOT Complete!!')
    return()
  }
  
  # rename columns  
  colnames(data) = c('ratid','SSRT','mrt','crgo','crstop',
                     'ratid.1','SSRT.1','mrt.1','crgo.1','crstop.1')
  quantiles = quantile(data$SSRT) 
  first.quarter = data[data$SSRT<=unname(quantiles[2]),]
  last.quarter = data[data$SSRT>=unname(quantiles[4]),]

  df = data.frame(ssrt = c(mean(first.quarter$SSRT),mean(first.quarter$SSRT.1),
                           mean(last.quarter$SSRT),mean(last.quarter$SSRT.1)),
                  ssrt_p = c(wilcox.test(first.quarter$SSRT,first.quarter$SSRT.1,paired=T)$p.value,
                             wilcox.test(last.quarter$SSRT,last.quarter$SSRT.1,paired=T)$p.value,
                             wilcox.test(first.quarter$SSRT,last.quarter$SSRT)$p.value,
                             wilcox.test(first.quarter$SSRT.1,last.quarter$SSRT.1)$p.value),
                  ssrt_se = c(sem(first.quarter$SSRT),sem(first.quarter$SSRT.1),
                              sem(last.quarter$SSRT),sem(last.quarter$SSRT.1)),
                  mrt = c(mean(first.quarter$mrt),mean(first.quarter$mrt.1),
                          mean(last.quarter$mrt),mean(last.quarter$mrt.1)),
                  mrt_p = c(wilcox.test(first.quarter$mrt, first.quarter$mrt.1,paired=T)$p.value,
                            wilcox.test(last.quarter$mrt, last.quarter$mrt.1,paired=T)$p.value,
                            wilcox.test(first.quarter$mrt,last.quarter$mrt)$p.value,
                            wilcox.test(first.quarter$mrt.1,last.quarter$mrt.1)$p.value),
                  mrt_se = c(sem(first.quarter$mrt),sem(first.quarter$mrt.1),
                             sem(last.quarter$mrt),sem(last.quarter$mrt.1)),
                  crgo = c(mean(first.quarter$crgo),mean(first.quarter$crgo.1),
                           mean(last.quarter$crgo),mean(last.quarter$crgo.1)),
                  crgo_p = c(wilcox.test(first.quarter$crgo, first.quarter$crgo.1,paired=T)$p.value,
                             wilcox.test(last.quarter$crgo,last.quarter$crgo.1,paired=T)$p.value,
                             wilcox.test(first.quarter$crgo,last.quarter$crgo)$p.value,
                             wilcox.test(first.quarter$crgo.1,last.quarter$crgo.1)$p.value),
                  crgo_se = c(sem(first.quarter$crgo),sem(first.quarter$crgo.1),
                              sem(last.quarter$crgo),sem(last.quarter$crgo.1)),
                  crstop = c(mean(first.quarter$crstop),mean(first.quarter$crstop.1),
                             mean(last.quarter$crstop),mean(last.quarter$crstop.1)),
                  crstop_p = c(wilcox.test(first.quarter$crstop, first.quarter$crstop.1,paired=T)$p.value,
                               wilcox.test(last.quarter$crstop, last.quarter$crstop.1,paired=T)$p.value,
                               wilcox.test(first.quarter$crstop,last.quarter$crstop)$p.value,
                               wilcox.test(first.quarter$crstop.1,last.quarter$crstop.1)$p.value),
                  crstop_se = c(sem(first.quarter$crstop),sem(first.quarter$crstop.1),
                                sem(last.quarter$crstop),sem(last.quarter$crstop.1)),
                  treatments = factor(c('Baseline',treatment_title,'Baseline',treatment_title),levels=c('Baseline',treatment_title)),
                  Groups = factor(c('Fast Group','Fast Group','Slow Group','Slow Group'))
  )
  return(df)
}


# Return stars of significance.
getSignificance = function(num){
  if(num>=0.05)
    return('n.s.')
  else if(num>=0.01 & num<0.05)
    return('*')
  else if(num>=0.001 & num<0.01)
    return('**')
  else if(num<0.001)
    return('***')
}

####SEM && SD CALCULATION FUNCTION####
sem = function(data){
  if(is.null(dim(data))){
    n = length(data)
    s = sd(data)
    return(s/sqrt(n))
  }else{
    sems = c()
    for(i in 1:dim(data)[2]){
      n = length(data[,i])
      s = sd(data[,i])
      sems=c(sems,s/sqrt(n))
    }
    return(sems)
  }
}



# Return stars of significance.
getSignificance = function(num){
  if(num>=0.05)
    return('n.s.')
  else if(num>=0.01 & num<0.05)
    return('*')
  else if(num>=0.001 & num<0.01)
    return('**')
  else if(num<0.001)
    return('***')
}


# Multiple plot function
#
# ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)
# - cols:   Number of columns in layout
# - layout: A matrix specifying the layout. If present, 'cols' is ignored.
#
# If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),
# then plot 1 will go in the upper left, 2 will go in the upper right, and
# 3 will go all the way across the bottom.
#
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  require(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}

ggcols <- function(n){
  # Return the first n default colors of ggplot2. Cool!
  hcl(h=seq(15, 375-360/n, length=n)%%360,c=100, l=65)
}

grid_arrange_shared_legend <- function(...) {
  plots <- list(...)
  g <- ggplotGrob(plots[[1]] + theme(legend.position="bottom"))$grobs
  legend <- g[[which(sapply(g, function(x) x$name) == "guide-box")]]
  lheight <- sum(legend$height)
  grid.arrange(
    do.call(arrangeGrob, lapply(plots, function(x)
      x + theme(legend.position="none"))),
    legend,
    ncol = 1,
    heights = unit.c(unit(1, "npc") - lheight, lheight))
}



# average data by ratid 
my_average = function(dataFrame1, dataFrame2){
  data = merge(dataFrame1, dataFrame2, by='ratid', all=T)
  for(i in 1:dim(data)[1]){
    if(!is.na(data[i,2])&!is.na(data[i,6])){
      data[i,2:5]=(data[i,2:5]+data[i,6:9])/2
    }else if(is.na(data[i,2])){
      data[i,2:5]=data[i,6:9]
    }
  }
  data = data[,1:5]
  colnames(data)=c('ratid','SSRT','mrt','crgo','crstop')
  return(data)
}


## testing for data missing.
testDataR = function(data){
  index1 = which((data$trialType==1)&(data$isRewarded==1)&(data$pokeInR==0))
  index2 = which((data$trialType==2)&(data$isRewarded==1)&(data$pokeInR>0))
  index3 = which((data$trialType==1)&(data$isRewarded==0)&(data$pokeInR>0))
  index4 = which((data$trialType==2)&(data$isRewarded==0)&(data$pokeInR==0))
  return(min(c(index1, index2, index3, index4)))
}
testDataL = function(data){
  index1 = which((data$trialType==1)&(data$isRewarded==1)&(data$pokeInL==0))
  index2 = which((data$trialType==2)&(data$isRewarded==1)&(data$pokeInL>0))
  index3 = which((data$trialType==1)&(data$isRewarded==0)&(data$pokeInL>0))
  index4 = which((data$trialType==2)&(data$isRewarded==0)&(data$pokeInL==0))
  return(min(c(index1, index2, index3, index4)))
}


############ Random Shuffle Test #########
random_shuffle_test = function(x, y, R=10000){
  statistic = mean(x-y)
  all_data = c(x,y)
  new_dist = c()
  for(i in 1:rep){
    randomx = sample(all_data, length(x))
    randomy = setdiff(all_data, randomx)
    new_dist = c(new_dist, mean(randomx-randomy))
  }
  p.value = pnorm(statistic, mean = mean(new_dist), sd = sd(new_dist))
  if(p.value>0.5){
    p.value = 1-p.value
  }
  return(p.value)
}

my_permutest <- function(data, col1 = 'SSRT', col2 = 'SSRT.1'){
  #permutation test for repeated measures.
  require(coin) 
  diff <- data[[col1]] - data[[col2]]
  len <- length(diff)
  y <- as.vector(t(cbind(abs(diff)*(diff<0), abs(diff)*(diff>=0))))
  x <- factor(rep(c('neg', 'pos'), len))
  block <- gl(len, 2)
  test <- oneway_test(y~x|block)
  return(test)
}

my_permutest2 <- function(data, col1 = 'SSRT', col2 = 'SSRT.1', R=9999){
  #permutation test for repeated measures.
  group1 = data[[col1]]
  group2 = data[[col2]]
  all = c(group1, group2)
  statistic <- mean(group1 - group2)
  len <- length(group1)
  stat.distri <- numeric(R)
  for(i in 1:R){
    index = rbinom(len, 1, 0.5)
    new.group1 = group1*index + group2*(1-index)
    new.group2 = group2*(1-index) + group2
    stat.distri[i] = mean(group1-group2)
  }
  p.value = (sum(stat.distri>statistic)+1)/(R+1)
  if(p.value > 0.5)
    p.value = 1-p.value
  return(2*p.value)
}

my_bootstrap <- function(data, col1 = 'SSRT', col2 = 'SSRT.1', R =  10000){
  l = dim(data)[1]
  combined_data = c(data[[col1]], data[[col2]])
  mean0 = mean(data[[col2]] - data[[col1]])
  diff.mean = c()
  for (i in 1:repeats){
    treat1 = sample(combined_data, l , replace = T)
    treat2 = sample(combined_data, l , replace = T)
    diff.mean = c(diff.mean, mean(treat2 - treat1))
  }
  ci = quantile(diff.mean, c(0.025, 0.975))
  return(list(statistic = mean0, distribution = diff.mean, ci = ci))
}

#####Normalization helper function########
myNormalize = function(x){
  return((x-min(x))/(max(x)-min(x)))
}

myNormalizeTotal <- function(x, total){
  return((x-min(total))/(max(total)-min(total)))
}

##### Get Text Size ######
getTextSize = function(p.value){
  if(getSignificance(p.value)=='n.s.'){
    return(7)
  }else{
    return(9)
  }
}

magic_remedy = function(data, direction='L'){
  
  if(direction=='L'){
    index1 = which((data$trialType==1)&(data$isRewarded==1)&(data$pokeInL==0))
    index2 = which((data$trialType==2)&(data$isRewarded==1)&(data$pokeInL>0))
    index3 = which((data$trialType==1)&(data$isRewarded==0)&(data$pokeInL>0))
    index4 = which((data$trialType==2)&(data$isRewarded==0)&(data$pokeInL==0))
  }else{
    index1 = which((data$trialType==1)&(data$isRewarded==1)&(data$pokeInR==0))
    index2 = which((data$trialType==2)&(data$isRewarded==1)&(data$pokeInR>0))
    index3 = which((data$trialType==1)&(data$isRewarded==0)&(data$pokeInR>0))
    index4 = which((data$trialType==2)&(data$isRewarded==0)&(data$pokeInR==0))
  }
  while(sum(index)>0){
    
  }

}

