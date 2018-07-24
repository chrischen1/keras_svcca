
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)
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

plot_history <- function(history_mat,out_file,height=1024,width=768){
  library(ggplot2)
  df_acc <- rbind.data.frame(cbind.data.frame('epoch'=1:nrow(history_mat),'acc'=history_mat$acc,'type'='acc'),
                             cbind.data.frame('epoch'=1:nrow(history_mat),'acc'=history_mat$val_acc,'type'='val_acc'))
  df_loss <- rbind.data.frame(cbind.data.frame('epoch'=1:nrow(history_mat),'loss'=history_mat$loss,'type'='loss'),
                             cbind.data.frame('epoch'=1:nrow(history_mat),'loss'=history_mat$val_loss,'type'='val_loss'))
  
  g_acc <-ggplot(df_acc, aes(x=epoch, y=acc, group=type)) + geom_line(aes(color=type))+geom_point(aes(color=type))+
    theme(axis.title.x=element_blank(),axis.text.x=element_blank(),axis.ticks.x=element_blank())
  g_loss <-ggplot(df_loss, aes(x=epoch, y=loss, group=type)) + geom_line(aes(color=type))+geom_point(aes(color=type))
  png(out_file,height,width)
  multiplot(g_acc,g_loss)
  dev.off()
}
