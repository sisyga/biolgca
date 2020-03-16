path <- "C:/Users/Franzi/PycharmProjects/biolgca/saved_data/"

library(ggmuller)
library(ape)

#name der Form "name.csv"
edges_input <- function(e_name){
    e_path <- paste(path, e_name, sep="")
    edges <- read.csv(e_path, header=T)

    return (edges)
}

pop_input <- function(p_name){
    p_path <- paste(path, p_name, sep="")
    pop <- read.csv(p_path, header=T)

    return (pop)
}

create_tree <- function(e_name){
    e_path <- paste(path, e_name, sep="")
    edges <- read.csv(e_path, header=T)

    tree <- adj_matrix_to_tree(edges)
    tree$tip.label <- sort(1:length(tree$tip.label), decreasing = T)
    plot(tree, show.tip.label = TRUE, tip.color = "slateblue4")
}

magic <- function(e_name, p_name){
    #1: Daten einlesen
    edges <- edges_input(e_name)
    pop <- pop_input(p_name)

    #2: plot-Daten erstellen
    plotdata <- get_Muller_df(edges, pop)
    print(plotdata)
    print("Fuer Ausgabe des Stammbaums create_tree(e_name) aufrufen!")

    #3:plot
    return(Muller_plot(plotdata, add_legend=T, xlab = "timesteps", ylab= "relative frequency"))

}

tend <- 6616
maxfam <-7076
#e_name <- "bspedges.csv"   #
e_name <- "real180_bspedges.csv"
#l_name <- "bsplast.csv"     #
l_name <- "real180_bsplast.csv"

timesteps <- rep(0:tend, maxfam+1)
families <- rep(0:maxfam, each=tend+1)
frequencies <- read.csv(paste(path, l_name, sep=""), header=T)

edges <- read.csv(paste(path, e_name, sep=""), header=T)
pop <- data.frame(Generation=timesteps, Identity=families, Population=frequencies)

plotdata <- get_Muller_df(edges, pop)
#ZENSIEREN
Muller_plot(plotdata)

