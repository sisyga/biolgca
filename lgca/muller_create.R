path <- "C:/Users/Franzi/PycharmProjects/biolgca/saved_data/"

library(ggmuller)
library(ape)

#name der Form "name.csv"
edges_input <- function(e_name){
    e_path <- paste(path, e_name, sep="")
    edges <- read.csv(e_path, header=T)

    return (edges)
}

trange_input <- function(t_name){
    t_path <- paste(path, t_name, sep="")
    trange <- read.csv(t_path, header=F, col.names="Generation")

    return (trange)
}

pop_input <- function(p_name){
    p_path <- paste(path, p_name, sep="")
    pop <- read.csv(p_path, header=T)

    return (pop)
}

create_tree <- function(e_name, save=False, filename="unknown"){
    e_path <- paste(path, e_name, sep="")
    edges <- read.csv(e_path, header=T)

    tree <- adj_matrix_to_tree(edges)
    tree$tip.label <- sort(1:length(tree$tip.label), decreasing = T)

    if (save==T) {
        jpeg(paste("C:/Users/Franzi/PycharmProjects/biolgca/pictures/", filename, "_tree.jpg"))
        plot(tree, show.tip.label = TRUE, tip.color = "slateblue4", cex=0.5)
        dev.off()
       }
    else {
        plot(tree, show.tip.label = TRUE, tip.color = "slateblue4")
    }
}

magic <- function(e_name, t_name, p_name, tbeg=0, tend=0){
    #1: Daten einlesen
    edges <- edges_input(e_name)
    nf <- nrow(edges)
    pop <- pop_input(p_name)
    
    # print(tend)
    if(missing(t_name)) {
        if(tend==0){
            tend <- nrow(pop) / (nf+1) - 1
            print(tend)
        }
        time <- c(rep(tbeg:tend))
        trange <- rep(time, nf+1)
        steps <- length(time)
    }
    else {
        time <- trange_input(t_name)
        trange <- rep(time[,1], nf+1)
        steps <- length(time[,1])
    }


    ids = c(rep(0, each=steps))
    for(i in edges[,2]) ids <- append(ids, c(rep(i, steps)), length(ids))

    pdata <- data.frame(Generation=trange, Identity=ids, Population=pop)


    #2: plot-Daten erstellen
    plotdata <- get_Muller_df(edges, pdata)
    print("Fuer Ausgabe des Stammbaums create_tree(e_name) aufrufen!")
    # windows()
    # platdata <- Muller_plot(plotdata, add_legend=T, xlab = "timesteps", ylab= "relative frequency")

    return(plotdata)
}

plotting <- function(data, filename="unknown"){
    #print("plotting")
    jpeg(paste("C:/Users/Franzi/PycharmProjects/biolgca/pictures/", filename, "_mp.jpg", sep=""), width=600, height=250) 
    plot(Muller_plot(data, xlab = "timesteps", ylab= "relative frequency"))
    dev.off()
}
#d <- magic(e_name="bsp_0-11_filtered 0.250_edges.csv", p_name="bsp_0-11_filtered 0.250_summed_population.csv", t_name="bsp_0-11_filtered 0.250_summed_timerange.csv")
#d <- magic(e_name="bsp_0-11_filtered 0.250_edges.csv", p_name="bsp_0-11_filtered 0.250_summed_population.csv")

# d <- magic(e_name="real180_bsp100edges.csv", p_name="real180_bsp100last.csv", tend=100)
# d <- magic(e_name="real180_bsp_0-6616_filtered 0.200_edges.csv", p_name="real180_bsp_0-6616_summed_population.csv", t_name="real180_bsp_0-6616_summed_timerange.csv")

# name <- "bsp_int_length=1_cutoff=0_ori"
# name <- "bsp_int_length=5_cutoff=0.25_ori"
# name <- "bsp_int_length=5_cutoff=0"
# name <- "bsp_int_length=5_cutoff=0.25"
# name <- "5011_0_f8684e7_int_length=250_cutoff=0_ori"
# name <- "5011_0_f8684e7_int_length=250_cutoff=0.45_ori"
# # d <- magic(e_name=paste(name, "_edges.csv", sep=""), p_name=paste(name, "_population.csv", sep=""))
# d <- magic(e_name=paste(name, "_edges.csv", sep=""), p_name=paste(name, "_population.csv", sep=""), t_name=paste(name, "_trange.csv", sep=""))
# plotting(d, filename=name)

names = c('5011_ges/5011_mut_062b726c-48ab-4c6a-b2ad-e4cc27cc165a',
'5011_ges/5011_mut_f0f4f654-b8b4-4d67-8553-783a622d2f9d')
for (i in 1:2){ 
    name <- paste(names[i], "_int_length=250_cutoff=0", sep="")
    print(name)
    
    d <- magic(e_name=paste(name, "_edges.csv", sep=""), p_name=paste(name, "_population.csv", sep=""), t_name=paste(name, "_trange.csv", sep=""))
    plotting(d, filename=name)
    } 
# names = c('5011_mut_01d15ca8-03d6-4ca0-985c-777dc41365d8',
#  '5011_mut_062b726c-48ab-4c6a-b2ad-e4cc27cc165a',
#   '5011_mut_498d4c70-5dc8-4f0f-bb52-51820fc66505',
#    '5011_mut_55186c3c-e01e-4609-8952-d1314b736521',
#  '5011_mut_623a24a3-be94-4a90-9141-9ddbafd4f0a8')
# for (i in 1:5){ 
#     name <- paste("5011_mut_04_01/", names[i], "_int_length=250_cutoff=0", sep="")
#     print(name)
    
#     d <- magic(e_name=paste(name, "_edges.csv", sep=""), p_name=paste(name, "_population.csv", sep=""), t_name=paste(name, "_trange.csv", sep=""))
#     plotting(d, filename=name)
# }







