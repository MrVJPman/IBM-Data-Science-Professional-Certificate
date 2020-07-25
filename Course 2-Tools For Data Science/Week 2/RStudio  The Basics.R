library (datasets)
data(iris)
View(iris) 

unique(iris$Species)

#install.packages(c("GGally", "ggplot2"))
#library(GGally)
#ggpairs(iris, mapping=ggplot2::aes(colour = Species))