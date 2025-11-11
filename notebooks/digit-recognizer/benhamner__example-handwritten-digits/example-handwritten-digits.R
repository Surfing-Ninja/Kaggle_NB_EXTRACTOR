# Plots 49 random handwritten digits drawn from the training set

library(ggplot2)
library(proto)
library(readr)
train <- data.frame(read_csv("../input/train.csv"))

labels   <- train[,1]
features <- train[,-1]

rowsToPlot <- sample(1:nrow(train), 49)

rowToMatrix <- function(row) {
    intensity <- as.numeric(row)/max(as.numeric(row))
    return(t(matrix((rgb(intensity, intensity, intensity)), 28, 28)))
}

geom_digit <- function (digits, labels) GeomRasterDigit$new(geom_params = list(digits=digits),
  stat = "identity", position = "identity", data = NULL, inherit.aes = TRUE)

GeomRasterDigit <- proto(ggplot2:::GeomRaster, expr={
  draw_groups <- function(., data, scales, coordinates, digits, ...) {
    bounds <- coord_transform(coordinates, data.frame(x = c(-Inf, Inf), y = c(-Inf, Inf)), scales)
    x_rng <- range(bounds$x, na.rm = TRUE)
    y_rng <- range(bounds$y, na.rm = TRUE)
    rasterGrob(as.raster(rowToMatrix(digits[data$rows,])), x_rng[1], y_rng[1], diff(x_rng), diff(y_rng), 
               default.units = "native", just = c("left","bottom"), interpolate = FALSE)
  }
})

p <- ggplot(data.frame(rows=rowsToPlot, labels=labels[rowsToPlot]), aes(x=.1, y=.9, rows=rows, label=labels)) + geom_blank() + xlim(0,1) + ylim(0,1) + xlab("") + ylab("") + 
  facet_wrap(~ rows, ncol=7) +
  geom_digit(features) +
  geom_text(colour="#53cfff") +
  theme(strip.background = element_blank(),
        strip.text.x = element_blank(),
        axis.text.x = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks = element_blank(),
        axis.line = element_blank()) +
  ggtitle("Example Handwritten Digits")

ggsave("example_digits.png", p, width=10, height=10)