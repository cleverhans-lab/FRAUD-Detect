library(ggplot2)


theme_ulr <- function(base_size = 18, base_family = "Helvetica") {
  theme_linedraw(base_size = base_size, base_family = base_family) %+replace%
    theme(
      strip.background = element_blank(),
      panel.spacing = unit(3.5, "lines"), 
    )
}


theme_yul <- function(base_size = 11, base_family = "Helvetica",
                          base_line_size = base_size / 22,
                          base_rect_size = base_size / 22) {
  # Starts with theme_bw and remove most parts
  theme_bw(
    base_size = base_size,
    base_family = base_family,
    base_line_size = base_line_size,
    base_rect_size = base_rect_size
  ) %+replace%
    theme(
      axis.ticks      = element_blank(),
      legend.background = element_blank(),
      legend.key        = element_blank(),
      panel.background  = element_blank(),
      panel.border      = element_blank(),
      strip.background  = element_blank(),
      plot.background   = element_blank(),

      complete = TRUE
    )
}

theme_light_v2 <- function(base_size = 11, base_family = "",
                        base_line_size = base_size / 22,
                        base_rect_size = base_size / 22) {
  half_line <- base_size / 2

  # Starts with theme_grey and then modify some parts
  theme_grey(
    base_size = base_size,
    base_family = base_family,
    base_line_size = base_line_size,
    base_rect_size = base_rect_size
  ) %+replace%
    theme(
      # white panel with light grey border
      panel.background = element_rect(fill = "white", colour = NA),
      panel.border     = element_rect(fill = NA, colour = "grey70", size = rel(1)),
      # light grey, thinner gridlines
      # => make them slightly darker to keep acceptable contrast
      panel.grid       = element_line(colour = "grey87"),
      panel.grid.major = element_line(size = rel(0.5)),
      panel.grid.minor = element_line(size = rel(0.25)),

      # match axes ticks thickness to gridlines and colour to panel border
      axis.ticks       = element_line(colour = "grey70", size = rel(0.5)),

      # match legend key to panel.background
      legend.key       = element_rect(fill = "white", colour = NA),

      # dark strips with light text (inverse contrast compared to theme_grey)
      strip.background = element_rect(fill = NA, colour = NA),
      strip.text       = element_text(
                           colour = "black",
                           size = rel(0.8),
                           margin = margin(0.8 * half_line, 0.8 * half_line, 0.8 * half_line, 0.8 * half_line)
                         ),

      complete = TRUE
    )

}
