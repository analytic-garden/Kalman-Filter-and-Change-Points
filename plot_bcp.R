#' plot_bcp
#'    Bayesian change point analysis of temperature anomaly data. 
#'
#' @param df - a data frame containing columns Year and Temperature_Anomaly
#' @param burnin - an int, the number of burnin iterations
#' @param mcmc - an int, the number of iterations after burnin.
#' @param p0 - a numeric between 0 and 1. prior on change point probabilities, U(0, p0)
#' @param d - a positive number, see bcp docs.
#' @param title - an optional main title
#'
#' @return - a list
#' df - a data frame with columns: Year, Temperature_Anomaly, Posterior_Probability, Posterior_Mean, Posterior_Variance
#' bcp - the output from the bcp function
#' params - a list, the input parameters
#' 
plot_bcp <- function(df,
                     burnin = 1000,
                     mcmc = 10000,
                     p0 = 0.2,
                     d = 10,
                     title = NULL) {
  require(bcp)
  require(tidyverse)
  require(ggpubr)
  
  # run bcp to fit the model
  bcp_out <- bcp(df$Temperature_Anomaly, df$Year, 
                 d = d,
                 burnin = burnin,
                 mcmc = mcmc,
                 p0 = p0)
  
  # the rest is juts plotting
  df2 <- data.frame(Year = df$Year,
                    Temperature_Anomaly = df$Temperature_Anomaly,
                    Posterior_Probability = bcp_out$posterior.prob,
                    Posterior_Mean = bcp_out$posterior.mean[, 1],
                    Posterior_Variance = bcp_out$posterior.var[, 1])
  
  p1 <- ggplot(df2) +
    geom_point(aes(x = Year, y = Temperature_Anomaly, color = 'Temperature Anomaly')) +
    geom_line(aes(x = Year, y = Posterior_Mean, color = 'BCP Posterior Mean')) +
    geom_ribbon(aes(x = Year, 
                    y = Posterior_Mean,
                    ymin = Posterior_Mean - sqrt(Posterior_Variance), 
                    ymax = Posterior_Mean + sqrt(Posterior_Variance),
                    color = '+/- SD'),
                fill = 'grey',
                alpha = 0.4) +
    scale_colour_manual(name = '',
                        values = c('grey', 'seagreen', 'brown')) +
    theme(legend.position="bottom")
  
  p2 <- ggplot(df2) +
    geom_linerange(aes(x = Year, ymin = 0, ymax = Posterior_Probability)) +
    ylab('Probability of Change Point')
  
  p <- ggarrange(plotlist = list(p1, p2),
                 ncol = 1,
                 nrow = 2)
  
  if(! is.null(title)) {
    p <- annotate_figure(p, top = text_grob(title, 
                                            face = "bold", 
                                            size = 14))
  }
  
  print(p)
  
  return(list(df = df2, 
              bcp = bcp_out,
              params = list(burnin = 50,
                            mcmc = 500,
                            p0 = 10,
                            d = 10)))
}