#' kf_gibbs_t - Gibbs sampler for Kalman filter
#' Assumes a heavy-tailed distribution (Student t) for variances rather than the normal distribution
#' Described in Chapter 4 of Dynamic Linear Models with R by Giovanni Petris . Sonia Petrone . Patrizia Campagnoli
#'
#' @param df - A dataframe with columns Year and Temperature_Anomaly
#' @param mod - A dlm model.
#' @param A_y - numerical, Prior mean for observational precision.
#' @param B_y  - numerical, Prior mean for observational variance.
#' @param A_theta - numerical, Prior mean for system precision. 
#' @param B_theta - numerical, Prior mean for system variance. 
#' @param burnin - an int, Gibbs burnin period. 
#' @param samples - an int, Gibbs sampling period. 
#' @param save_states - logical, Should sampled states be saved? 
#' @param thin - an int, Discard thin iterations for every saved iteration?
#' @param cutoff - Values of omega_theta <= cutoff are considere outliers.
#' @param title - Optional  title for model plot.
#'
#' @return - a list
#'           d: a dlm, output of dlmGibbsDIGt, 
#'           df2: a dataframe, with Year, Temperature_Anomaly, omega_y, omega_theta, posterior mean of samples and 
#'           sd of damples = df2, outliers = outliers.
#'           outliers: index of outlier rows in dataframe
#'           params - a list, input parameters
#' 
#' @requires - cutoff <= 1, 
#'             dlmGibbsDIGt from https://www.jstatsoft.org/index.php/jss/article/downloadSuppFile/v036i12/dlmGibbsDIGt.R
#' 
kf_gibbs_t <- function(df,
                       mod = dlmModPoly(1),
                       A_y = 1, B_y = 1,
                       A_theta = 10, B_theta = 10,
                       burnin = 2000, samples = 10000,
                       save_states = TRUE,
                       thin = 1,
                       cutoff = 0.95,
                       title = NULL) {
  require(dlm)
  require(tidyverse)
  require(ggpubr)
  
  # use dlmGibbsDIGt from https://www.jstatsoft.org/index.php/jss/article/downloadSuppFile/v036i12/dlmGibbsDIGt.R 
  # to do the heavy lifting.
  # See Dynamic Linear Models with R section 4.5.3 for details
  
  d <- dlmGibbsDIGt(df$Temperature_Anomaly, 
                    mod = mod, 
                    A_y = A_y, B_y = B_y, 
                    A_theta = A_theta, B_theta  = B_theta, 
                    n.sample = burnin + samples, 
                    thin = thin, 
                    save.states = save_states)
  
  # get the posterion mean and sd of latent variables, omegas
  df2 <- data.frame(Year = df$Year,
                    Temperature_Anomaly = df$Temperature_Anomaly,
                    omega_y = colMeans(d$omega_y[-(1:burnin), ]),
                    omega_theta = rowMeans(d$omega_theta[, 1, -(1:burnin)])) 
  outliers <- which(df2$omega_theta <= cutoff)   # these are positions with large variance in predicted mean
  
  if(save_states) {
    # get posterior mean and sd of the Kalman filter predictions and plot them
    df2 <- cbind(df2, 
                 data.frame(Posterior_Mean = rowMeans(d$theta[-1, 1, -(1:burnin)]),
                            sd = apply(d$theta[-1, 1, -(1:burnin)], 1, sd)))
    
    p1 <- ggplot(df2) +
      geom_point(aes(x = Year, y = Temperature_Anomaly, color = 'Temperature Anomaly')) +
      geom_line(aes(x = Year, y = Posterior_Mean, color = 'Posterior Mean')) +
      geom_point(aes(x = Year, y = Posterior_Mean, color = 'Posterior Mean')) +
      geom_ribbon(aes(x = Year, y = Posterior_Mean, 
                      ymin = Posterior_Mean - sd, 
                      ymax = Posterior_Mean + sd,
                      color = '+/- SD'),
                  fill = 'grey',
                  alpha = 0.4) +
      geom_point(data = data.frame(Year = df2$Year[outliers], Temperature_Anomaly = df2$Temperature_Anomaly[outliers]),
                 aes(x = Year, y = Temperature_Anomaly, color = 'Outliers')) +
      ylab('Temperature Anomaly') +
      scale_colour_manual(name = '',
                          labels = c('+/- SD', expression(paste('Outliers ', omega[theta])), 'Posterior Mean', 'Temperature Anomaly'),
                          values = c('grey', 'red', 'seagreen', 'blue'))
    
    if(! is.null(title)) {
      p1 <- p1 + ggtitle(title)
    }
    
    print(p1)
  }
  
  # plot the latent variables
  p2 <- ggplot(df2) +
    geom_point(aes(x = Year, y = omega_y)) +
    geom_hline(yintercept = 1) +
    ylab(expression(omega[y]))
  
  p3 <- ggplot(df2) +
    geom_point(aes(x = Year, y = omega_theta)) +
    geom_hline(yintercept = 1) +
    geom_hline(yintercept = cutoff, color = 'red') +
    ylab(expression(omega[theta]))
  
  p4 <- ggarrange(plotlist = list(p2, p3),
                  nrow = 2,
                  ncol = 1)
  
  print(p4)
  
  return(list(d = d, df2 = df2, outliers = outliers,
              params = list(mod = mod, A_y = A_y, B_y = B_y, A_theta = A_theta, B_theta = B_theta,
                            burnin = burnin, samples = samples, save_states = save_states, thin = thin,
                            cutoff = cutoff, title = title)))
}