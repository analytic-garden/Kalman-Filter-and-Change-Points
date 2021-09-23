#' kf_gibbs - Kalman Filter Gibbs sampler
#' This is a simple tend following Kalman filter. It should be easy to adapt fpr more complicated models.
#' Based on Dynamic Linear Models with R by Giovanni Petris . Sonia Petrone . Patrizia Campagnoli
#' and the R dlm pacakage.
#'
#' @param df - A dataframe with columns Year and Temperature_Anomaly
#' @param burnin - an int, Gibbs burnin period.  
#' @param samples - an int, Gibbs sampling period. 
#' @param C0 - an int, the variance of the pre-sample state vector.
#' @param a1 - an int, parameter for gamma prior
#' @param b1 - an int, parameter for gamma prior
#' @param a2 - an int, parameter for gamma prior
#' @param b2 - an int, parameter for gamma prior
#' @param thin - an int, discard thin iterations for every saved iteration
#' @param title - a string, option title for main plot.
#'
#' @return a list
#'         dlm - a dlm model (basically a list, see dlm pacakage docs)
#'         phi1, phi2 - saved inverse gamma draws
#'         df - a large dataframe, Year, Temperature_Anomaly from input df, posterior mean and sd of Kalman predictions,
#'              and draws for each iteration
#'         LL - the log likelihood for each iteration
#'         params - a list, input parameters
#' 
#' @requires all int parameters >= 0
#' 
kf_gibbs <- function(df, 
                     burnin = 2000,
                     samples = 10000,
                     C0 = 1e+07,
                     a1 = 2,
                     b1 = 0.0001,
                     a2 = 2,
                     b2 = 0.0001,
                     thin = 1,
                     title = NULL) {
  require(tidyverse)
  require(ggpubr)
  require(dlm)
  
  # from Dynamic Linear Models with R b Giovanni Petris . Sonia Petrone . Patrizia Campagnoli
  # section 4.4.3
  
  # starting values for posterior variance parameters
  phi1 <- 1  # 1/V
  phi2 <- 1  # 1/W
  
  # setup a dlm
  d <- dlmModPoly(order = 1, dV = 1 / phi1, dW = 1 / phi2, C0 = C0)
  
  iter_save = thin + 1
  mc <- (burnin + samples) * iter_save
  
  #save parameter samples
  phi1_save <- numeric(burnin + samples)
  phi2_save <- numeric(burnin + samples)
  
  ll_save <- numeric(burnin + samples)  # save log likelihood
  
  n <- nrow(df)
  
  # posterior params for inverse gamma
  sh1 <- a1 + n / 2
  sh2 <- a2 + n / 2
  
  df2 <- data.frame(Year = df$Year, 
                    Temperature_Anomaly = df$Temperature_Anomaly,
                    posterior_mean = numeric(nrow(df)),
                    sd = numeric(nrow(df)))
  
  save_count <- 1
  # Gibbs sampler
  for(iter in 1:mc) {
    # show progress
    if(iter %% 100 == 0) {
      cat('Iteration ', iter, '\r')
    }
    
    # draw the states: FFBS
    filt <- dlmFilter(df2$Temperature_Anomaly, d)
    level <- dlmBSample(filt)
    
    if(iter %% iter_save == 0) {
      ll_save[save_count] <- dlmLL(df2$Temperature_Anomaly, d)
    
      # Save the samples
      df2 <- cbind(df2, level[-1])
      names(df2)[ncol(df2)] <- paste0('sample_', save_count)
    }
    
    # draw observation precision phi1
    rate <- b1 + crossprod(df$Temperature_Anomaly - level[-1]) / 2
    phi1 <- rgamma(1, shape = sh1, rate = rate)
    
    # draw state precision phi2
    rate <- b2 + crossprod(level[-1] - level[-n]) / 2
    phi2 <- rgamma(1, shape = sh2, rate = rate)
    
    # update filter params
    V(d) <- 1 / phi1
    W(d) <- 1 / phi2
    
    if(iter %% iter_save == 0) {
      phi1_save[save_count] <- 1/phi1
      phi2_save[save_count] <- 1/phi2
      save_count <- save_count + 1
    }
  }
  
  # posterior mean of sampled Kalman filter draws
  df2$posterior_mean <- rowMeans(df2[ , -(1:(burnin+4))])
  df2$sd <- apply(df2[ , -(1:(burnin+4))], 1, sd)
  
  p <- ggplot(df2, aes(x = Year, y = Temperature_Anomaly, color = 'Temperature Anomaly')) + 
    geom_point() +
    # geom_line() +
    geom_point(aes(x = Year, y = posterior_mean, color = 'Posterior Mean')) +
    geom_line(aes(x = Year, y = posterior_mean, color = 'Posterior Mean')) +
    geom_ribbon(aes(x = Year, y = posterior_mean, 
                    ymin = posterior_mean - sd, 
                    ymax = posterior_mean + sd,
                    color = '+/- SD'),
                fill = 'grey',
                alpha = 0.4) +
    ylab('Temperature Anomaly') +
    scale_colour_manual(name = '',
                        values = c('grey', 'seagreen', 'brown'))
  
  if(! is.null(title)) {
    p <- p + ggtitle(title)
  }

  print(p)
  
  # plot sampled variances
  df3 <- data.frame(Iteration = (burnin+1):(mc/iter_save), phi1 = phi1_save[-(1:burnin)], phi2 = phi2_save[-(1:burnin)])
  
  p2 <- ggplot(df3, aes(x = Iteration, y = phi1)) + 
    geom_line() +
    ylab(expression(phi[y])) +
    ggtitle(expression(phi[y]))
  p3 <- ggplot(df3, aes(x = Iteration, y = phi2)) + 
    geom_line() +
    ylab(expression(phi[theta])) +
    ggtitle(expression(phi[theta]))

  p4 <- ggarrange(plotlist = list(p2, p3),
                  ncol = 1,
                  nrow = 2)
  print(p4)
  
  return(list(dlm = d, phi1 = phi1_save, phi2 = phi2_save, df = df2, LL = ll_save,
              params = list(a1 = a1 , b1 = b1, a2 = a2, b2 = b2, 
                            burnin = burnin, samples = samples, thin = thin, title = title)))
 }