#' trend_filter - a local linear trend Kalman filter
#'
#' @param df - a dataframe with columns Year and Temperature_Anomaly
#' @param title - a string, option title for main plot.
#'
#' @return - a list
#'           dlm - a dlm returned by the dlm function
#'           kf - the Kalman filter returned by dlmFilter
#'           smooth - the moothed Kalman filter
#'           LL - the log likelihood of the filter
#'           
trend_filter <- function(df, 
                      title = NULL) {
  require(tidyverse)
  require(dlm)
  
  # see Dynamic Linear Models with R by Giovanni Petris, Sonia Petrone, and Patrizia Campagnoli
  # model is in section 2.4
  
  ############################################################################################
  
  # Minimization function used by dlmMLE
  # parameters are fit as exponentions to avoind negative variances
  # from https://cran.r-project.org/web/packages/dlm/vignettes/dlm.pdf
  buildFun <- function(x, FF, GG, m0, C0) {
    dlm(V = exp(x[1]), 
        W = diag(c(exp(x[2]), exp(x[3]))),
        FF = FF,
        GG = GG,
        m0 = m0,
        C0 = C0)
  }

  ############################################################################################
  
  # set up the model
  FF <-  matrix(c(1, 0), nr = 1)
  V <- 1.4
  GG <- matrix(c(1, 0, 1, 1), nr = 2)
  W <- diag(c(0.1, 0.2))
  m0 <- rep(0, 2)
  C0 <- 10 * diag(2)

  # optimize V and the W's
  fit <- dlmMLE(df$Temperature_Anomaly, parm = c(0, 0, 0), build = buildFun, method = "L-BFGS-B",
                FF = FF,
                GG = GG,
                m0 = m0,
                C0 = C0)
  V <- exp(fit$par[1])
  W <- diag(exp(fit$par[2:3]))
  
  # the dlm model
  d <- dlm(V = V, 
           W = W,
           FF = FF, 
           GG = GG, 
           m0 = m0, 
           C0 = C0)
  
  # get filter and smoothed filter
  res <- dlmFilter(df$Temperature_Anomaly, d)
  s <- dlmSmooth(res)
  ll <- dlmLL(df$Temperature_Anomaly, d)
  
  # plot everything
  df2 <- data.frame(df, Fit = res$m[-1, 1], Smoothed = s$s[-1, 1])

  p <- ggplot(df2, aes(x = Year, y = Temperature_Anomaly, color = "Temperature Anomaly")) + 
    geom_point() + 
    # geom_line() +
    geom_point(aes(x = Year, y = Fit, color = "Posterior Estimate")) + 
    geom_line(aes(x = Year, y = Fit, color = "Posterior Estimate")) +
    geom_point(aes(x = Year, y = Smoothed, color = "Smoothed Estimate")) + 
    geom_line(aes(x = Year, y = Smoothed, color = "Smoothed Estimate")) +
    ylab('Temperature Anomaly') +
    scale_colour_manual(name = '',
                        values = c("brown", "blue", "seagreen")) 
  
  if(! is.null(title)) {
    p <- p + ggtitle(title)
  }
    
  print(p)
  
  return(list(dlm = d, kf = res, smooth = s, LL = -ll))
}