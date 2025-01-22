library(BayesFactor)
library(dplyr)

############################# load data
# Load
data <- read.csv('AnalyseSSVEPs/Results/SSVEPsresults.csv')
head(data, 5)

# subselect data to analyse
data <- data[(data$flickertype == 'interpflicker') & (data$freqrange == 'lowerfreqs')& (data$harmonic== 1),]
data$Group <- factor(data$Group)

############################### SSVEP Amplitude - ANOVA

# Prepare factors for ANOVA
tmp1 = data[,c('Group','Higher.SF.Db', 'subid')]
tmp1['SF'] = 'Higher'
# tmp1['subid'] = c(1,2,3,4,5,6,7,8,9,10,11,12,13)
names(tmp1)[names(tmp1) == "Higher.SF.Db"] <- "SSVEP"

tmp2 = data[,c('Group','Lower.SF.Db', 'subid')]
tmp2['SF'] = 'Lower'
# tmp2['subid'] = c(1,2,3,4,5,6,7,8,9,10,11,12,13)
names(tmp2)[names(tmp2) == "Lower.SF.Db"] <- "SSVEP"

data2 <- rbind(tmp1,tmp2)
data2$subid <- factor(data2$subid)
data2$SF <- factor(data2$SF)

# Get metadata
stats = data2  %>% group_by(Group)  %>% summarise(mean(SSVEP), min(SSVEP), max(SSVEP), sd(SSVEP))
stats
# Group   `mean(SSVEP)` `min(SSVEP)` `max(SSVEP)` `sd(SSVEP)`
# <fct>           <dbl>        <dbl>        <dbl>       <dbl>
# 1 AMD              9.61        0.453         18.8        4.29
# 2 Control         11.4         2.98          22.1        3.99

stats = data2  %>% group_by(Group, SF)  %>% summarise(mean(SSVEP), min(SSVEP), max(SSVEP), sd(SSVEP))
stats
# Group   SF     `mean(SSVEP)` `min(SSVEP)` `max(SSVEP)` `sd(SSVEP)`
# <fct>   <fct>          <dbl>        <dbl>        <dbl>       <dbl>
# 1 AMD     Higher          5.99        0.453         8.17        2.21
# 2 AMD     Lower          13.2        10.7          18.8         2.26
# 3 Control Higher         11.7         3.53         22.1         4.76
# 4 Control Lower          11.0         2.98         15.1         3.17

# Run anova
bf = anovaBF(SSVEP ~ Group + SF + Group:SF + subid, data = data2, whichRandom = "subid",
             progress=FALSE, whichModels='top')

bf_interaction = 1/bf[1]
# Bayes factor analysis
# --------------
# [1] Group + SF + Group:SF + subid : 9421.94 ±23.84%
#
# Against denominator:
#   SSVEP ~ Group + SF + subid
bf_mainSF = 1/bf[2]
# > bf_mainSF
# Bayes factor analysis
# --------------
# [1] Group + SF + Group:SF + subid : 733.589 ±24.03%
#
# Against denominator:
#   SSVEP ~ Group + Group:SF + subid
bf_mainGroup = 1/bf[3]
# > bf_mainGroup
# Bayes factor analysis
# --------------
# [1] Group + SF + Group:SF + subid : 1.506098 ±23.87%
#
# Against denominator:
#   SSVEP ~ SF + Group:SF + subid

## Compare the two models

# Bayes factor analysis
# --------------
# [1] Group + SF + Group:SF + subid : 6997.859 ±3.16%
#
# Against denominator:
#   SSVEP ~ Group + SF + subid


############################### SSVEP Amplitude - followup t-tests

###### Higher.SF
# visualise
plot(Higher.SF.Db ~  Group, data = data, main = "Higher SF SSVEPs by group")

## traditional t test
t.test(Higher.SF.Db  ~  Group, data = data, var.eq=TRUE, whichRandom = "subid")
# data:  Higher.SF.Db by Group
# t = -4.2131, df = 29, p-value = 0.0002237
# alternative hypothesis: true difference in means between group AMD and group Control is not equal to 0
# 95 percent confidence interval:
# -8.430936 -2.920475
# sample estimates:
# mean in group AMD mean in group Control
# 5.98567              11.66138

## Compute Bayes factor

bf = ttestBF(formula = Higher.SF.Db ~  Group, data = data, whichRandom = "subid")
bf #Alt., r=0.707 : 106.6115 ±0%

chains = posterior(bf, iterations = 10000)
plot(chains[,2]) # these seem to have converged nicely


###### Lower.SF
# visualise
plot(Lower.SF.Db ~  Group, data = data, main = "Lower SF SSVEPs by group")

## traditional t test
t.test(Lower.SF.Db ~  Group, data = data, var.eq=TRUE, whichRandom = "subid")
# t = 2.1962, df = 29, p-value = 0.03622
# alternative hypothesis: true difference in means between group AMD and group Control is not equal to 0
# 95 percent confidence interval:
# 0.1503014 4.2215054
# sample estimates:
# mean in group AMD mean in group Control
# 13.23404              11.04814

## Compute Bayes factor
bf = ttestBF(formula = Lower.SF.Db ~  Group, data = data, whichRandom = "subid")
bf # Alt., r=0.707 : 1.98321 ±0.01%

chains = posterior(bf, iterations = 10000)
plot(chains[,2]) # these seem to have converged nicely
summary(chains)



############################# SSVEP ratio (low/high sf)
# compute orig ratio
data['SSVEPratio1'] = data['Lower.SF'] / data['Higher.SF']

# visualise
plot(SSVEPratio1 ~  Group, data = data, main = "SSVEP Ratios by group")
plot(SSVEP.ratio..low.high.sf. ~  Group, data = data, main = "SSVEP Ratios by group")

library(ggpubr)
ggqqplot(data$SSVEPratio1)
ggqqplot(data$SSVEP.ratio..low.high.sf.)

shapiro.test(data$SSVEPratio1)
# Shapiro-Wilk normality test
#
# data:  data$SSVEPratio1
# W = 0.78454, p-value = 2.705e-05

shapiro.test(data$SSVEP.ratio..low.high.sf.)
# Shapiro-Wilk normality test
#
# data:  data$SSVEP.ratio..low.high.sf.
# W = 0.93799, p-value = 0.0726

## traditional t test
t.test(SSVEP.ratio..low.high.sf. ~  Group, data = data, var.eq=TRUE, whichRandom = "subid")
# t = 5.1311, df = 29, p-value = 1.762e-05
# alternative hypothesis: true difference in means between group AMD and group Control is not equal to 0
# 95 percent confidence interval:
# 1.088670 2.531735
# sample estimates:
# mean in group AMD mean in group Control
# 1.6689992            -0.1412032


## Compute Bayes factor
bf <- ttestBF(formula = SSVEP.ratio..low.high.sf. ~  Group, data = data, whichRandom = "subid")
bf # [1] Alt., r=0.707 : 940.4702 ±0%

chains <- posterior(bf, iterations = 10000)
plot(chains[,2]) # these seem to have converged nicely

# Get metadata
stats = data  %>% group_by(Group)  %>% summarise(mean(SSVEP.ratio..low.high.sf.),
                                                 max(SSVEP.ratio..low.high.sf.), sd(SSVEP.ratio..low.high.sf.))
stats

# Group   `mean(SSVEP.ratio..low.high.sf.)` `max(SSVEP.ratio..low.high.sf.)` `sd(SSVEP.ratio..low.high.sf.)`
# <fct>                               <dbl>                            <dbl>                           <dbl>
# 1 AMD                                 1.67                              2.97                           0.614
# 2 Control                            -0.141                             1.59                           1.23
# # i abbreviated names: 1: `mean(SSVEP.ratio..low.high.sf.)`,
# #   2: `min(SSVEP.ratio..low.high.sf.)`, 3: `max(SSVEP.ratio..low.high.sf.)`
# # i 1 more variable: `sd(SSVEP.ratio..low.high.sf.)` <dbl>
# >




############ regression
# normalise data
datuse = data[,c('SSVEP.ratio..low.high.sf.','logMAR', 'logCS', 'Reading.Speed')]
scaled <- scale(datuse)
scaled <- data.frame(scaled)

# calculate relationship for logMAR
summary(lm(logMAR ~ SSVEP.ratio..low.high.sf., data = scaled))
bfReg = lmBF(logMAR ~ SSVEP.ratio..low.high.sf., data = scaled)
bfReg
# [1]  SSVEP.ratio..low.high.sf. : 81.02535 ±0%
# Against denominator:
# Intercept only
chains = posterior(bfReg, iterations = 10000)
summary(chains)
#
# 1. Empirical mean and standard deviation for each variable,
# plus standard error of the mean:
# #
# Estimate Std. Error t value Pr(>|t|)
# (Intercept)               -2.477e-16  1.454e-01   0.000 1.000000
# SSVEP.ratio..low.high.sf.  6.051e-01  1.478e-01   4.093 0.000311 ***
# Mean        SD  Naive SE Time-series SE
# mu                        -0.001014    0.1532  0.001532       0.001532
# SSVEP.ratio..low.high.sf.  0.540136    0.1579  0.001579       0.001855
# sig2                       0.718273    0.2019  0.002019       0.002269
# g                         32.205530 2738.5453 27.385453      27.385453



# calculate relationship for logCS
summary(lm(logCS ~ SSVEP.ratio..low.high.sf., data = scaled))
bfReg = lmBF(logCS ~ SSVEP.ratio..low.high.sf., data = scaled)
bfReg
# Bayes factor analysis
# --------------
# [1] SSVEP.ratio..low.high.sf. : 7.656656 ±0%

# Estimate Std. Error t value Pr(>|t|)
# (Intercept)               -1.508e-16  1.599e-01   0.000  1.00000
# SSVEP.ratio..low.high.sf. -4.838e-01  1.625e-01  -2.977  0.00583 **

#
# Against denominator:
# Intercept only
chains = posterior(bfReg, iterations = 10000)
summary(chains)
# Mean      SD Naive SE Time-series SE
# mu                         0.001836  0.1682 0.001682       0.001676
# SSVEP.ratio..low.high.sf. -0.412081  0.1708 0.001708       0.001973
# sig2                       0.862571  0.2426 0.002426       0.002486
# g                          1.394494 10.0588 0.100588       0.100588







# calculate relationship for Reading speed
summary(lm(Reading.Speed ~ SSVEP.ratio..low.high.sf., data = scaled))
bfReg = lmBF(Reading.Speed ~ SSVEP.ratio..low.high.sf., data = scaled)
bfReg
# Bayes factor analysis
# --------------
# [1] SSVEP.ratio..low.high.sf. : 2.465203 ±0%
#
# Against denominator:
# Intercept only

chains = posterior(bfReg, iterations = 10000)
summary(chains)
# Mean       SD Naive SE Time-series SE
# mu                        -0.0008767   0.2610 0.002610       0.002610
# SSVEP.ratio..low.high.sf. -0.4399530   0.2577 0.002577       0.003176
# sig2                       0.8793454   0.4410 0.004410       0.005321
# g                          2.9357634 109.5576 1.095576       1.095576


## LogMAR

## Compute Bayes factor
bf <- ttestBF(formula = logMAR ~  Group, data = data)
bf #[1] Alt., r=0.707 : 450.2823 ±0%

chains <- posterior(bf, iterations = 10000)
plot(chains[,2]) # these seem to have converged nicely

# visualise
plot( logMAR ~  Group, data = data, main = "logmar by group")

## traditional t test
t.test( logMAR ~  Group, data = data, var.eq=TRUE)
# data:  logMAR by Group
# t = 4.8257, df = 29, p-value = 4.118e-05
# alternative hypothesis: true difference in means between group AMD and group Control is not equal to 0
# 95 percent confidence interval:
#  0.1586166 0.3919667
# sample estimates:
#     mean in group AMD mean in group Control
#             0.3046667             0.0293750

stats = data  %>% group_by(Group)  %>% summarise(mean(logMAR), min(logMAR), max(logMAR), sd(logMAR))
# A tibble: 2 x 5
# Group   `mean(logMAR)` `min(logMAR)` `max(logMAR)` `sd(logMAR)`
# <fct>            <dbl>         <dbl>         <dbl>        <dbl>
# 1 AMD             0.305           0.07          0.81        0.202
# 2 Control         0.0294         -0.22          0.23        0.103


## LogCS
## Compute Bayes factor
bf <- ttestBF(formula = logCS ~  Group, data = data)
bf #[1] Alt., r=0.707 : 48.84005 ±0%

chains <- posterior(bf, iterations = 10000)
plot(chains[,2]) # these seem to have converged nicely


######  Mean.SSVEP.Amp..SNR.
# visualise
plot( logCS ~  Group, data = data, main = "logmar by group")

## traditional t test
t.test( logCS ~  Group, data = data, var.eq=TRUE)
#t = -3.8678, df = 29, p-value = 0.0005721
# alternative hypothesis: true difference in means between group AMD and group Control is not equal to 0
# 95 percent confidence interval:
# -0.8063030 -0.2485303
# sample estimates:
# mean in group AMD mean in group Control
# 1.215333              1.742750

stats = data  %>% group_by(Group)  %>% summarise(mean(logCS), min(logCS), max(logCS), sd(logCS))
stats
# # A tibble: 2 x 5
#   Group   `mean(logCS)` `min(logCS)` `max(logCS)` `sd(logCS)`
#   <fct>           <dbl>        <dbl>        <dbl>       <dbl>
# 1 AMD              1.22         0.2          1.94       0.515
# 2 Control          1.74         1.42         2.11       0.176


############ Table for paper
tabledat = data[, c('Group', 'subid', 'logMAR', 'logCS', 'Higher.SF.Db', 'Lower.SF.Db', 'SSVEP.ratio..low.high.sf.')]