library(BayesFactor)
library(dplyr)

############################# load data
# Load
data <- read.csv('AnalyseSSVEPs/Results/SSVEPsresults_byvideo.csv')
head(data, 5)

# subselect data to analyse
data <- data[(data$flickertype == 'interpflicker') & (data$freqrange == 'lowerfreqs')& (data$harmonic== 1),]
data$Group <- factor(data$Group)
data$video <- factor(data$video)

# Metadata
stats = data  %>% group_by(Group, video)  %>% summarise(mean(SSVEP.ratio..low.high.sf.), sd(SSVEP.ratio..low.high.sf.)) #,
stats


## Compute Bayes factor
for (vid in c('V1', 'V2', 'V3', 'V4', 'V5', 'V6')) {
  print(vid)
  datause = data[data$video==vid,]
  bf = ttestBF(formula = SSVEP.ratio..low.high.sf. ~  Group, data = datause, whichRandom = "subid")
  print(bf)

}
#
# [1] "V1"
# [1] Alt., r=0.707 : 0.4303071 ±0%
#
# [1] "V2"
# [1] Alt., r=0.707 : 1.422696 ±0.01%
#
# [1] "V3"
# [1] Alt., r=0.707 : 4.838807 ±0%
#
# [1] "V4"
# [1] Alt., r=0.707 : 0.3577593 ±0%
#
# [1] "V5"
# [1] Alt., r=0.707 : 0.3713139 ±0%
#
# [1] "V6"
# [1] Alt., r=0.707 : 1.187516 ±0%


# ANOVA
data$videoid <- factor(data$video)
data$videoid  = factor(as.numeric(data$videoid))
data$subid <- factor(data$subid)
# Run anova
bf = anovaBF(SSVEP.ratio..low.high.sf. ~  Group + videoid + Group:videoid + subid, data = data, whichRandom = "subid",
             progress=FALSE, whichModels='top')


# When effect is omitted from Group + videoid + Group:videoid + subid , BF is...
# [1] Omit Group:videoid : 2.282855  ±1.25%
# [2] Omit videoid       : 26.11523  ±1.32%
# [3] Omit Group         : 0.5095491 ±1.17%
#
# Against denominator:
# SSVEP.ratio..low.high.sf. ~ Group + videoid + Group:videoid + subid
# ---
# Bayes factor type: BFlinearModel, JZS