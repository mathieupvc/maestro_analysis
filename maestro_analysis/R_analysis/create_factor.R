library(tidyverse)

behavior = read.csv("~/These/data/meg_maestro/behavior_analysis/behavior_data_matrix.csv")

behavior$note_rate = (behavior %>%
  pivot_longer(cols = c("X2.5Hz", "X3.5Hz", "X4.5Hz", "X5.5Hz", "X6.5Hz"),
               names_to = "note_rate") %>%
  filter(value == 1) %>%
  select(note_rate))$note_rate

behavior$rep = (behavior %>%
                        pivot_longer(cols = c("rep1", "rep2", "rep3"),
                                     names_to = "rep") %>%
                        filter(value == 1) %>%
                        select(rep))$rep