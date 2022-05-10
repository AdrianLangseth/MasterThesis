First run of all:

| val_loss (acc)      | Neural        | Dot           | Sum           | Avg           | Max           |
|---------------------|---------------|---------------|---------------|---------------|---------------|
| news                | .0795 (98.3%) | .0780 (98.0%) | .0113 (97.5%) | .0787 (97.8%) | .0844 (97.8%) |
| loc_to_loc          | .0995 (98.1%) | .0729 (98.1%) | .4284 (83.3%) | .0590 (98.8%) | XXXXXXXXXXXXX |
| pos_to_loc          | .0491 (98.8%) | .2300 (97.5%) | .1742 (96.6%) | .1329 (97.4%) | XXXXXXXXXXXXX |
| hist_pos_to_loc     | .0466 (98.8%) | .1405 (98.1%) | .1016 (98.2%) | .0987 (98.3%) | .0794 (98.1%) |
| comb_pos_to_loc     | .0472 (98.7%) | .1995 (97.2%) | .1616 (97.0%) | .1433 (96.5%) | .1567 (95.6%) |
| pos_loc_to_loc      | .0608 (98.5%) |               |               | XXXXXXXXXXXXX | XXXXXXXXXXXXX |
| hist_pos_loc_to_loc | .0497 (98.9%) |               |               | .1329 (97.4%) | XXXXXXXXXXXXX |
| comb_pos_loc_to_loc | .0807 (98.6%) |               |               | .1329 (97.4%) | XXXXXXXXXXXXX |



comb_pos_loc_to_loc
hist_pos_loc_to_loc
pos_loc_to_loc
loc_to_loc
comb_pos_to_loc
hist_pos_to_loc
pos_to_loc
news