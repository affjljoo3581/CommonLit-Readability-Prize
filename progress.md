# Progress Table
| filename | implementation | test code | docstring |
|----------|:--------------:|:---------:|:---------:|
| **src**                                                                                       | ✅ | ✅ | ✅ |
| &nbsp; &nbsp; **src/data**                                                                    | ✅ | ✅ | ✅ |
| &nbsp; &nbsp; &nbsp; &nbsp; src/data/test_dataset.py                                          | ✅ | ✅ | ✅ |
| &nbsp; &nbsp; &nbsp; &nbsp; src/data/train_dataset.py                                         | ✅ | ✅ | ✅ |
| &nbsp; &nbsp; **src/lightning**                                                               | ✅ | ✅ | ✅ |
| &nbsp; &nbsp; &nbsp; &nbsp; **src/lightning/callbacks**                                       | ✅ | ✅ | ✅ |
| &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; src/lightning/callbacks/best_score.py               | ✅ | ✅ | ✅ |
| &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; src/lightning/callbacks/pruning.py                  | ✅ | ✅ | ✅ |
| &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; src/lightning/callbacks/validation_epoch.py         | ✅ | ✅ | ✅ |
| &nbsp; &nbsp; &nbsp; &nbsp; **src/lightning/finetuning**                                      | ✅ | ✅ | ✅ |
| &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; src/lightning/finetuning/datamodule.py              | ✅ | 🟧 | ✅ |
| &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; src/lightning/finetuning/module.py                  | ✅ | 🟧 | ✅ |
| &nbsp; &nbsp; &nbsp; &nbsp; **src/lightning/pretraining**                                     | ✅ | ✅ | ✅ |
| &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; src/lightning/pretraining/datamodule.py             | ✅ | 🟧 | ✅ |
| &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; src/lightning/pretraining/module.py                 | ✅ | 🟧 | ✅ |
| &nbsp; &nbsp; &nbsp; &nbsp; src/lightning/extensions.py                                       | ✅ | ✅ | ✅ |
| &nbsp; &nbsp; **src/modeling**                                                                | ✅ | ✅ | ✅ |
| &nbsp; &nbsp; &nbsp; &nbsp; src/modeling/miscellaneous.py                                     | ✅ | ✅ | ✅ |
| &nbsp; &nbsp; &nbsp; &nbsp; src/modeling/mixins.py                                            | ✅ | 🟧 | ✅ |
| &nbsp; &nbsp; &nbsp; &nbsp; **src/modeling/models**                                           | ✅ | ✅ | ✅ |
| &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; src/modeling/models/auto.py                         | ✅ | ✅ | 🟧 |
| &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; src/modeling/models/bart.py                         | ✅ | ✅ | ✅ |
| &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; src/modeling/models/bert.py                         | ✅ | ✅ | ✅ |
| &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; src/modeling/models/distilbert.py                   | ✅ | ✅ | ✅ |
| &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; src/modeling/models/electra.py                      | ✅ | ✅ | ✅ |
| &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; src/modeling/models/roberta.py                      | ✅ | ✅ | ✅ |
| &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; src/modeling/models/xlnet.py                        | ✅ | ✅ | ✅ |
| &nbsp; &nbsp; **src/optimization**                                                            | ✅ | ✅ | ✅ |
| &nbsp; &nbsp; &nbsp; &nbsp; src/optimization/lr_scheduling.py                                 | ✅ | ✅ | ✅ |
| &nbsp; &nbsp; &nbsp; &nbsp; src/optimization/param_groups.py                                  | ✅ | ✅ | ✅ |
| &nbsp; &nbsp; **src/tuning**                                                                  | ✅ | ✅ | ✅ |
| &nbsp; &nbsp; &nbsp; &nbsp; src/tuning/parameter.py                                           | ✅ | ✅ | ✅ |
| &nbsp; &nbsp; &nbsp; &nbsp; src/tuning/search_space.py                                        | ✅ | ✅ | ✅ |
| &nbsp; &nbsp; &nbsp; &nbsp; src/tuning/study.py                                               | ✅ | ✅ | ✅ |
| &nbsp; &nbsp; **src/utils**                                                                   | ✅ | ✅ | ✅ |
| &nbsp; &nbsp; &nbsp; &nbsp; src/utils/configuration.py                                        | ✅ | ✅ | ✅ |
| &nbsp; &nbsp; src/predict.py                                                                  | ✅ | 🟧 | 🟧 |
| &nbsp; &nbsp; src/pretrain.py                                                                 | ✅ | 🟧 | 🟧 |
| &nbsp; &nbsp; src/experiment.py                                                               | ✅ | 🟧 | 🟧 |
| &nbsp; &nbsp; src/finetune.py                                                                 | ✅ | 🟧 | 🟧 |
