import os.path as osp
import numpy as np

from ogb.lsc import PygPCQM4Mv2Dataset, PCQM4Mv2Evaluator


### automatic dataloading and splitting
dataset = PygPCQM4Mv2Dataset(root = '/mnt/yrfs/yukuo/datasets/pcq-pos')

split_idx = dataset.get_idx_split()

### automatic evaluator. takes dataset name as input
evaluator = PCQM4Mv2Evaluator()

# y_true = torch.cat(y_true, dim = 0)
# y_pred = torch.cat(y_pred, dim = 0)

y_true = dataset.data.y[split_idx["valid"]].numpy()
# print((y_true.__dict__.keys()))
# print(y_true.shape)


input_path = "/data/yukuo/logs/"
names = ["L12", "L18", "L18_1", "L12_1"]
ratio = [0.3, 0.5, 0.15, 0.05]

pred_vals = []
for idx, name in enumerate(names):
    valid_path = osp.join(input_path, name, f"{name}_pred_val.npy")
    pred_val = np.load(valid_path)
    input_dict = {"y_true": y_true, "y_pred": pred_val}
    print(evaluator.eval(input_dict)["mae"])
    pred_vals.append(pred_val * ratio[idx])

avg_pred_vals = np.sum(pred_vals, 0)
print(avg_pred_vals.shape)


input_dict = {"y_true": y_true, "y_pred": avg_pred_vals}

avg_mae = evaluator.eval(input_dict)["mae"]
print(avg_mae)
