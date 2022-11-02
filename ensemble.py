import os.path as osp
import numpy as np

from ogb.lsc import PygPCQM4Mv2Dataset, PCQM4Mv2Evaluator


### automatic dataloading and splitting
dataset = PygPCQM4Mv2Dataset(root = '/home/cenyukuo/pcq-pos')

split_idx = dataset.get_idx_split()

### automatic evaluator. takes dataset name as input
evaluator = PCQM4Mv2Evaluator()

# y_true = torch.cat(y_true, dim = 0)
# y_pred = torch.cat(y_pred, dim = 0)

y_true = dataset.data.y[split_idx["valid"]].numpy()
# print((y_true.__dict__.keys()))
# print(y_true.shape)


input_path = "/home/cenyukuo/Transformer-M/logs/"
# names = ["L12", "L18"]
# names = ["L12", "L18", "L18_1", "L12_1", "L18_2", "L12_2"]
# ratio = [0.5, 0.5]
# ratio = [0.3, 0.5, 0.2]
# ratio = [0.3, 0.5, 0.15, 0.05]
# ratio = [0.15, 0.3, 0.05, 0.05, 0.3, 0.15] 0.07531613111495972
# ratio = [0.15, 0.3, 0.10, 0.00, 0.3, 0.15] # 0.07529587298631668

names = ["L18", "L18_2", "L18_3", "L18_4", "L12", "L12_2", "L12_3", "L12_4"]
# ratio = [0.15, 0.15, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1] # 0.07515674084424973
# ratio = [0.2, 0.2, 0.2, 0.2, 0.05, 0.05, 0.05, 0.05] # 0.07512176036834717
ratio = [0.3, 0.15, 0.15, 0.15, 0.1, 0.05, 0.05, 0.05] # 0.07503590732812881


# names = ["L18", "L18_1", "L18_2", "L18_3", "L18_4", "L12", "L12_2", "L12_3", "L12_4"]
# ratio = [0.3, 0.05, 0.15, 0.15, 0.15, 0.05, 0.05, 0.05, 0.05] # 0.07502983510494232

pred_vals = []
for idx, name in enumerate(names):
    valid_path = osp.join(input_path, name, f"{name}_pred_valid.npy")
    pred_val = np.load(valid_path)
    input_dict = {"y_true": y_true, "y_pred": pred_val}
    print(evaluator.eval(input_dict)["mae"])
    pred_vals.append(pred_val * ratio[idx])

avg_pred_vals = np.sum(pred_vals, 0)
print(avg_pred_vals.shape)


input_dict = {"y_true": y_true, "y_pred": avg_pred_vals}

avg_mae = evaluator.eval(input_dict)["mae"]
print(avg_mae)


pred_tests = []
for idx, name in enumerate(names):
    test_path = osp.join(input_path, name, f"{name}_pred_challenge.npy")
    pred_test = np.load(test_path)
    pred_tests.append(pred_test * ratio[idx])

avg_pred_tests = np.sum(pred_tests, 0)
print(avg_pred_tests.shape)
print(avg_pred_tests)
# print(np.sum(avg_pred_tests<0), np.sum(avg_pred_tests>20))

input_dict = {'y_pred': avg_pred_tests}
evaluator.save_test_submission(input_dict = input_dict, dir_path = "/home/cenyukuo/Transformer-M/submit/", mode = 'test-challenge')
