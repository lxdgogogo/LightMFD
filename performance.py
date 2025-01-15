import argparse
import sys
from time import time

sys.path.append('')
from detector.improve_detector import GraphDetector, eval
from utils.utils import Dataset, save_results


def train(config: dict, graph_data):
    dataset = config['dataset']
    file_name = f"{dataset}_{config['training_ratio']}"
    start = time()
    detector = GraphDetector(graph_data['edge_index'], graph_data['nodes'], graph_data['labels'],
                             graph_data['mask'], graph_data['feat_data'], config)
    pred_y = detector.train(graph_data['train_idx'], graph_data['y_train'], graph_data['test_idx'])
    AUROC, AUPRC, RecK, f1_micro, f1_macro, recall, g_mean = eval(graph_data['y_test'], pred_y)
    end = time()
    model_result = {}
    model_result['AUROC'] = AUROC
    model_result['AUPRC'] = AUPRC
    model_result['f1_micro'] = f1_micro
    model_result['f1_macro'] = f1_macro
    model_result['RecK'] = RecK
    model_result['gmean'] = g_mean
    model_result['recall'] = recall
    model_result['Time'] = end - start
    save_results(config, model_result, file_name)
    return model_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LightMGFD')
    parser.add_argument("--dataset", type=str, default="ACM",
                        help="Dataset for this model (yelp/amazon)")  # ACM DBLP
    parser.add_argument("--train_ratio", type=float, default=0.05, help="Training ratio")  # 0.4
    parser.add_argument("--lr", type=float, default=0.02, help="Learning rate")
    parser.add_argument("--max_depth", type=int, default=12, help="Maximum tree depth of XGBoost")
    parser.add_argument("--T", type=int, default=200, help="Number of estimator trees of XGBoost")
    parser.add_argument("--theta", type=float, default=1, help="Parameter theta")
    parser.add_argument("--device", type=str, default='cpu', help="Device")
    args = parser.parse_args()
    train_config = {'n_estimators': args.T, 'learning_rate': args.lr, 'max_depth': args.max_depth,
                    'dataset': args.dataset, 'device': args.device, 'theta': args.theta, 'training_ratio': args.train_ratio}
    dataset = train_config['dataset']
    data = Dataset(dataset, train_config['device'])
    nodes, mask, y, train_idx, test_idx, y_train, y_test, feat_data, edge_index = data.process_data(
        train_config['training_ratio'], train_config['device'])
    graph_data = {'train_idx': train_idx, 'test_idx': test_idx, 'y_train': y_train, 'y_test': y_test,
                  'feat_data': feat_data, 'edge_index': edge_index, 'nodes': nodes, 'mask': mask, 'labels': y}
    print(args)
    train(train_config, graph_data)

