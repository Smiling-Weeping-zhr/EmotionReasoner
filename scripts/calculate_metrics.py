import os
import glob
import json
import numpy as np

from params import config
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

import os
import numpy as np
from utils.toolkit.read_files import *
from utils.toolkit.qwen import *
from utils.toolkit.functions import *
from utils.evaluation.wheel import func_get_name2reason
from dataset.mer2024 import MER2024_Dataset
from dataset.mer2023 import MER2023_Dataset
from dataset.cmu_mosei import CMU_MOSEI_Dataset
from dataset.cmu_mosi import CMU_MOSI_Dataset
from dataset.ch_sims import CH_SIMS_Dataset
from dataset.ch_sims_v2 import CH_SIMS_v2_Dataset
from dataset.meld import MELD_Dataset
from dataset.iemocap import IEMOCAP_Dataset
from dataset.ov_merd import OVMERD_Dataset
from utils.evaluation.ew_metric import *
from utils.evaluation.wheel import *
from omegaconf import OmegaConf

def get_dataset2cls(dataset):
    directory = f'/root/autodl-tmp/EMER/experiment/eval_affectgpt' # the disposition of metrics
    cfg = OmegaConf.load(os.path.join(directory, 'hparams.yaml'))
    
    if dataset == 'MER2023':     return MER2023_Dataset(cfg=cfg)
    if dataset == 'MER2024':     return MER2024_Dataset(cfg=cfg)
    if dataset == 'MELD':        return MELD_Dataset(cfg=cfg)
    if dataset == 'IEMOCAP': return IEMOCAP_Dataset(cfg=cfg)
    if dataset == 'CMU-MOSI':     return CMU_MOSI_Dataset(cfg=cfg)
    if dataset == 'CMU-MOSEI':    return CMU_MOSEI_Dataset(cfg=cfg)
    if dataset == 'CH-SIMS':        return CH_SIMS_Dataset(cfg=cfg)
    if dataset == 'CH-SIMS-v2':      return CH_SIMS_v2_Dataset(cfg=cfg)
    if dataset == 'OV-MERD':        return OVMERD_Dataset(cfg=cfg)
    print ('dataset cls not provided!')
    return None


def get_discrete_or_dimension_flag(dataset):
    if dataset in ['MER2023', 'MER2024', 'MELD', 'IEMOCAP']:
        return 'discrete'
    elif dataset in ['CMU-MOSI', 'CMU-MOSEI', 'CH-SIMS', 'CH-SIMS-v2']:
        return 'dimension'
    elif dataset in ['OV-MERD']:
        return 'ovlabel'
    else:
        ValueError('unsupported dataset input')


def get_emo2idx_idx2emo(dataset_cls):
    emo2idx, idx2emo = {}, {}

    if hasattr(dataset_cls, 'get_emo2idx_idx2emo'): 
        emo2idx, idx2emo = dataset_cls.get_emo2idx_idx2emo()
        # post process [不同数据集的标签表示有些许差异，进行统一化处理]
        if 'happy' in emo2idx: emo2idx['joy']   = emo2idx['happy']
        if 'anger' in emo2idx: emo2idx['angry'] = emo2idx['anger']
        if 'sad'   in emo2idx: emo2idx['sadness'] = emo2idx['sad']
        if 'joy'   in emo2idx: emo2idx['happy'] = emo2idx['joy']
        if 'angry' in emo2idx: emo2idx['anger'] = emo2idx['angry']
    return emo2idx, idx2emo

def func_read_batch_calling_model(modelname, device):
    # ipdb.set_trace()
    model_path = config.PATH_TO_LLM[modelname]
    llm = LLM(model=model_path, device=device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)
    return llm, tokenizer, sampling_params


## similarity score for: openset <-> discrete
def calculate_discrete_zeroshot(epoch_root, name2gt, llm, tokenizer, sampling_params, inter_print=True):
    # epoch_root=(name2reason) => openset
    openset_npz = epoch_root[:-4]+'-openset.npz'
    if not os.path.exists(openset_npz):
        extract_openset_batchcalling(reason_pkl=epoch_root, store_npz=openset_npz,
                                     llm=llm, tokenizer=tokenizer, sampling_params=sampling_params)
    # 计算 hitrate, mscore
    hitrate, mscore = hitrate_metric_calculation(name2gt=name2gt, openset_npz=openset_npz, inter_print=inter_print)
    return hitrate, mscore


def calculate_ov_zeroshot(epoch_root, name2gt, llm, tokenizer, sampling_params, inter_print=True):
    
    # epoch_root=(name2reason) => openset
    openset_npz = epoch_root[:-4]+'-openset.npz'
    if not os.path.exists(openset_npz):
        extract_openset_batchcalling(reason_pkl=epoch_root, store_npz=openset_npz,
                                     llm=llm, tokenizer=tokenizer, sampling_params=sampling_params)
        
    # 计算 EW-based metrics
    name2pred = {}
    filenames = np.load(openset_npz, allow_pickle=True)['filenames']
    fileitems = np.load(openset_npz, allow_pickle=True)['fileitems']
    for (name, item) in zip(filenames, fileitems):
        name2pred[name] = item
    fscore, precision, recall = wheel_metric_calculation(name2gt=name2gt, name2pred=name2pred, inter_print=inter_print)
    return fscore, precision, recall


## similarity score for: openset -> sentiment <-> sentiment
def calculate_dimension_zeroshot(epoch_root, name2gt, llm, tokenizer, sampling_params, inter_print=True):

    # 1. 抽取 openset
    openset_npz = epoch_root[:-4]+'-openset.npz'
    if not os.path.exists(openset_npz):
        extract_openset_batchcalling(reason_pkl=epoch_root, store_npz=openset_npz,
                                     llm=llm, tokenizer=tokenizer, sampling_params=sampling_params)
    
    # 2. 将 openset 转成 [positive, negative, neutral]
    sentiment_npz = openset_npz[:-4]+'-sentiment.npz'
    if not os.path.exists(sentiment_npz):
        openset_to_sentiment_batchcalling(openset_npz=openset_npz, store_npz=sentiment_npz,
                                          llm=llm, tokenizer=tokenizer, sampling_params=sampling_params)

    # 3. 计算 scores
    ## 3.0 openset 自然语言形式标签 -> float
    name2pred = {}
    filenames = np.load(sentiment_npz, allow_pickle=True)['filenames']
    fileitems = np.load(sentiment_npz, allow_pickle=True)['fileitems']
    for (name, item) in zip(filenames, fileitems):
        if item == 'positive':
            name2pred[name] = 1
        elif item == 'negative':
            name2pred[name] = -1
        elif item == 'neutral':
            name2pred[name] = 0
        else: # 其他无法操作的标签
            if inter_print: print ('error sample:', name, item)
            name2pred[name] = 0
    ## 3.1 conversion
    val_labels, val_preds = [], []
    for name in name2gt:
        val_labels.append(name2gt[name]) # name2gpt -> val_labels (float)
        val_preds.append(name2pred[name])
    val_labels = np.array(val_labels)
    val_preds = np.array(val_preds)
    ## 3.2 metric calculation (name2gt, name2pred) -> scores
    non_zeros = np.array([i for i, e in enumerate(val_labels) if e != 0]) # remove 0, and remove mask
    accuracy = accuracy_score((val_labels[non_zeros] > 0), (val_preds[non_zeros] > 0))
    fscore = f1_score((val_labels[non_zeros] > 0), (val_preds[non_zeros] > 0), average='weighted')
    return fscore, accuracy


def main_zeroshot_scores(input_dir, cfg, dataset, debug=False, test_epochs='', inter_print=True):

    if os.path.exists(input_dir) == False:
        raise ValueError(f'input_dir {input_dir} does not exist!')
    # read dataset infos
    disordim_flag = get_discrete_or_dimension_flag(dataset)
    if inter_print: print (f'process dataset: {dataset} => {disordim_flag}')
    dataset_cls = get_dataset2cls(dataset)
    name2gt = dataset_cls.get_test_name2gt()
    if inter_print: print (f'target sample number: {len(name2gt)}')

    # discrete: 自然语言形式标签；dimension: float score
    if disordim_flag == 'discrete':
        _, idx2emo = get_emo2idx_idx2emo(dataset_cls)
        # => update (name2gt)
        for name in name2gt:
            gt = name2gt[name]
            if not isinstance(gt, str):
                name2gt[name] = idx2emo[gt]
    # print (name2gt) # debug

    # load model
    llm, tokenizer, sampling_params = None, None, None
    if debug == False:
        llm, tokenizer, sampling_params = func_read_batch_calling_model(modelname='Qwen25', device=cfg.model.device)
    
    # main process
    whole_score1s, whole_score2s, whole_score3s = [], [], []
    for epoch_root in sorted(glob.glob(input_dir + '/*.pkl')):

        if epoch_root.find('openset') != -1 or epoch_root.find('sentiment') != -1:
            continue

        # =============== process for {epoch_root} ===============
        epochname = os.path.basename(epoch_root)
        if inter_print: print (epochname)

        # 1. score calculation
        if disordim_flag == 'discrete':
            hitrate, _ = calculate_discrete_zeroshot(epoch_root, name2gt, llm, tokenizer, sampling_params, inter_print)
            if inter_print: print(f'hitrate: {hitrate}')
            whole_score1s.append(hitrate)
            whole_score2s.append(0)
            whole_score3s.append(0)
            
        elif disordim_flag == 'dimension':
            fscore, acc = calculate_dimension_zeroshot(epoch_root, name2gt, llm, tokenizer, sampling_params, inter_print)
            if inter_print: print(f'fscore: {fscore}, acc: {acc}')
            whole_score1s.append(fscore)
            whole_score2s.append(acc)
            whole_score3s.append(0)
        
        elif disordim_flag == 'ovlabel':
            fscore, precision, recall = calculate_ov_zeroshot(epoch_root, name2gt, llm, tokenizer, sampling_params, inter_print)
            if inter_print: print(f'fscore: {fscore}, precision: {precision}, recall: {recall}')
            whole_score1s.append(fscore)
            whole_score2s.append(precision)
            whole_score3s.append(recall)

        if inter_print: print ('=========================')

    # whole_score1s => main metric
    best_index = np.argmax(whole_score1s)
    best_score1 = whole_score1s[best_index]
    best_score2 = whole_score2s[best_index]
    best_score3 = whole_score3s[best_index]
    if cfg.model_name.endswith('icl'):
        json_path = f'./experiment/eval_{cfg.model_name}/result_fewshot{cfg.icl.fewshot}.json'
    else:
        json_path = f'./experiment/eval_{cfg.model_name}/result.json'
        
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            results = json.load(f)
    else:
        results = {}
        
    if disordim_flag == 'discrete':
        if inter_print: print (f'{dataset}: best hitrate: %.4f; best mscore: %.4f' %(best_score1, best_score2))
        results[dataset] = {"dataset": dataset, "HIT Rate": best_score1}
    elif disordim_flag == 'dimension':
        if inter_print: print (f'{dataset}: best fscore: %.4f; best acc: %.4f' %(best_score1, best_score2))
        results[dataset] = {"dataset": dataset, "WAF": best_score1, "Accuracy": best_score2}
    elif disordim_flag == 'ovlabel':
        if inter_print: print (f'{dataset}: best fscore: %.4f; best precision: %.4f; best recall: %.4f' %(best_score1, best_score2,best_score3))
        results[dataset] = {"dataset": dataset, "Fscores": best_score1, "Precision": best_score2, "Recall": best_score3}
    
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
    # return the best scores
    return best_score1, best_score2, best_score3



def func_return_scores_one(modelname=None, dataset_candidates='affectgpt'):
    ## => (process datasets)
    if dataset_candidates=='affectgpt':
        process_datasets = ["mer2023", "mer2024", "meld", "iemocapfour", "cmumosi", "cmumosei", "sims", "simsv2"]
    elif dataset_candidates=='mer2025ov':
        process_datasets = ['mer2025ov']

    print_per_dataset, avg_score = [], []
    for dataset in process_datasets:
        process_root = f"output/results-{dataset}/{modelname}"
        ## 计算指标
        score1, score2, score3 = main_zeroshot_scores(process_root, debug=True, test_epochs='', inter_print=False)
        print_per_dataset.extend([score1])
        avg_score.append(score1)
    # append a avg value for ranking
    avg_score = np.mean(avg_score)
    print_per_dataset.append(avg_score)
    print_per_dataset = ["& %.2f" %(item*100) for item in print_per_dataset]
    return print_per_dataset, avg_score
