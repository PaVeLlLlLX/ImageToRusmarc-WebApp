import torch
from itertools import islice
import math
from tqdm import tqdm
from torch.utils.data import DataLoader
import Levenshtein
from model import ImageToRusmarcModel
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

def decode_sequence(token_ids, alphabet, sos_token_id, eos_token_id, pad_token_id):
    chars = []
    for token_id in token_ids:
        if isinstance(token_id, torch.Tensor):
             token_id = token_id.item()

        if token_id == eos_token_id:
            break 
            
        if token_id in [sos_token_id, pad_token_id]:
            continue

        if 0 <= token_id < len(alphabet):
            chars.append(alphabet[token_id])

    return "".join(chars)

def calculate_cer(model: ImageToRusmarcModel, 
                    dataloader: DataLoader, 
                    device, 
                    alphabet, 
                    sos_token_id, 
                    eos_token_id, 
                    pad_token_id,
                    subset_percentage: float = 0.1
                    ):
    
    if not (0 < subset_percentage <= 1.0):
        raise ValueError("subset_percentage must be between 0 (exclusive) and 1.0 (inclusive)")

    model.eval()
    total_cer = 0.0
    total_samples = 0
    
    try:
        num_batches_total = len(dataloader)
        if num_batches_total == 0:
            print("Warning: DataLoader is empty.")
            model.train()
            return 0.0
    except TypeError:
         print("Warning: DataLoader length not available, calculating CER on the full dataset.")
         subset_percentage = 1.0
         num_batches_total = None

    if subset_percentage < 1.0 and num_batches_total is not None:
        batches_to_take = max(1, math.ceil(num_batches_total * subset_percentage))
        print(f"Calculating CER on a subset: {batches_to_take}/{num_batches_total} batches ({subset_percentage*100:.1f}%).")
        subset_iterable = islice(dataloader, batches_to_take)
        total_for_tqdm = batches_to_take
    else:
        print("Calculating CER on the full dataset.")
        subset_iterable = dataloader
        total_for_tqdm = num_batches_total

    progress_bar = tqdm(subset_iterable, 
                        desc="Calculating CER", 
                        leave=True, 
                        ncols=100, 
                        total=total_for_tqdm)

    with torch.no_grad():
        for images, labels in progress_bar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            batch_size = images.size(0)
            
            current_max_len = labels.shape[1] + 10
            pred_token_ids = model.generate(images, max_len=current_max_len)
            pred_token_ids = pred_token_ids.cpu()
            labels = labels.cpu()

            batch_cer = 0.0
            for i in range(batch_size):
                pred_text = decode_sequence(pred_token_ids[i], alphabet, sos_token_id, eos_token_id, pad_token_id)
                true_text = decode_sequence(labels[i], dataloader.dataset.alphabet, dataloader.dataset.sos_token_id, dataloader.dataset.eos_token_id, dataloader.dataset.pad_token_id)
                
                distance = Levenshtein.distance(pred_text, true_text)
                cer = distance / max(len(true_text), 1) 
                batch_cer += cer

            total_cer += batch_cer
            total_samples += batch_size
            
            current_avg_cer = total_cer / total_samples if total_samples > 0 else 0.0
            progress_bar.set_postfix({'Avg CER': f'{current_avg_cer:.4f}'})

    model.train()

    if total_samples == 0:
       print("Warning: No samples processed for CER calculation in the subset.")
       return 0.0
       
    final_avg_cer = total_cer / total_samples
    return final_avg_cer