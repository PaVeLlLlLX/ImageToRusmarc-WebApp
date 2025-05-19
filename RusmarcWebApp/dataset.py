import torch
import re
from PIL import Image
from torchvision import transforms
from pathlib import Path
from torch.utils.data import Dataset
from functools import lru_cache

class GPNTBDataset(Dataset):
    def __init__(self, images_dir, txt_file, transform=None):
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.max_seq_len = 2400
        self.cards = self._load_cards(txt_file)
        self.image_paths = self._validate_image_paths()
        # " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_abcdefghijklmnopqrstuvwxyz{|}~¦§©«®°±»ЁЎАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяёљў—„•€№™"
        self.alphabet, self.char_to_idx = self._build_alphabet()
        #print(len(self.alphabet), self.alphabet)

    def _load_cards(self, txt_file):
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                return f.read().split("*****")
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки файла {txt_file}: {str(e)}")

    def _validate_image_paths(self):
        paths = []
        pattern = re.compile(r"([^\\/]+)[\\/]([^\\/]+)[\\/]([^\\/]+\.jpg)")
        
        for idx, card in enumerate(self.cards):
            match = pattern.search(card)
            if not match:
                raise ValueError(f"Неверный формат карточки {idx}: {card}")
            rel_path = Path(*match.groups())
            full_path = self.images_dir / rel_path
            if len(self.cards[idx]) > self.max_seq_len or len(self.cards[idx]) == 0:
                print(len(self.cards[idx]))
                #print(self.cards[idx])
                #self.cards.pop(idx)
                #continue
            if not full_path.exists():
                raise FileNotFoundError(f"Изображение {full_path} не найдено")
                #print(self.cards[idx])
                #self.cards.pop(idx)
            paths.append(full_path)

        return paths
    
    def _build_alphabet(self):
        unique_chars = set()
        
        for i, card in enumerate(self.cards):
            card_chars = set(card)
            unique_chars.update(card_chars)
        uc = unique_chars.copy()
        for char in unique_chars:
            if char in "@QWYqЎљў":
                uc.remove(char)

        unique_chars = uc

        sorted_chars = sorted(list(unique_chars))

        special_tokens = ['@', '<SOS>', '<EOS>'] # '@' - PAD
        for sp in special_tokens:
            if sp in sorted_chars:
                sorted_chars.remove(sp)

        final_alphabet = special_tokens + sorted_chars
        char_mapping = {char: idx for idx, char in enumerate(final_alphabet)}

        self.pad_token_id = char_mapping['@']
        self.sos_token_id = char_mapping['<SOS>']
        self.eos_token_id = char_mapping['<EOS>']
        print(f"PAD ID: {self.pad_token_id}, SOS ID: {self.sos_token_id}, EOS ID: {self.eos_token_id}")
        print(f"Размер словаря: {len(final_alphabet)}")
        
        print(char_mapping['@'])
        return final_alphabet, char_mapping

    def __len__(self):
        return len(self.cards)

    def __getitem__(self, idx):
        image = self._load_image(self.image_paths[idx])
        target_sequence_str = self.alphabet[self.sos_token_id] + self.cards[idx] + self.alphabet[self.eos_token_id]
        label = self._encode_text(target_sequence_str)
        # print(self.cards[idx])
        # plt.imshow(image[0], cmap='gray')
        # plt.axis('off')
        # plt.show()
        return image, torch.tensor(label, dtype=torch.long)

    @lru_cache(maxsize=1000)
    def _load_image(self, path):
        try:
            image = Image.open(path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки изображения {path}: {str(e)}")

    def _encode_text(self, text):
        return [self.char_to_idx.get(char, self.pad_token_id) for char in text]


transform = transforms.Compose([
    #transforms.RandomRotation(4),
    #transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
    transforms.GaussianBlur(kernel_size=(3, 3)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Resize((384, 512)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((384, 512)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])