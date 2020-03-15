
import os, ast
import logging
from tqdm import tqdm


logging.basicConfig(filename='train.log',level=logging.INFO)

DEVICE = "cuda"
CUDA_VISIBILE_DEVICES=0
IMG_HEIGHT=137
IMG_WIDTH=236
EPOCHS=10
TRAIN_BATCH_SIZE=256
TEST_BATCH_SIZE=64
MODEL_MEAN=(0.485, 0.456, 0.406)
MODEL_STD=(0.229, 0.224, 0.225)
TRAINING_FOLDS=(0, 1, 2, 3)
VALIDATION_FOLDS=(4, )
BASE_MODEL='bengali'


def loss_fn(outputs, targets):
    o1, o2, o3 = outputs
    t1, t2, t3 = targets

    l1 = nn.CrossEntropyLoss()(o1, t1)
    l2 = nn.CrossEntropyLoss()(o2, t2)
    l3 = nn.CrossEntropyLoss()(o3, t3)
    return (l1 + l2 + l3)


def train(dataset, data_loader, model, optimizer):
     model.train()

     for bi, data in tqdm(enumerate(data_loader), total=int(len(dataset) / data_loader.batch_size)):
        logging.info(f'TRAINING BATCH: {bi}/ {int(len(dataset) / data_loader.batch_size)}')
        image = data['image']
        grapheme_root = data['grapheme_root']
        vowel_diacritic = data['vowel_diacritic']
        consonant_diacritic = data['consonant_diacritic']

        image = image.to(DEVICE, dtype=torch.float)
        grapheme_root = grapheme_root.to(DEVICE, dtype=torch.long)
        vowel_diacritic = vowel_diacritic.to(DEVICE, dtype=torch.long)
        consonant_diacritic = consonant_diacritic.to(DEVICE, dtype=torch.long)

        optimizer.zero_grad()

        outputs = model(image)
        targets = (grapheme_root, vowel_diacritic, consonant_diacritic)
        loss = loss_fn(outputs, targets)

        loss.backward()
        optimizer.step()


def evaluate(dataset, data_loader, model):
    model.eval()
    
    final_loss = 0
    with torch.no_grad():
        for bi, data in tqdm(enumerate(data_loader), total=int(len(dataset) / data_loader.batch_size)):
            logging.info(f'EVALUATING BATCH: {bi}/ {int(len(dataset) / data_loader.batch_size)}')
            image = data['image']
            grapheme_root = data['grapheme_root']
            vowel_diacritic = data['vowel_diacritic']
            consonant_diacritic = data['consonant_diacritic']

            image = image.to(DEVICE, dtype=torch.float)
            grapheme_root = grapheme_root.to(DEVICE, dtype=torch.long)
            vowel_diacritic = vowel_diacritic.to(DEVICE, dtype=torch.long)
            consonant_diacritic = consonant_diacritic.to(DEVICE, dtype=torch.long)

            outputs = model(image)
            targets = (grapheme_root, vowel_diacritic, consonant_diacritic)
            loss = loss_fn(outputs, targets)
            final_loss += loss

    return final_loss / bi

torch.cuda.empty_cache()
model = Resnet34(pretrained=True)
model.load_state_dict(torch.load('bengali_fold_4.bin'))
model.to(DEVICE)

train_dataset = BengaliDataset(
    folds=TRAINING_FOLDS,
    img_height=IMG_HEIGHT,
    img_width=IMG_WIDTH,
    mean=MODEL_MEAN,
    std=MODEL_STD
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    num_workers=4
)

validation_dataset = BengaliDataset(
    folds=VALIDATION_FOLDS,
    img_height=IMG_HEIGHT,
    img_width=IMG_WIDTH,
    mean=MODEL_MEAN,
    std=MODEL_STD
)

validation_loader = torch.utils.data.DataLoader(
    dataset=validation_dataset,
    batch_size=TEST_BATCH_SIZE,
    shuffle=True,
    num_workers=4
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.3, verbose=True)

if torch.cuda.device_count() > 1:
    model = nn.DataParallel()

for epoch in range(EPOCHS):
    train(train_dataset, train_loader, model, optimizer)
    val_score = evaluate(validation_dataset, validation_loader, model)
    print(val_score)
    scheduler.step(val_score)
    logging.info(f'EPOCH: {epoch}, VAL_SCORE:{val_score}')
    torch.save(model.state_dict(), f"{BASE_MODEL}_fold_{VALIDATION_FOLDS[0]}.bin")