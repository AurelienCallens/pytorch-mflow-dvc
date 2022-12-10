import yaml
import pytorch_lightning as pl
from src.models.train_model import LightDataset
from sklearn.metrics import classification_report

if __name__ == '__main__':
    # Load num to labels map
    label_map = yaml.safe_load(open('./data/processed/labels_map.yaml'))
    # Replace with model load 
    test_dataloader = LightDataset(batch_size=1).test_dataloader()
    pred = trainer.predict(model, test_dataloader)



    y_pred = list(map(lambda x: x.item(), pred))

    y_true = []
    for test_data in test_dataloader:
        y_true.append(test_data[1].item())

    print(classification_report(y_true, y_pred, digits=4, target_names=label_map.values()))
