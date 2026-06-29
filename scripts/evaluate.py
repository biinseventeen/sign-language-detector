import zipfile
from src.config import *
from src.inference import evaluate
from src.model import TimeSformerClassifier

def main():
    model = TimeSformerClassifier(num_classes=NUM_CLASSES).to(DEVICE)
    
    # Public test
    evaluate(
        model=model,
        folder_path=os.path.join(LOCAL_EXTRACT_PATH, 'dataset/public_test'),
        label_to_idx_path=LABEL_MAP_PATH,
        model_path=SAVE_PATH,
        output_csv='public_submission.csv',
    )
    with zipfile.ZipFile('public_submission.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        z.write('public_submission.csv')
    print('✓ public_submission.zip')

    # Private test
    evaluate(
        model=model,
        folder_path=os.path.join(LOCAL_EXTRACT_PATH, 'dataset/private_test'),
        label_to_idx_path=LABEL_MAP_PATH,
        model_path=SAVE_PATH,
        output_csv='private_submission.csv',
    )
    with zipfile.ZipFile('private_test.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        z.write('private_submission.csv')
    print('✓ private_test.zip')
    
if __name__ == '__main__':
    main()