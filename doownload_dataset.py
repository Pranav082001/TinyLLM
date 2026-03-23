import os
from datasets import load_dataset

HF_DATASET_ID = "HuggingFaceFW/fineweb-edu"
HF_DATASET_CONFIG = "sample-10BT" 
LOCAL_FOLDER_NAME = "fineweb_edu_10B"

BASE_SAVE_DIR = "/nethome/prku/pretraining_llm_group1/training_data" 
LOCAL_SAVE_PATH = os.path.join(BASE_SAVE_DIR, LOCAL_FOLDER_NAME)


def download_and_save_fineweb():

    print("---" * 15)
    print(f"Starting download of: **{LOCAL_FOLDER_NAME}**")
    print(f"  - HF ID: {HF_DATASET_ID}")
    print("---" * 15)

    os.makedirs(LOCAL_SAVE_PATH, exist_ok=True)
    
    try:
        dataset_dict = load_dataset(
            path=HF_DATASET_ID,
            name=HF_DATASET_CONFIG,
            trust_remote_code=True
        )
        
        dataset_dict.save_to_disk(LOCAL_SAVE_PATH)
        
        print("\n Download and save complete!")
        print(f"Dataset splits saved: {list(dataset_dict.keys())}")

    except Exception as e:
        print(f"   Error: {e}")

if __name__ == "__main__":
    download_and_save_fineweb()