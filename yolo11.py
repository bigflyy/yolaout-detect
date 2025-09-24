from itertools import islice
from datasets import load_dataset
import os
from ultralytics import YOLO
import kagglehub
import random, shutil
from pathlib import Path
import yaml

BASE_DIR = "/kaggle/input/sichpng/" # folder where our books test images are 
SAVE_DIR = "/kaggle/working/results_imgs" # folder where we save our test image folders 
SAVE_DIR = Path(SAVE_DIR)
SAVE_DIR.mkdir(parents=True, exist_ok=True)  # make sure folder exists

TEST_IMG_FOLDER_NAME = 50 # the name of folder inside results jpg folder, to symbolize like of which run this test images are from 
PROJECT_NAME = f"yolo11s" # folder where all runs are
RUN_NAME = f"yolo11s_run_" # name of current run inside project_name folder
# recomended: yolo{version}{size}_run_{number_of_epochs}_{accuracy}


# IMAGES FOR TEST STUFF
tables_imgs = ["suchgpt_page_0004-161.jpg", "suchgpt_page_0004-162.jpg"]
diagram_imgs = ["suchgpt_page_0001-014.jpg", "suchgpt_page_0001-029.jpg", "suchgpt_page_0001-030.jpg", "suchgpt_page_0001-034.jpg", "suchgpt_page_0001-037.jpg", "suchgpt_page_0001-040.jpg", "suchgpt_page_0001-043.jpg", "suchgpt_page_0002-068.jpg", "suchgpt_page_0002-069.jpg", "suchgpt_page_0002-072.jpg", "suchgpt_page_0002-074.jpg", "suchgpt_page_0002-087.jpg", "suchgpt_page_0003-116.jpg", "suchgpt_page_0003-118.jpg", "suchgpt_page_0003-120.jpg", "suchgpt_page_0003-124.jpg", "suchgpt_page_0003-128.jpg", "suchgpt_page_0003-129.jpg"]
equation_imgs = ["suchgpt_page_0001-012.jpg", "suchgpt_page_0003-136.jpg", "suchgpt_page_0003-137.jpg", "suchgpt_page_0003-139.jpg", "suchgpt_page_0003-140.jpg"]
list_imgs = ["suchgpt_page_0002-052.jpg", "suchgpt_page_0002-053.jpg", "suchgpt_page_0002-065.jpg", "suchgpt_page_0002-066.jpg", "suchgpt_page_0002-094.jpg", "suchgpt_page_0004-196.jpg"]

# Наши учебники
imgs_lists = [tables_imgs, diagram_imgs, equation_imgs]
imgs_lists_fixed = []
for imglist in imgs_lists:
    imgs_lists_fixed.append([BASE_DIR + img for img in imglist])   

# Учебники из датасета
test_imgs = [["/kaggle/input/doclaynet-v1-2-yolo/DocLayNet-v1.2-YOLODetection/images/test/075864.png", "/kaggle/input/doclaynet-v1-2-yolo/DocLayNet-v1.2-YOLODetection/images/test/075868.png"]]




def delete_files(rm_dir):
    #working_dir = "/kaggle/working"
    for f in os.listdir(rm_dir):
        path = os.path.join(rm_dir, f)
        if os.path.isfile(path) or os.path.islink(path):
            os.unlink(path)  # remove file or symlink
        elif os.path.isdir(path):
            shutil.rmtree(path)  # remove folder and contents


def test_model(model, imgs_lists, dir):
    """
    expects imgs_lists as full path
    model is yolo model 
    dir: the name of a folder inside SAVE_DIR that will 
    """
    global SAVE_DIR
    save_dir = SAVE_DIR / f"{dir}"
    save_dir.mkdir(parents=True, exist_ok=True) 
    imgs_shown = 0
    for imgs_list in imgs_lists:
        results = model(imgs_list)
        for i, result in enumerate(results):
            file_path = save_dir / f"image_{dir}_{imgs_shown}.jpg"
            result.save(filename=file_path)
            imgs_shown += 1

def fix_yaml():
    path = "/kaggle/input/doclaynet-v1-2-yolo/DocLayNet-v1.2-YOLODetection/data.yml"
    with open(path) as f:
        data = yaml.safe_load(f)

    # fix the paths
    root = "/kaggle/input/doclaynet-v1-2-yolo/DocLayNet-v1.2-YOLODetection"
    data["train"] = f"{root}/images/train"
    data["val"]   = f"{root}/images/validation"
    data["test"]  = f"{root}/images/test"

    # save to a new yaml
    new_path = "/kaggle/working/data_fixed.yml"
    with open(new_path, "w") as f:
        yaml.safe_dump(data, f)

    print("Fixed yaml saved to", new_path)

def zip_test_imgs(folder_to_zip = f"/kaggle/working/results_imgs/{TEST_IMG_FOLDER_NAME}", zip_path = f"/kaggle/working/results_imgs{TEST_IMG_FOLDER_NAME}.zip"):
    shutil.make_archive(base_name=zip_path.replace('.zip',''), format='zip', root_dir=folder_to_zip)

if __name__ == "__main__":
    # For dataset
    fix_yaml()

    # BASELINE, no training model
    model_base = YOLO("/kaggle/input/yolov11/pytorch/default/3/yolo11s.pt")
    test_model(model_base, imgs_lists_fixed)

    model_train = YOLO("/kaggle/input/yolov11/pytorch/default/3/yolo11s.pt")
    # Main train loop
    results = model_train.train(data="/kaggle/working/data_fixed.yml", epochs=20, batch=64, imgsz=640,  scale=0.8, mosaic=0.2, mixup=0.25, copy_paste=0.1, device="0,1", hsv_s=1.0, degrees=1.0, translate=0.1, fraction=0.17, project=PROJECT_NAME, name=RUN_NAME, exist_ok=True, save_period=1)
    try:
        model_last = YOLO(f"/kaggle/working/{PROJECT_NAME}/{RUN_NAME}/weights/last.pt")

        print(f"TEST from our books: ")
        test_model(model_last, imgs_lists_fixed)

        print(f"TEST from dataset: ")
        test_model(model_last, test_imgs)

        model_train = model_last
    except Exception as e:
        print(f'\n\n\n\nexcept {e}\n\n\n\n\n')

    # Transformer based model results 
    dima_model = YOLO("/kaggle/input/yolo12/pytorch/default/1/YOLO_v12m_ed_1_68_3.pt")
    test_model(dima_model, imgs_lists_fixed)