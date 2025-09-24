
from datasets import load_dataset
import os
from ultralytics import YOLO
import random
import datetime
import shutil
from pathlib import Path
import yaml

WORK_DIR = Path("/kaggle/working")
INPUT_DIR = Path("/kaggle/input")

SUCH_IMG_DIR = INPUT_DIR / Path("sichpng") # folder where our books test images are 
SAVE_DIR = WORK_DIR / "results_imgs" # folder where we save our test image folders 
SAVE_DIR = Path(SAVE_DIR)
SAVE_DIR.mkdir(parents=True, exist_ok=True)  # make sure folder exists

PROJECT_NAME = f"yolo11s" # folder where all runs are
RUN_NAME = f"yolo11s_run_" # name of current run inside project_name folder
# recomended: yolo{version}{size}_run_{number_of_epochs}_{accuracy}


# IMAGES FOR TEST STUFF
tables_imgs = ["suchgpt_page_0004-161.jpg", "suchgpt_page_0004-162.jpg"]
diagram_imgs = ["suchgpt_page_0001-014.jpg", "suchgpt_page_0001-029.jpg", "suchgpt_page_0001-030.jpg", "suchgpt_page_0001-034.jpg", "suchgpt_page_0001-037.jpg", "suchgpt_page_0001-040.jpg", "suchgpt_page_0001-043.jpg", "suchgpt_page_0002-068.jpg", "suchgpt_page_0002-069.jpg", "suchgpt_page_0002-072.jpg", "suchgpt_page_0002-074.jpg", "suchgpt_page_0002-087.jpg", "suchgpt_page_0003-116.jpg", "suchgpt_page_0003-118.jpg", "suchgpt_page_0003-120.jpg", "suchgpt_page_0003-124.jpg", "suchgpt_page_0003-128.jpg", "suchgpt_page_0003-129.jpg"]
equation_imgs = ["suchgpt_page_0001-012.jpg", "suchgpt_page_0003-136.jpg", "suchgpt_page_0003-137.jpg", "suchgpt_page_0003-139.jpg", "suchgpt_page_0003-140.jpg"]
list_imgs = ["suchgpt_page_0002-052.jpg", "suchgpt_page_0002-053.jpg", "suchgpt_page_0002-065.jpg", "suchgpt_page_0002-066.jpg", "suchgpt_page_0002-094.jpg", "suchgpt_page_0004-196.jpg"]

def append_base_dir_to_imgs_lists(imgs_lists, base_dir):
    imgs_lists_fixed=[]
    for imglist in imgs_lists:
        imgs_lists_fixed.append([base_dir / img for img in imglist])   
    return imgs_lists_fixed

# Наши учебники
imgs_lists = [tables_imgs, diagram_imgs, equation_imgs, list_imgs]
imgs_lists_fixed = append_base_dir_to_imgs_lists(imgs_lists, base_dir=SUCH_IMG_DIR)

# Учебники из датасета
test_imgs = [[INPUT_DIR / "doclaynet-v1-2-yolo/DocLayNet-v1.2-YOLODetection/images/test/075864.png", INPUT_DIR / "doclaynet-v1-2-yolo/DocLayNet-v1.2-YOLODetection/images/test/075868.png"]]




def delete_files(rm_dir):
    #working_dir = "/kaggle/working"
    for f in os.listdir(rm_dir):
        path = os.path.join(rm_dir, f)
        if os.path.isfile(path) or os.path.islink(path):
            os.unlink(path)  # remove file or symlink
        elif os.path.isdir(path):
            shutil.rmtree(path)  # remove folder and contents

def zip_folder(folder_to_zip, zip_name="zipped_imgs"):
    zip_path = SAVE_DIR / f"{zip_name}.zip"  # Fixed path handling
    shutil.make_archive(base_name=zip_path.replace('.zip',''), format='zip', root_dir=folder_to_zip)

def test_model(model, imgs_lists, dir):
    """
    imgs_lists: list of lists, in each list there are fulls paths to images 
    model: is yolo model 
    dir: the name of a folder inside SAVE_DIR that will represent this test, it will contain the results of model inference on imgs_lists images
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
    zip_folder(save_dir, "zipped_imgs_" + dir)
    

def fix_yaml():
    path = INPUT_DIR / "doclaynet-v1-2-yolo/DocLayNet-v1.2-YOLODetection/data.yml"
    with open(path) as f:
        data = yaml.safe_load(f)

    # fix the paths
    root = INPUT_DIR / "doclaynet-v1-2-yolo/DocLayNet-v1.2-YOLODetection"
    data["train"] = f"{root}/images/train"
    data["val"]   = f"{root}/images/validation"
    data["test"]  = f"{root}/images/test"

    # save to a new yaml
    new_path = WORK_DIR / "data_fixed.yml"
    with open(new_path, "w") as f:
        yaml.safe_dump(data, f)

    print("Fixed yaml saved to", new_path)



if __name__ == "__main__":
    # For dataset
    fix_yaml()

    # BASELINE, no training model
    model_base = YOLO(INPUT_DIR / "yolov11/pytorch/default/3/yolo11s.pt")
    test_model(model_base, imgs_lists_fixed, dir="our_book_base_model")

    model_train = YOLO(INPUT_DIR / "yolov11/pytorch/default/3/yolo11s.pt")
    # Main train loop
    results = model_train.train(data= WORK_DIR / "data_fixed.yml", epochs=1, batch=64, imgsz=640,  scale=0.8, mosaic=0.2, mixup=0.25, copy_paste=0.1, device="0,1", hsv_s=1.0, degrees=1.5, translate=0.1, fraction=0.01, project=PROJECT_NAME, name=RUN_NAME, exist_ok=True, save_period=1)
    try:
        model_last = YOLO( WORK_DIR / f"{PROJECT_NAME}/{RUN_NAME}/weights/last.pt")

        print(f"TEST from our books: ")
        test_model(model_last, imgs_lists_fixed, dir="our_book_train_model")

        print(f"TEST from dataset: ")
        test_model(model_last, test_imgs, dir="dataset_train_model")

        model_train = model_last
    except Exception as e:
        print(f'\n\n\n\nexcept {e}\n\n\n\n\n')

    zip_folder(folder_to_zip=WORK_DIR / PROJECT_NAME / RUN_NAME / "weights",)

    # Transformer based model results 
    dima_model = YOLO(INPUT_DIR / "yolo12/pytorch/default/1/YOLO_v12m_ed_1_68_3.pt")
    test_model(dima_model, imgs_lists_fixed, dir="our_book_dima_model")