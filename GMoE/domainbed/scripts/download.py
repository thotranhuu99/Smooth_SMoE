# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from torchvision.datasets import MNIST
import xml.etree.ElementTree as ET
from zipfile import ZipFile
import argparse
import tarfile
import shutil
import gdown
import uuid
import json
import os

from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from wilds.datasets.fmow_dataset import FMoWDataset


# utils #######################################################################

def stage_path(data_dir, name):
    full_path = os.path.join(data_dir, name)

    if not os.path.exists(full_path):
        os.makedirs(full_path)

    return full_path


def download_and_extract(url, dst, remove=True):
    gdown.download(url, dst, quiet=False)

    if dst.endswith(".tar.gz"):
        tar = tarfile.open(dst, "r:gz")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".tar"):
        tar = tarfile.open(dst, "r:")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".zip"):
        zf = ZipFile(dst, "r")
        zf.extractall(os.path.dirname(dst))
        zf.close()

    if remove:
        os.remove(dst)


# VLCS ########################################################################

# Slower, but builds dataset from the original sources
#
# def download_vlcs(data_dir):
#     full_path = stage_path(data_dir, "VLCS")
#
#     tmp_path = os.path.join(full_path, "tmp/")
#     if not os.path.exists(tmp_path):
#         os.makedirs(tmp_path)
#
#     with open("domainbed/misc/vlcs_files.txt", "r") as f:
#         lines = f.readlines()
#         files = [line.strip().split() for line in lines]
#
#     download_and_extract("http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar",
#                          os.path.join(tmp_path, "voc2007_trainval.tar"))
#
#     download_and_extract("https://drive.google.com/uc?id=1I8ydxaAQunz9R_qFFdBFtw6rFTUW9goz",
#                          os.path.join(tmp_path, "caltech101.tar.gz"))
#
#     download_and_extract("http://groups.csail.mit.edu/vision/Hcontext/data/sun09_hcontext.tar",
#                          os.path.join(tmp_path, "sun09_hcontext.tar"))
#
#     tar = tarfile.open(os.path.join(tmp_path, "sun09.tar"), "r:")
#     tar.extractall(tmp_path)
#     tar.close()
#
#     for src, dst in files:
#         class_folder = os.path.join(data_dir, dst)
#
#         if not os.path.exists(class_folder):
#             os.makedirs(class_folder)
#
#         dst = os.path.join(class_folder, uuid.uuid4().hex + ".jpg")
#
#         if "labelme" in src:
#             # download labelme from the web
#             gdown.download(src, dst, quiet=False)
#         else:
#             src = os.path.join(tmp_path, src)
#             shutil.copyfile(src, dst)
#
#     shutil.rmtree(tmp_path)


def download_vlcs(data_dir):
    # Original URL: http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017
    full_path = stage_path(data_dir, "VLCS")

    download_and_extract("https://drive.google.com/uc?id=1skwblH1_okBwxWxmRsp9_qi15hyPpxg8",
                         os.path.join(data_dir, "VLCS.tar.gz"))


# MNIST #######################################################################

def download_mnist(data_dir):
    # Original URL: http://yann.lecun.com/exdb/mnist/
    full_path = stage_path(data_dir, "MNIST")
    MNIST(full_path, download=True)


# PACS ########################################################################

def download_pacs(data_dir):
    # Original URL: http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017
    full_path = stage_path(data_dir, "PACS")

    download_and_extract("https://drive.google.com/uc?id=1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd",
                         os.path.join(data_dir, "PACS.zip"))

    os.rename(os.path.join(data_dir, "kfold"),
              full_path)

import os
from pathlib import Path
from datasets import load_dataset
from PIL import Image
import torch

import os
from pathlib import Path
from datasets import load_dataset
from PIL import Image

def download_pacs_hf(data_dir):
    """
    Tải bộ dữ liệu PACS từ Hugging Face và lưu vào thư mục data_dir.

    Args:
        data_dir (str): Thư mục đích để lưu trữ bộ dữ liệu.

    Returns:
        str: Đường dẫn đầy đủ đến thư mục chứa bộ dữ liệu PACS.

    Raises:
        Exception: Nếu có lỗi trong quá trình tải hoặc lưu.
    """
    # Định nghĩa đường dẫn đích
    full_path = os.path.join(data_dir, "PACS")

    # Tạo thư mục đích nếu chưa tồn tại
    Path(full_path).mkdir(parents=True, exist_ok=True)

    # Tải dataset từ Hugging Face (sử dụng repo flwrlabs/pacs)
    print("Đang tải bộ dữ liệu PACS từ Hugging Face...")
    dataset = load_dataset("flwrlabs/pacs")

    # Dataset có splits: train (chứa tất cả dữ liệu, vì PACS không có splits train/test mặc định)
    # Cấu trúc: {'image': PIL.Image, 'domain': str, 'label': int}
    data = dataset['train']

    print(f"Đã tải dataset với {len(data)} mẫu.")

    # Tạo thư mục cho từng domain
    domains = {'photo': 'Photo', 'art_painting': 'Art Painting', 'cartoon': 'Cartoon', 'sketch': 'Sketch'}
    domain_folders = {}
    for domain_key, domain_name in domains.items():
        domain_dir = os.path.join(full_path, domain_name)
        Path(domain_dir).mkdir(exist_ok=True)
        domain_folders[domain_key] = domain_dir

    # Lưu từng ảnh vào thư mục tương ứng với domain
    print("Đang lưu ảnh vào thư mục...")
    for idx, item in enumerate(data):
        domain_key = item['domain'].lower().replace(' ', '_')  # e.g., 'art painting' -> 'art_painting'
        label = item['label']  # 7 classes: dog=0, elephant=1, giraffe=2, guitar=3, horse=4, house=5, person=6
        class_names = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
        class_name = class_names[label]

        # Đường dẫn ảnh
        img = item['image']
        img_path = os.path.join(domain_folders[domain_key], f"{class_name}_{idx}.jpg")

        # Chuyển ảnh sang định dạng RGB (để tránh lỗi với ảnh có palette hoặc mode không chuẩn)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Lưu ảnh dưới định dạng JPG, bỏ qua metadata EXIF để tránh lỗi
        img.save(img_path, 'JPEG', quality=95)

        if (idx + 1) % 1000 == 0:
            print(f"Đã lưu {idx + 1} ảnh...")

    print(f"Đã lưu tất cả ảnh vào {full_path}")

    # Tạo file README để mô tả cấu trúc
    readme_path = os.path.join(full_path, "README.txt")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("Bộ dữ liệu PACS (Domain Generalization)\n")
        f.write("Cấu trúc: PACS/domain/class/image.jpg\n")
        f.write("Domains: Photo, Art Painting, Cartoon, Sketch\n")
        f.write("Classes: dog, elephant, giraffe, guitar, horse, house, person\n")
        f.write("Nguồn: https://huggingface.co/datasets/flwrlabs/pacs\n")

    return full_path

import os
from pathlib import Path
import shutil

def reorganize_pacs(data_dir):
    """
    Sửa cấu trúc thư mục PACS thành: PACS/{domain}/{class}/{class}_{index}.jpg
    Index bắt đầu từ 1 cho mỗi class trong mỗi domain.

    Args:
        data_dir (str): Thư mục gốc chứa PACS (chứa PACS_hf).

    Returns:
        str: Đường dẫn đến thư mục PACS đã được tổ chức lại.
    """
    # Định nghĩa đường dẫn
    old_pacs_dir = os.path.join(data_dir, "PACS_hf")
    new_pacs_dir = os.path.join(data_dir, "PACS")

    # Kiểm tra thư mục PACS_hf tồn tại
    if not os.path.exists(old_pacs_dir):
        raise FileNotFoundError(f"Thư mục {old_pacs_dir} không tồn tại")

    # Tạo thư mục PACS mới nếu chưa tồn tại
    Path(new_pacs_dir).mkdir(exist_ok=True)

    # Định nghĩa domains và classes
    domain_mapping = {
        'P': 'Photo',
        'A': 'Art Painting',
        'C': 'Cartoon',
        'S': 'Sketch'
    }
    classes = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']

    print("Đang tổ chức lại cấu trúc thư mục PACS...")

    # Duyệt qua từng domain (P, A, C, S)
    for domain_short, domain_name in domain_mapping.items():
        old_domain_dir = os.path.join(old_pacs_dir, domain_short)
        if not os.path.exists(old_domain_dir):
            print(f"Không tìm thấy thư mục {old_domain_dir}, bỏ qua")
            continue

        # Tạo thư mục domain trong PACS
        new_domain_dir = os.path.join(new_pacs_dir, domain_name)
        Path(new_domain_dir).mkdir(exist_ok=True)

        # Tạo dictionary để đếm index cho mỗi class
        class_index = {cls: 1 for cls in classes}

        # Duyệt qua các file trong thư mục domain cũ
        for filename in os.listdir(old_domain_dir):
            if not filename.endswith('.jpg'):
                continue

            # Lấy class từ tên file (ví dụ: dog_4401.jpg -> dog)
            class_name = filename.split('_')[0]
            if class_name not in classes:
                print(f"Bỏ qua file không hợp lệ: {filename}")
                continue

            # Tạo thư mục class trong domain mới
            new_class_dir = os.path.join(new_domain_dir, class_name)
            Path(new_class_dir).mkdir(exist_ok=True)

            # Định nghĩa tên file mới với index bắt đầu từ 1
            new_filename = f"{class_name}_{class_index[class_name]}.jpg"
            old_file_path = os.path.join(old_domain_dir, filename)
            new_file_path = os.path.join(new_class_dir, new_filename)

            # Di chuyển và đổi tên file
            shutil.move(old_file_path, new_file_path)
            print(f"Di chuyển {old_file_path} -> {new_file_path}")

            # Tăng index cho class
            class_index[class_name] += 1

        print(f"Hoàn thành tổ chức domain {domain_name}")

    # Xóa thư mục PACS_hf cũ nếu rỗng
    for domain_short in domain_mapping.keys():
        old_domain_dir = os.path.join(old_pacs_dir, domain_short)
        if os.path.exists(old_domain_dir) and not os.listdir(old_domain_dir):
            shutil.rmtree(old_domain_dir)

    if os.path.exists(old_pacs_dir) and not os.listdir(old_pacs_dir):
        shutil.rmtree(old_pacs_dir)
        print(f"Đã xóa thư mục rỗng {old_pacs_dir}")

    # Cập nhật README
    readme_path = os.path.join(new_pacs_dir, "README.txt")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("Bộ dữ liệu PACS (Domain Generalization)\n")
        f.write("Cấu trúc: PACS/{domain}/{class}/{class}_{index}.jpg\n")
        f.write("Domains: Photo, Art Painting, Cartoon, Sketch\n")
        f.write("Classes: dog, elephant, giraffe, guitar, horse, house, person\n")
        f.write("Nguồn: https://huggingface.co/datasets/flwrlabs/pacs\n")

    print(f"Đã tổ chức lại cấu trúc vào {new_pacs_dir}")
    return new_pacs_dir


# Office-Home #################################################################

def download_office_home(data_dir):
    # Original URL: http://hemanthdv.org/OfficeHome-Dataset/
    full_path = stage_path(data_dir, "office_home")

    download_and_extract("https://drive.google.com/uc?id=1uY0pj7oFsjMxRwaD3Sxy0jgel0fsYXLC",
                         os.path.join(data_dir, "office_home.zip"))

    os.rename(os.path.join(data_dir, "OfficeHomeDataset_10072016"),
              full_path)


# DomainNET ###################################################################

def download_domain_net(data_dir):
    # Original URL: http://ai.bu.edu/M3SDA/
    full_path = stage_path(data_dir, "domain_net")

    urls = [
        "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip",
        "http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip",
        "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip",
        "http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip",
        "http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip",
        "http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip"
    ]

    for url in urls:
        download_and_extract(url, os.path.join(full_path, url.split("/")[-1]))

    with open("domainbed/misc/domain_net_duplicates.txt", "r") as f:
        for line in f.readlines():
            try:
                os.remove(os.path.join(full_path, line.strip()))
            except OSError:
                pass


#  ##############################################################

def download_terra_incognita_original(data_dir):
    # Original URL: https://beerys.github.io/CaltechCameraTraps/
    # New URL: http://lila.science/datasets/caltech-camera-traps

    full_path = stage_path(data_dir, "terra_incognita")

    download_and_extract(
        "https://lilablobssc.blob.core.windows.net/caltechcameratraps/eccv_18_all_images_sm.tar.gz",
        os.path.join(full_path, "terra_incognita_images.tar.gz"))

    download_and_extract(
        "https://lilablobssc.blob.core.windows.net/caltechcameratraps/labels/caltech_camera_traps.json.zip",
        os.path.join(full_path, "caltech_camera_traps.json.zip"))

    include_locations = ["38", "46", "100", "43"]

    include_categories = [
        "bird", "bobcat", "cat", "coyote", "dog", "empty", "opossum", "rabbit",
        "raccoon", "squirrel"
    ]

    images_folder = os.path.join(full_path, "eccv_18_all_images_sm/")
    annotations_file = os.path.join(full_path, "caltech_images_20210113.json")
    destination_folder = full_path

    stats = {}

    if not os.path.exists(destination_folder):
        os.mkdir(destination_folder)

    with open(annotations_file, "r") as f:
        data = json.load(f)

    category_dict = {}
    for item in data['categories']:
        category_dict[item['id']] = item['name']

    for image in data['images']:
        image_location = image['location']

        if image_location not in include_locations:
            continue

        loc_folder = os.path.join(destination_folder,
                                  'location_' + str(image_location) + '/')

        if not os.path.exists(loc_folder):
            os.mkdir(loc_folder)

        image_id = image['id']
        image_fname = image['file_name']

        for annotation in data['annotations']:
            if annotation['image_id'] == image_id:
                if image_location not in stats:
                    stats[image_location] = {}

                category = category_dict[annotation['category_id']]

                if category not in include_categories:
                    continue

                if category not in stats[image_location]:
                    stats[image_location][category] = 0
                else:
                    stats[image_location][category] += 1

                loc_cat_folder = os.path.join(loc_folder, category + '/')

                if not os.path.exists(loc_cat_folder):
                    os.mkdir(loc_cat_folder)

                dst_path = os.path.join(loc_cat_folder, image_fname)
                src_path = os.path.join(images_folder, image_fname)

                shutil.copyfile(src_path, dst_path)

    shutil.rmtree(images_folder)
    os.remove(annotations_file)

def download_terra_incognita(data_dir):

    full_path = stage_path(data_dir, "terra_incognita")

    # tải ảnh nhỏ
    download_and_extract(
        "https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/eccv_18_all_images_sm.tar.gz",
        os.path.join(full_path, "terra_incognita_images.tar.gz")
    )

    # tải metadata bằng link bạn vừa đưa
    download_and_extract(
        "https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/eccv_18_annotations.tar.gz",
        os.path.join(full_path, "terra_incognita_annotations.tar.gz")
    )

    include_locations = ["38", "46", "100", "43"]
    include_categories = [
        "bird", "bobcat", "cat", "coyote", "dog", "empty", "opossum", "rabbit",
        "raccoon", "squirrel"
    ]

    images_folder = os.path.join(full_path, "eccv_18_all_images_sm/")
    # Có thể file annotations sau khi giải nén sẽ có tên khác, bạn kiểm tra
    annotations_file = os.path.join(full_path, "eccv_18_annotations.json")
    destination_folder = full_path

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    with open(annotations_file, "r") as f:
        data = json.load(f)

    category_dict = { item['id']: item['name'] for item in data['categories'] }

    for image in data['images']:
        loc = image['location']
        if loc not in include_locations:
            continue

        image_fname = image['file_name']
        image_id = image['id']

        for ann in data['annotations']:
            if ann['image_id'] == image_id:
                category = category_dict[ann['category_id']]
                if category not in include_categories:
                    continue

                loc_cat_folder = os.path.join(destination_folder, f"location_{loc}", category)
                os.makedirs(loc_cat_folder, exist_ok=True)

                src = os.path.join(images_folder, image_fname)
                dst = os.path.join(loc_cat_folder, image_fname)
                if os.path.exists(src):
                    shutil.copyfile(src, dst)

    # cleanup
    shutil.rmtree(images_folder)
    os.remove(annotations_file)


def process_terra_incognita_dataset(data_dir):
    """
    Process Terra Incognita dataset into DomainBed-compatible format.
    """
    # Input paths
    images_folder = os.path.join(data_dir, "terra_incognita/eccv_18_all_images_sm")
    ann_dir = os.path.join(data_dir, "terra_incognita/eccv_18_annotation_files")

    # Output root folder
    destination_folder = os.path.join(data_dir, "terra_incognita_processed")
    os.makedirs(destination_folder, exist_ok=True)

    # Locations and categories we care about
    include_locations = ["38", "46", "100", "43"]
    include_categories = [
        "bird", "bobcat", "cat", "coyote", "dog",
        "empty", "opossum", "rabbit", "raccoon", "squirrel"
    ]

    # Collect annotation files
    ann_files = [
        "train_annotations.json",
        "cis_val_annotations.json",
        "cis_test_annotations.json",
        "trans_val_annotations.json",
        "trans_test_annotations.json"
    ]
    ann_files = [os.path.join(ann_dir, f) for f in ann_files if os.path.exists(os.path.join(ann_dir, f))]

    stats = {}

    for ann_path in ann_files:
        print(f"Processing {ann_path} ...")
        with open(ann_path, "r") as f:
            data = json.load(f)

        # Category mapping
        category_dict = {c["id"]: c["name"] for c in data["categories"]}

        for image in data["images"]:
            image_location = str(image["location"])
            if image_location not in include_locations:
                continue

            image_id = image["id"]
            image_fname = image["file_name"]

            # Find annotations for this image
            anns_for_image = [ann for ann in data["annotations"] if ann["image_id"] == image_id]

            for ann in anns_for_image:
                category = category_dict[ann["category_id"]]
                if category not in include_categories:
                    continue

                # Count stats
                stats.setdefault(image_location, {}).setdefault(category, 0)
                stats[image_location][category] += 1

                # Create folders
                loc_cat_folder = os.path.join(destination_folder, f"location_{image_location}", category)
                os.makedirs(loc_cat_folder, exist_ok=True)

                # Copy image
                src_path = os.path.join(images_folder, image_fname)
                dst_path = os.path.join(loc_cat_folder, image_fname)
                if os.path.exists(src_path) and not os.path.exists(dst_path):
                    shutil.copyfile(src_path, dst_path)

    print("✅ Dataset processing complete.")
    print("Stats per location/category:")
    print(json.dumps(stats, indent=2))


# SVIRO #################################################################

def download_sviro(data_dir):
    # Original URL: https://sviro.kl.dfki.de
    full_path = stage_path(data_dir, "sviro")

    download_and_extract("https://sviro.kl.dfki.de/?wpdmdl=1731",
                         os.path.join(data_dir, "sviro_grayscale_rectangle_classification.zip"))

    os.rename(os.path.join(data_dir, "SVIRO_DOMAINBED"),
              full_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download datasets')
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()

    # download_mnist(args.data_dir)
    # download_pacs(args.data_dir)
    # download_pacs_hf(args.data_dir)
    # reorganize_pacs(args.data_dir)
    # download_office_home(args.data_dir)
    # download_domain_net(args.data_dir)
    # download_vlcs(args.data_dir)
    # download_terra_incognita(args.data_dir)
    process_terra_incognita_dataset(args.data_dir)
    # download_sviro(args.data_dir)
    # Camelyon17Dataset(root_dir=args.data_dir, download=True)
    # FMoWDataset(root_dir=args.data_dir, download=True)
