import json

def convert_mscoco_json_to_tsv(input_file, output_file):
    in_file = open(input_file)
    json_data = json.load(in_file)

    annotations = json_data["annotations"]
    image_info = json_data["images"]
    img_id_to_name_info = {}

    n_imgs = 0
    for img in image_info:
        img_id_to_name_info[str(img["id"])] = img["file_name"]
        n_imgs += 1

    print("No of imgs: ", n_imgs)

    n_capts = 0
    with open(output_file, 'w', encoding='utf-8') as fo:
        for i, annot in enumerate(annotations):

            if i%100 == 0:
                print(i, " done")

            caption = annot["caption"]
            caption_id = str(annot["id"])
            img_id = str(annot["image_id"])
            img_name = img_id_to_name_info[str(img_id)]
            tsv_file_id = str(i)

            n_capts += 1

            fo.write(tsv_file_id + '\t' \
                    + img_name + '\t' \
                    + img_id + '\t' \
                    + caption + '\t' \
                    + caption_id + '\n')


    print("No of captions: ", n_capts)

if __name__ == "__main__":
    input_file = "/sr_img_data/users/saghotra/datasets/img/train/mscoco_train2017/annotations/captions_train2017.json"
    output_file = "/tmp/captions_train2017.tsv"
    convert_mscoco_json_to_tsv(input_file, output_file)
