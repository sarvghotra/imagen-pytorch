from data.img_text_dataset import DatasetImgText, DatasetText

def create_dataset(opts):
    if opts['dataset_type'] == "imgtext":
        return DatasetImgText(opts)
    elif opts['dataset_type'] == "text":
        return DatasetText(opts)
