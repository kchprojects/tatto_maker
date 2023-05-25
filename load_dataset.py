from datasets import load_dataset,Image
# from PIL.Image import Image


dataset = load_dataset("parquet", data_files={'train': '/home/karelch/Downloads/train-00000-of-00001-ca00c7703d9694f6.parquet'}).cast_column("image", Image(decode=True))


for d in dataset["train"]:
    d["image"].save("tatto_data/" + d["text"].replace(" ", "_")+".png")