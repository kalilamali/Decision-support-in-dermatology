import argparse
import os

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/fake_dataset', help="Directory with the SIGNS dataset")
parser.add_argument('--output_dir', default='data/64x64_SIGNS', help="Where to write the new data")
parser.add_argument('--size', default=224, help="size to resize", type=int)
parser.add_argument('--keep', default=False, help="keeping aspect ratio")



def run_fast_scandir(dir, ext):    # dir: str, ext: list
    """
    Function that takes a path and an extension
    and returns a list of file paths with that extension.
    """
    files = []
    for f in os.scandir(dir):
        if f.is_dir():
            subfolder = f.path
            subfiles = run_fast_scandir(subfolder, ext)
            files.extend(subfiles)
        if f.is_file():
            if os.path.splitext(f.name)[1].lower() in '.ds_store':
                pass
            else:
                if os.path.splitext(f.name)[1].lower() in ext:
                    files.append(f.path)

    return files

def resize_and_save(filename, output_dir, mysize, keep):
    mysize = int(mysize)
    """Resize the image contained in `filename` and save it to the `output_dir`"""
    image = Image.open(filename)
    if keep:
        height = mysize
        wpercent = (height/float(image.size[1]))

        width = int(image.size[0]*float(wpercent))
        image = image.resize((width, height), Image.ANTIALIAS)
    else:
    # Use bilinear interpolation instead of the default "nearest neighbor" method
        image = image.resize((mysize, mysize), Image.BILINEAR)
        #print(image.size)

    outfilename = filename.replace(args.data_dir, args.output_dir)

    outdirs = os.path.dirname(outfilename)
    if not os.path.exists(outdirs):
        os.makedirs(outdirs)

    image.save(outfilename)



if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    # Get the filenames in each directory (train and test)
    filenames = run_fast_scandir(args.data_dir, '.jpg')
    #print(filenames)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("Warning: output dir {} already exists".format(args.output_dir))

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    else:
        print("Warning: dir {} already exists".format(output_dir))

    for filename in tqdm(filenames):
        resize_and_save(filename, output_dir, mysize=args.size, keep=args.keep)

    print("Done resizing data")
