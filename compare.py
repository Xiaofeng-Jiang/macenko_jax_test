import stainNorm_Macenko_jax as sm_jax
import stainNorm_Macenko as sm
import cv2
import numpy as np
import jax.numpy as jnp
from pathlib import Path
import argparse
import time


def get_normalizer(sampleImagePath, normalizer):
    target = cv2.imread(sampleImagePath)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

    normalizer.fit(target)
    return normalizer


if __name__ == '__main__':
    #parsing all arguments from the command line
    parser = argparse.ArgumentParser()
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument("-ip", "--inputPath", help="Input path of the to-be-normalised tiles", type=Path,
                               required=True)
    requiredNamed.add_argument("-op", "--outputPath", help="Output path to store normalised tiles", type=Path,
                               required=True)
    requiredNamed.add_argument("-op1", "--outputPath1", help="Output path to store normalised tiles", type=Path,
                               required=True)
    requiredNamed.add_argument("-op2", "--outputPath2", help="Output path to store normalised tiles", type=Path,
                               required=True)
    parser.add_argument("-si", "--sampleImagePath", help="Image used to determine the colour distribution, uses "
                                                         "GitHub one by default", type=str)
    parser.add_argument("-nt", "--threads", help="Number of threads used for processing, 2 by default", type=int)
    args = parser.parse_args()

    sampleImagePath = args.sampleImagePath
    input_path = args.inputPath
    output_path = args.outputPath
    output_path1 = args.outputPath1
    output_path2 = args.outputPath2

    normalizer = get_normalizer(sampleImagePath, sm.Normalizer())
    normalizer_jax = get_normalizer(sampleImagePath, sm_jax.Normalizer())

    #numpy 1st
    start_time = time.time()
    for test_img in input_path.glob('**/*.jpg'):
        img = cv2.imread(str(test_img))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        edge = cv2.Canny(img, 40, 100)
        edge = edge / np.max(edge) if np.max(edge) != 0 else 0
        edge = (np.sum(np.sum(edge)) / (img.shape[0] * img.shape[1])) * 100 if np.max(edge) != 0 else 0
        if edge > 2:
            nor_img = normalizer.transform(img)
            output_file = str(test_img).replace(str(input_path), str(output_path))
            Path(output_file).parent.mkdir(exist_ok=True, parents=True)
            cv2.imwrite(output_file, cv2.cvtColor(nor_img, cv2.COLOR_RGB2BGR))
    end_time = time.time()
    total_time = end_time - start_time
    print("Numpy 1st time:", total_time, "seconds")

    #jax 1st
    start_time = time.time()
    for test_img in input_path.glob('**/*.jpg'):
        img = cv2.imread(str(test_img))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        edge = cv2.Canny(img, 40, 100)
        edge = edge / jnp.max(edge) if jnp.max(edge) != 0 else 0
        edge = (jnp.sum(jnp.sum(edge)) / (img.shape[0] * img.shape[1])) * 100 if jnp.max(edge) != 0 else 0
        if edge > 2:
            nor_img = normalizer_jax.transform(img)
            nor_img = np.array(nor_img)
            output_file = str(test_img).replace(str(input_path), str(output_path1))
            Path(output_file).parent.mkdir(exist_ok=True, parents=True)
            cv2.imwrite(output_file, cv2.cvtColor(nor_img, cv2.COLOR_RGB2BGR))
    end_time = time.time()
    total_time = end_time - start_time
    print("Jax time:", total_time, "seconds")

    #numpy 2nd
    start_time = time.time()
    for test_img in input_path.glob('**/*.jpg'):
        img = cv2.imread(str(test_img))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        edge = cv2.Canny(img, 40, 100)
        edge = edge / np.max(edge) if np.max(edge) != 0 else 0
        edge = (np.sum(np.sum(edge)) / (img.shape[0] * img.shape[1])) * 100 if np.max(edge) != 0 else 0
        if edge > 2:
            nor_img = normalizer.transform(img)
            output_file = str(test_img).replace(str(input_path), str(output_path2))
            Path(output_file).parent.mkdir(exist_ok=True, parents=True)
            cv2.imwrite(output_file, cv2.cvtColor(nor_img, cv2.COLOR_RGB2BGR))
    end_time = time.time()
    total_time = end_time - start_time
    print("Numpy 2nd time:", total_time, "seconds")