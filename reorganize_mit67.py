"""
Reorganize the MIT Indoor Scene 67 dataset into an ImageFolder-style layout.

Source layout (as distributed, see https://web.mit.edu/torralba/www/indoor.html):
    <data-dir>/Images/<scene>/<image>.jpg
    <data-dir>/TrainImages.txt   (each line: "<scene>/<image>.jpg")
    <data-dir>/TestImages.txt

Output layout:
    <out-dir>/train/<scene>/<image>.jpg   (from TrainImages.txt)
    <out-dir>/val/<scene>/<image>.jpg     (from TestImages.txt)

Example:
    python reorganize_mit67.py --data-dir ../datasets/mit67 --out-dir ../datasets/mit67_split
"""

import os
import shutil
import argparse


def load_split(split_file):
    """Return a list of "<scene>/<image>.jpg" entries from a split file."""
    with open(split_file) as f:
        return [line.strip().replace('\\', '/') for line in f if line.strip()]


def reorganize_split(entries, images_dir, dest_dir, mode):
    """Place every image listed in `entries` under dest_dir/<scene>/<image>."""
    placed, missing = 0, []
    for rel in entries:
        src = os.path.join(images_dir, *rel.split('/'))
        if not os.path.isfile(src):
            missing.append(rel)
            continue

        scene = rel.split('/')[0]
        scene_dir = os.path.join(dest_dir, scene)
        os.makedirs(scene_dir, exist_ok=True)
        dst = os.path.join(scene_dir, os.path.basename(rel))

        if mode == 'copy':
            shutil.copy2(src, dst)
        elif mode == 'move':
            shutil.move(src, dst)
        elif mode == 'symlink':
            if os.path.lexists(dst):
                os.remove(dst)
            os.symlink(os.path.abspath(src), dst)
        placed += 1

    return placed, missing


def main():
    parser = argparse.ArgumentParser(
        description='Reorganize MIT Indoor 67 into train/val ImageFolder dirs.')
    parser.add_argument('--data-dir', type=str, default='../datasets/mit67',
                        help='Root dir containing Images/, TrainImages.txt, TestImages.txt.')
    parser.add_argument('--out-dir', type=str, default='../datasets/mit67_split',
                        help='Destination root; train/ and val/ are created inside it.')
    parser.add_argument('--mode', type=str, default='copy',
                        choices=['copy', 'move', 'symlink'],
                        help='How to place files: copy (default), move, or symlink.')
    args = parser.parse_args()

    images_dir = os.path.join(args.data_dir, 'Images')
    if not os.path.isdir(images_dir):
        images_dir = args.data_dir  # fall back to scene folders directly under data-dir

    splits = {
        'train': os.path.join(args.data_dir, 'TrainImages.txt'),
        'val': os.path.join(args.data_dir, 'TestImages.txt'),
    }

    for split_name, split_file in splits.items():
        if not os.path.isfile(split_file):
            raise FileNotFoundError(f'Split file not found: {split_file}')

        entries = load_split(split_file)
        dest_dir = os.path.join(args.out_dir, split_name)
        placed, missing = reorganize_split(entries, images_dir, dest_dir, args.mode)

        n_scenes = len(os.listdir(dest_dir)) if os.path.isdir(dest_dir) else 0
        print(f'[{split_name}] {placed}/{len(entries)} images -> {dest_dir} '
              f'({n_scenes} scene folders)')
        if missing:
            print(f'  WARNING: {len(missing)} listed images were not found, e.g. {missing[:3]}')

    print('Done.')


if __name__ == '__main__':
    main()
