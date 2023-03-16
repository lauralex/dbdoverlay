import collections
import json
import os.path

import datasets
from datasets.download.download_manager import FilesIterable

_DESCRIPTION = """\
DBD Icons Dataset
"""

_CITATION = """\
@inproceedings{DBDIcons,
    title={DBD Icons: A Large-Scale Dataset for Icon Detection and Recognition},
    author={Alessandro Bellia},
    year={2022}
}
"""

_CATEGORIES = ['bleeding', 'blessed', 'blindness', 'bloodlust', 'broken', 'cursed', 'deepWound', 'endurance',
               'exhaustion', 'exposed', 'gliph', 'haste', 'hearing', 'hindered', 'incapacitated', 'madness', 'mangled',
               'oblivious', 'sleepPenalty', 'undetectable', 'vision']


class DbdIconsLoader(datasets.GeneratorBasedBuilder):
    """DBD Icons Dataset."""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "image_id": datasets.Value("int64"),
                    "image": datasets.Image(),
                    "width": datasets.Value("int32"),
                    "height": datasets.Value("int32"),
                    "objects": datasets.Sequence(
                        {
                            "id": datasets.Value("int64"),
                            "area": datasets.Value("int64"),
                            "bbox": datasets.Sequence(datasets.Value("float32"), length=4),
                            "category": datasets.ClassLabel(names=_CATEGORIES),
                        }
                    ),

                }
            ),
            citation=_CITATION
        )

    def _split_generators(self, dl_manager):
        files = FilesIterable.from_paths('coco/images')
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "annotation_file_path": 'train.json',
                    "files": files
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "annotation_file_path": 'val.json',
                    "files": files
                },
            ),
        ]

    def _generate_examples(self, annotation_file_path, files):
        def process_annot(annot, category_id_to_gategory):
            return {
                "id": annot["id"],
                "area": annot["area"],
                "bbox": annot["bbox"],
                "category": category_id_to_gategory[annot["category_id"]],
            }

        image_id_to_image = {}
        idx = 0

        for path in files:
            file_name = os.path.basename(path)
            with open(path, "rb") as f:
                if annotation_file_path in path:
                    annotations = json.load(f)
                    category_id_to_gategory = {c["id"]: c["name"] for c in annotations["categories"]}
                    image_id_to_annotations = collections.defaultdict(list)
                    for annot in annotations["annotations"]:
                        image_id_to_annotations[annot["image_id"]].append(annot)
                    image_id_to_image = {annot['file_name']: annot for annot in annotations['images']}
                elif file_name in image_id_to_image:
                    image = image_id_to_image[file_name]
                    objects = [process_annot(annot, category_id_to_gategory) for annot in
                               image_id_to_annotations[image["id"]]]
                    yield idx, {
                        "image_id": image["id"],
                        "image": {"path": path, "bytes": f.read()},
                        "width": image["width"],
                        "height": image["height"],
                        "objects": objects,
                    }
                    idx += 1

#generators = list(FilesIterable.from_paths('coco/images'))
#cocco = list(DbdIconsLoader()._generate_examples('train.json', generators))
#print('ciao')