import numpy as np

from .builder import DATASETS
from .base_dataset import BaseDataset

@DATASETS.register_module()
class myfilelist(BaseDataset):
    CLASSES = [
        'tench, Tinca tinca',
        'goldfish, Carassius auratus',
        'great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias',  # noqa: E501
        'tiger shark, Galeocerdo cuvieri',
        'hammerhead, hammerhead shark',
        'electric ray, crampfish, numbfish, torpedo',
        'stingray',
        'cock',
        'hen',
        'ostrich, Struthio camelus',
        'brambling, Fringilla montifringilla',
        'goldfinch, Carduelis carduelis',
        'house finch, linnet, Carpodacus mexicanus',
        'junco, snowbird',
        'indigo bunting, indigo finch, indigo bird, Passerina cyanea',
        'robin, American robin, Turdus migratorius',
        'bulbul',
        'jay',
        'magpie',
        'chickadee',
        'water ouzel, dipper',
        'kite',
        'bald eagle, American eagle, Haliaeetus leucocephalus',
        'vulture',
        'great grey owl, great gray owl, Strix nebulosa',
        'European fire salamander, Salamandra salamandra',
        'common newt, Triturus vulgaris',
        'eft',
        'spotted salamander, Ambystoma maculatum',
        'axolotl, mud puppy, Ambystoma mexicanum',
        'bullfrog, Rana catesbeiana',
        'tree frog, tree-frog',
        'tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui',
        'loggerhead, loggerhead turtle, Caretta caretta',
        'leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea',  # noqa: E501
        'mud turtle',
        'terrapin',
        'box turtle, box tortoise',
        'banded gecko',
        'common iguana, iguana, Iguana iguana',
        'American chameleon, anole, Anolis carolinensis',
        'whiptail, whiptail lizard',
        'agama',
        'frilled lizard, Chlamydosaurus kingi',
        'alligator lizard',
        'Gila monster, Heloderma suspectum',
        'green lizard, Lacerta viridis',
        'African chameleon, Chamaeleo chamaeleon',
        'Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis',  # noqa: E501
        'African crocodile, Nile crocodile, Crocodylus niloticus',
        'American alligator, Alligator mississipiensis',
        'triceratops',
        'thunder snake, worm snake, Carphophis amoenus',
        'ringneck snake, ring-necked snake, ring snake',
        'hognose snake, puff adder, sand viper',
        'green snake, grass snake',
        'king snake, kingsnake',
        'garter snake, grass snake',
        'water snake',
        'vine snake',
        'night snake, Hypsiglena torquata',
        'boa constrictor, Constrictor constrictor',
        'rock python, rock snake, Python sebae',
        'Indian cobra, Naja naja',
        'green mamba',
        'sea snake',
        'horned viper, cerastes, sand viper, horned asp, Cerastes cornutus',
        'diamondback, diamondback rattlesnake, Crotalus adamanteus',
        'sidewinder, horned rattlesnake, Crotalus cerastes',
        'trilobite',
        'harvestman, daddy longlegs, Phalangium opilio',
        'scorpion',
        'black and gold garden spider, Argiope aurantia',
        'barn spider, Araneus cavaticus',
        'garden spider, Aranea diademata',
        'black widow, Latrodectus mactans',
        'tarantula',
        'wolf spider, hunting spider',
        'tick',
        'centipede',
        'black grouse',
        'ptarmigan',
        'ruffed grouse, partridge, Bonasa umbellus',
        'prairie chicken, prairie grouse, prairie fowl',
        'peacock',
        'quail',
        'partridge',
        'African grey, African gray, Psittacus erithacus',
        'macaw',
        'sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita',
        'lorikeet',
        'coucal',
        'bee eater',
        'hornbill',
        'hummingbird',
        'jacamar',
        'toucan',
        'drake',
        'red-breasted merganser, Mergus serrator',
        'goose',
        'black swan, Cygnus atratus',
        'tusker',
    ]

    def load_annotations(self):
        assert isinstance(self.ann_file, str)

        data_infos = []
        with open(self.ann_file) as f:
            samples = [x.strip().split(' ') for x in f.readlines()]
            for filename, gt_label in samples:
                info = {'img_prefix': self.data_prefix}
                info['img_info'] = {'filename': filename}
                info['gt_label'] = np.array(gt_label, dtype=np.int64)
                data_infos.append(info)
            return data_infos