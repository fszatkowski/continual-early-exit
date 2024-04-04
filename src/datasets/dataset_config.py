from os.path import join

_BASE_DATA_PATH = "../data"

datasets_root = _BASE_DATA_PATH

dataset_config = {
    "flowers": {
        "path": join(datasets_root, "flowers/"),
        "resize": (256, 256),
        "crop": 224,
        "flip": True,
        "normalize": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    },
    "scenes": {
        "path": join(datasets_root, "indoor_scenes/"),
        "resize": (256, 256),
        "crop": 224,
        "flip": True,
        "normalize": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    },
    "birds": {
        "path": join(datasets_root, "cubs_cropped/"),
        "resize": (256, 256),
        "crop": 224,
        "flip": True,
        "normalize": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    },
    "cars": {
        "path": join(datasets_root, "stanford_cars_cropped/"),
        "resize": (256, 256),
        "crop": 224,
        "flip": True,
        "normalize": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    },
    "aircrafts": {
        "path": join(datasets_root, "aircrafts_cropped/"),
        "resize": (256, 256),
        "crop": 224,
        "flip": True,
        "normalize": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    },
    "actions": {
        "path": join(datasets_root, "Stanford40/"),
        "resize": (256, 256),
        "crop": 224,
        "flip": True,
        "normalize": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    },
    "mnist": {
        "path": join(_BASE_DATA_PATH, "mnist"),
        "normalize": ((0.1307,), (0.3081,)),
        # Use the next 3 lines to use MNIST with a 3x32x32 input
        # 'extend_channel': 3,
        # 'pad': 2,
        # 'normalize': ((0.1,), (0.2752,))    # values including padding
    },
    "svhn": {
        "path": join(_BASE_DATA_PATH, "svhn"),
        "resize": (224, 224),
        "crop": None,
        "flip": False,
        "normalize": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    },
    "cifar100": {
        "path": join(_BASE_DATA_PATH, "cifar100"),
        "resize": None,
        "pad": 4,
        "crop": 32,
        "flip": True,
        "normalize": ((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
    },
    "cifar100_fixed": {
        "path": join(_BASE_DATA_PATH, "cifar100"),
        "resize": None,
        "pad": 4,
        "crop": 32,
        "flip": True,
        "normalize": ((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
        "class_order": [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            68,
            69,
            70,
            71,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
            80,
            81,
            82,
            83,
            84,
            85,
            86,
            87,
            88,
            89,
            90,
            91,
            92,
            93,
            94,
            95,
            96,
            97,
            98,
            99,
        ],
    },
    "cifar100_icarl": {
        "path": join(_BASE_DATA_PATH, "cifar100"),
        "resize": None,
        "pad": 4,
        "crop": 32,
        "flip": True,
        "normalize": ((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
        "class_order": [
            68,
            56,
            78,
            8,
            23,
            84,
            90,
            65,
            74,
            76,
            40,
            89,
            3,
            92,
            55,
            9,
            26,
            80,
            43,
            38,
            58,
            70,
            77,
            1,
            85,
            19,
            17,
            50,
            28,
            53,
            13,
            81,
            45,
            82,
            6,
            59,
            83,
            16,
            15,
            44,
            91,
            41,
            72,
            60,
            79,
            52,
            20,
            10,
            31,
            54,
            37,
            95,
            14,
            71,
            96,
            98,
            97,
            2,
            64,
            66,
            42,
            22,
            35,
            86,
            24,
            34,
            87,
            21,
            99,
            0,
            88,
            27,
            18,
            94,
            11,
            12,
            47,
            25,
            30,
            46,
            62,
            69,
            36,
            61,
            7,
            63,
            75,
            5,
            32,
            4,
            51,
            48,
            73,
            93,
            39,
            67,
            29,
            49,
            57,
            33,
        ],
    },
    "vggface2": {
        "path": join(_BASE_DATA_PATH, "VGGFace2"),
        "resize": 256,
        "crop": 224,
        "flip": True,
        "normalize": ((0.5199, 0.4116, 0.3610), (0.2604, 0.2297, 0.2169)),
    },
    "imagenet_256": {
        "path": join(_BASE_DATA_PATH, "ILSVRC12_256"),
        "resize": None,
        "crop": 224,
        "flip": True,
        "normalize": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    },
    "imagenet_subset": {
        "path": join(_BASE_DATA_PATH, "ILSVRC12_256"),
        "resize": None,
        "crop": 224,
        "flip": True,
        "normalize": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        "class_order": [
            68,
            56,
            78,
            8,
            23,
            84,
            90,
            65,
            74,
            76,
            40,
            89,
            3,
            92,
            55,
            9,
            26,
            80,
            43,
            38,
            58,
            70,
            77,
            1,
            85,
            19,
            17,
            50,
            28,
            53,
            13,
            81,
            45,
            82,
            6,
            59,
            83,
            16,
            15,
            44,
            91,
            41,
            72,
            60,
            79,
            52,
            20,
            10,
            31,
            54,
            37,
            95,
            14,
            71,
            96,
            98,
            97,
            2,
            64,
            66,
            42,
            22,
            35,
            86,
            24,
            34,
            87,
            21,
            99,
            0,
            88,
            27,
            18,
            94,
            11,
            12,
            47,
            25,
            30,
            46,
            62,
            69,
            36,
            61,
            7,
            63,
            75,
            5,
            32,
            4,
            51,
            48,
            73,
            93,
            39,
            67,
            29,
            49,
            57,
            33,
        ],
    },
    "imagenet_32_reduced": {
        "path": join(_BASE_DATA_PATH, "ILSVRC12_32"),
        "resize": None,
        "pad": 4,
        "crop": 32,
        "flip": True,
        "normalize": ((0.481, 0.457, 0.408), (0.260, 0.253, 0.268)),
        "class_order": [
            472,
            46,
            536,
            806,
            547,
            976,
            662,
            12,
            955,
            651,
            492,
            80,
            999,
            996,
            788,
            471,
            911,
            907,
            680,
            126,
            42,
            882,
            327,
            719,
            716,
            224,
            918,
            647,
            808,
            261,
            140,
            908,
            833,
            925,
            57,
            388,
            407,
            215,
            45,
            479,
            525,
            641,
            915,
            923,
            108,
            461,
            186,
            843,
            115,
            250,
            829,
            625,
            769,
            323,
            974,
            291,
            438,
            50,
            825,
            441,
            446,
            200,
            162,
            373,
            872,
            112,
            212,
            501,
            91,
            672,
            791,
            370,
            942,
            172,
            315,
            959,
            636,
            635,
            66,
            86,
            197,
            182,
            59,
            736,
            175,
            445,
            947,
            268,
            238,
            298,
            926,
            851,
            494,
            760,
            61,
            293,
            696,
            659,
            69,
            819,
            912,
            486,
            706,
            343,
            390,
            484,
            282,
            729,
            575,
            731,
            530,
            32,
            534,
            838,
            466,
            734,
            425,
            400,
            290,
            660,
            254,
            266,
            551,
            775,
            721,
            134,
            886,
            338,
            465,
            236,
            522,
            655,
            209,
            861,
            88,
            491,
            985,
            304,
            981,
            560,
            405,
            902,
            521,
            909,
            763,
            455,
            341,
            905,
            280,
            776,
            113,
            434,
            274,
            581,
            158,
            738,
            671,
            702,
            147,
            718,
            148,
            35,
            13,
            585,
            591,
            371,
            745,
            281,
            956,
            935,
            346,
            352,
            284,
            604,
            447,
            415,
            98,
            921,
            118,
            978,
            880,
            509,
            381,
            71,
            552,
            169,
            600,
            334,
            171,
            835,
            798,
            77,
            249,
            318,
            419,
            990,
            335,
            374,
            949,
            316,
            755,
            878,
            946,
            142,
            299,
            863,
            558,
            306,
            183,
            417,
            64,
            765,
            565,
            432,
            440,
            939,
            297,
            805,
            364,
            735,
            251,
            270,
            493,
            94,
            773,
            610,
            278,
            16,
            363,
            92,
            15,
            593,
            96,
            468,
            252,
            699,
            377,
            95,
            799,
            868,
            820,
            328,
            756,
            81,
            991,
            464,
            774,
            584,
            809,
            844,
            940,
            720,
            498,
            310,
            384,
            619,
            56,
            406,
            639,
            285,
            67,
            634,
            792,
            232,
            54,
            664,
            818,
            513,
            349,
            330,
            207,
            361,
            345,
            279,
            549,
            944,
            817,
            353,
            228,
            312,
            796,
            193,
            179,
            520,
            451,
            871,
            692,
            60,
            481,
            480,
            929,
            499,
            673,
            331,
            506,
            70,
            645,
            759,
            744,
            459,
        ],
    },
    "imagenet_subset_kaggle": {
        "path": join(datasets_root, "seed_1993_subset_100_imagenet"),
        "test_resize": 256,
        "crop": 224,
        "flip": True,
        "normalize": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        "lbl_order": [
            "n03710193",
            "n03089624",
            "n04152593",
            "n01806567",
            "n02107574",
            "n04409515",
            "n04599235",
            "n03657121",
            "n03942813",
            "n04026417",
            "n02640242",
            "n04591157",
            "n01689811",
            "n07614500",
            "n03085013",
            "n01882714",
            "n02112706",
            "n04266014",
            "n02786058",
            "n02526121",
            "n03141823",
            "n03775071",
            "n04074963",
            "n01531178",
            "n04428191",
            "n02096177",
            "n02091467",
            "n02971356",
            "n02116738",
            "n03017168",
            "n02002556",
            "n04355933",
            "n02840245",
            "n04371430",
            "n01774384",
            "n03223299",
            "n04399382",
            "n02088094",
            "n02033041",
            "n02814860",
            "n04604644",
            "n02669723",
            "n03884397",
            "n03250847",
            "n04153751",
            "n03016953",
            "n02101388",
            "n01914609",
            "n02128385",
            "n03075370",
            "n02363005",
            "n09468604",
            "n02011460",
            "n03785016",
            "n12267677",
            "n12768682",
            "n12620546",
            "n01537544",
            "n03532672",
            "n03691459",
            "n02749479",
            "n02105056",
            "n02279972",
            "n04442312",
            "n02107908",
            "n02229544",
            "n04525305",
            "n02102318",
            "n15075141",
            "n01514668",
            "n04550184",
            "n02115913",
            "n02094258",
            "n07892512",
            "n01984695",
            "n01990800",
            "n02948072",
            "n02112137",
            "n02123597",
            "n02917067",
            "n03485407",
            "n03759954",
            "n02280649",
            "n03290653",
            "n01775062",
            "n03527444",
            "n03967562",
            "n01744401",
            "n02128757",
            "n01729322",
            "n03000247",
            "n02950826",
            "n03891332",
            "n07831146",
            "n02536864",
            "n03697007",
            "n02120079",
            "n02951585",
            "n03109150",
            "n02168699",
        ],
        "class_order": list(range(100)),
    },
    "tiny_scaled_imnet": {
        "path": join(datasets_root, "tiny-imagenet-200"),
        "resize": 32,
        "test_resize": 32,
        "crop": 32,
        "flip": True,
        "normalize": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # ((0.4802, 0.4480, 0.3975), (0.2770, 0.2691, 0.2821))
        "lbl_order": [
            "n02124075",
            "n04067472",
            "n04540053",
            "n04099969",
            "n07749582",
            "n01641577",
            "n02802426",
            "n09246464",
            "n07920052",
            "n03970156",
            "n03891332",
            "n02106662",
            "n03201208",
            "n02279972",
            "n02132136",
            "n04146614",
            "n07873807",
            "n02364673",
            "n04507155",
            "n03854065",
            "n03838899",
            "n03733131",
            "n01443537",
            "n07875152",
            "n03544143",
            "n09428293",
            "n03085013",
            "n02437312",
            "n07614500",
            "n03804744",
            "n04265275",
            "n02963159",
            "n02486410",
            "n01944390",
            "n09256479",
            "n02058221",
            "n04275548",
            "n02321529",
            "n02769748",
            "n02099712",
            "n07695742",
            "n02056570",
            "n02281406",
            "n01774750",
            "n02509815",
            "n03983396",
            "n07753592",
            "n04254777",
            "n02233338",
            "n04008634",
            "n02823428",
            "n02236044",
            "n03393912",
            "n07583066",
            "n04074963",
            "n01629819",
            "n09332890",
            "n02481823",
            "n03902125",
            "n03404251",
            "n09193705",
            "n03637318",
            "n04456115",
            "n02666196",
            "n03796401",
            "n02795169",
            "n02123045",
            "n01855672",
            "n01882714",
            "n02917067",
            "n02988304",
            "n04398044",
            "n02843684",
            "n02423022",
            "n02669723",
            "n04465501",
            "n02165456",
            "n03770439",
            "n02099601",
            "n04486054",
            "n02950826",
            "n03814639",
            "n04259630",
            "n03424325",
            "n02948072",
            "n03179701",
            "n03400231",
            "n02206856",
            "n03160309",
            "n01984695",
            "n03977966",
            "n03584254",
            "n04023962",
            "n02814860",
            "n01910747",
            "n04596742",
            "n03992509",
            "n04133789",
            "n03937543",
            "n02927161",
            "n01945685",
            "n02395406",
            "n02125311",
            "n03126707",
            "n04532106",
            "n02268443",
            "n02977058",
            "n07734744",
            "n03599486",
            "n04562935",
            "n03014705",
            "n04251144",
            "n04356056",
            "n02190166",
            "n03670208",
            "n02002724",
            "n02074367",
            "n04285008",
            "n04560804",
            "n04366367",
            "n02403003",
            "n07615774",
            "n04501370",
            "n03026506",
            "n02906734",
            "n01770393",
            "n04597913",
            "n03930313",
            "n04118538",
            "n04179913",
            "n04311004",
            "n02123394",
            "n04070727",
            "n02793495",
            "n02730930",
            "n02094433",
            "n04371430",
            "n04328186",
            "n03649909",
            "n04417672",
            "n03388043",
            "n01774384",
            "n02837789",
            "n07579787",
            "n04399382",
            "n02791270",
            "n03089624",
            "n02814533",
            "n04149813",
            "n07747607",
            "n03355925",
            "n01983481",
            "n04487081",
            "n03250847",
            "n03255030",
            "n02892201",
            "n02883205",
            "n03100240",
            "n02415577",
            "n02480495",
            "n01698640",
            "n01784675",
            "n04376876",
            "n03444034",
            "n01917289",
            "n01950731",
            "n03042490",
            "n07711569",
            "n04532670",
            "n03763968",
            "n07768694",
            "n02999410",
            "n03617480",
            "n06596364",
            "n01768244",
            "n02410509",
            "n03976657",
            "n01742172",
            "n03980874",
            "n02808440",
            "n02226429",
            "n02231487",
            "n02085620",
            "n01644900",
            "n02129165",
            "n02699494",
            "n03837869",
            "n02815834",
            "n07720875",
            "n02788148",
            "n02909870",
            "n03706229",
            "n07871810",
            "n03447447",
            "n02113799",
            "n12267677",
            "n03662601",
            "n02841315",
            "n07715103",
            "n02504458",
        ],
    },
    "domainnet": {
        "path": join(datasets_root, "domainnet"),
        "resize": 256,
        "crop": 224,
        "flip": True,
        "normalize": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    },
}

# Add missing keys:
for dset in dataset_config.keys():
    for k in [
        "resize",
        "test_resize",
        "pad",
        "crop",
        "normalize",
        "class_order",
        "extend_channel",
    ]:
        if k not in dataset_config[dset].keys():
            dataset_config[dset][k] = None
    if "flip" not in dataset_config[dset].keys():
        dataset_config[dset]["flip"] = False
