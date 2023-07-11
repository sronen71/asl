import json

from .config import CFG


def get_char_dict(config=CFG):
    char_dict_file = f"{config.input_path}/character_to_prediction_index.json"
    with open(char_dict_file) as f:
        char_dict = json.load(f)
    char_dict["P"] = 59
    char_dict["SOS"] = 60
    char_dict["EOS"] = 61
    return char_dict


class Constants:
    ROWS_PER_FRAME = 543
    MAX_STRING_LEN = 50
    INPUT_PAD = -100.0
    char_dict = get_char_dict()
    LABEL_PAD = char_dict["P"]
    inv_dict = {v: k for k, v in char_dict.items()}
    NOSE = [1, 2, 98, 327]
    LIP = [
        0,
        61,
        185,
        40,
        39,
        37,
        267,
        269,
        270,
        409,
        291,
        146,
        91,
        181,
        84,
        17,
        314,
        405,
        321,
        375,
        78,
        191,
        80,
        81,
        82,
        13,
        312,
        311,
        310,
        415,
        95,
        88,
        178,
        87,
        14,
        317,
        402,
        318,
        324,
        308,
    ]

    REYE = [
        33,
        7,
        163,
        144,
        145,
        153,
        154,
        155,
        133,
        246,
        161,
        160,
        159,
        158,
        157,
        173,
    ]
    LEYE = [
        263,
        249,
        390,
        373,
        374,
        380,
        381,
        382,
        362,
        466,
        388,
        387,
        386,
        385,
        384,
        398,
    ]

    LHAND = list(range(468, 489))
    RHAND = list(range(522, 543))

    LNOSE = [98]
    RNOSE = [327]

    LLIP = [84, 181, 91, 146, 61, 185, 40, 39, 37, 87, 178, 88, 95, 78, 191, 80, 81, 82]
    RLIP = [
        314,
        405,
        321,
        375,
        291,
        409,
        270,
        269,
        267,
        317,
        402,
        318,
        324,
        308,
        415,
        310,
        311,
        312,
    ]
    POSE = [500, 502, 504, 501, 503, 505, 512, 513]
    LPOSE = [513, 505, 503, 501]
    RPOSE = [512, 504, 502, 500]

    POINT_LANDMARKS_PARTS = [LHAND, RHAND, LLIP, RLIP, LPOSE, RPOSE, NOSE, REYE, LEYE]
    # POINT_LANDMARKS_PARTS = [LHAND, RHAND, NOSE]
    POINT_LANDMARKS = [item for sublist in POINT_LANDMARKS_PARTS for item in sublist]
    parts = {
        "LLIP": LLIP,
        "RLIP": RLIP,
        "LHAND": LHAND,
        "RHAND": RHAND,
        "LPOSE": LPOSE,
        "RPOSE": RPOSE,
        "LNOSE": LNOSE,
        "RNOSE": RNOSE,
        "REYE": REYE,
        "LEYE": LEYE,
    }

    LANDMARK_INDICES = {}  # type: ignore
    for part in parts:
        LANDMARK_INDICES[part] = []
        for landmark in parts[part]:
            if landmark in POINT_LANDMARKS:
                LANDMARK_INDICES[part].append(POINT_LANDMARKS.index(landmark))

    CENTER_LANDMARKS = LNOSE + RNOSE
    CENTER_INDICES = LANDMARK_INDICES["LNOSE"] + LANDMARK_INDICES["RNOSE"]

    NUM_NODES = len(POINT_LANDMARKS)
    CHANNELS = 6 * NUM_NODES
